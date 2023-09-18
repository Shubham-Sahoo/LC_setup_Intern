"""Supervised Learning Training Script"""
import itertools
import gym
import torch
import torch.distributions as td
import numpy as np
from sacred import SETTINGS, cli_option
from sacred.utils import apply_backspaces_and_linefeeds
import tqdm

from common.buffers import ImitationRolloutBuffer
from data.synthia import Frame
from devices.synthia import TrainEpIndexIterator, EvalEpIndexIterator
from envs.base_env import BaseEnv
from eval.gt import EvaluatorGT
from policies.actor import Actor
from policies.baseline import BaselineActor
from policies.nn.residual.actor import NNResidualActor
from policies.nn.absolute.actor import NNAbsoluteActor
import utils

# sacred
SETTINGS.DISCOVER_SOURCES = 'sys'
SETTINGS.CAPTURE_MODE = 'sys'

ex = utils.Experiment("Supervised Training")
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.add_source_file("sl_train.py")
ex.add_config("config/sl_train.yaml")


def loss_l2(pi: td.Distribution,
            dem: np.ndarray) -> torch.Tensor:
    """
    Args:
        pi (td.Distribution, dtype=torch.float32, BS=(B,), ES=(C,)): distribution predicted by policy.
        dem (np.ndarray, dtype=np.float32, shape=(B, C)): demonstration.

    Returns:
        loss (torch.Tensor, dtype=np.float32, shape=([])): L2 loss.
    """
    dem = torch.from_numpy(dem)  # (B, C)
    if torch.cuda.is_available():
        dem = dem.cuda()

    mu = pi.mean  # (B, C)
    loss = torch.square(mu - dem).mean()
    return loss


def loss_smooth_clipped_ratio(pi: td.Distribution,
                              dem: np.ndarray,
                              k: float = 25) -> torch.Tensor:
    """
    Args:
        pi (td.Distribution, dtype=torch.float32, BS=(B,), ES=(C,)): distribution predicted by policy.
        dem (np.ndarray, dtype=np.float32, shape=(B, C)): demonstration.
        k (float): loss parameter that controls the smoothness of the loss for r > 1.

    Returns:
        loss (torch.Tensor, dtype=np.float32, shape=([])): smooth clipped ratio loss.
            - smooth_clipped_ratio_loss = min(r, exp(-k(r-1)) where r = predicted_range / gt_range.
            - ideally r >= 0. But if it is < 0, smooth_clipped_ratio_loss < 0. Hence we clamp it to be >= 0.
    """
    dem = torch.from_numpy(dem)  # (B, C)
    if torch.cuda.is_available():
        dem = dem.cuda()

    mu = pi.mean  # (B, C)

    ratio = mu / dem  # (B, C)
    smooth_clipped_ratio = torch.min(ratio, torch.exp(-k * (ratio - 1))).clamp(min=0)  # (B, C)
    loss = -smooth_clipped_ratio.mean()
    return loss


@ex.automain
def sl_train(_config):
    train_env = BaseEnv(split="train",       # 191 videos
                        debug=False, cam=False, pc=False, preload=False, progress=False)
    valid_env = BaseEnv(split="mini_train",  # 12 videos
                        debug=False, cam=False, pc=False, preload=False, progress=False)
    test_env  = BaseEnv(split="mini_test",   # 15 videos
                        debug=False, cam=False, pc=False, preload=False, progress=False)

    b_policy = BaselineActor()
    nn_actor_class = globals()[_config['actor']]  # NNResidualActor or NNAbsoluteActor
    r_policy = nn_actor_class(thetas=train_env.thetas,
                              base_policy=b_policy)

    feat_dim, act_dim = r_policy.feat_dim, len(train_env.thetas)
    feat_space = gym.spaces.Box(low=-20 * np.ones(feat_dim, dtype=np.float32),
                                high=20 * np.ones(feat_dim, dtype=np.float32))
    act_space  = gym.spaces.Box(low=-20 * np.ones(act_dim, dtype=np.float32),
                                high=20 * np.ones(act_dim, dtype=np.float32))
    buffer = ImitationRolloutBuffer(buffer_size=_config["buffer_size"],
                                    observation_space=feat_space,
                                    action_space=act_space)

    train_ep_index_iter = iter(TrainEpIndexIterator(train_env.gt_state_device))
    optimizer = torch.optim.Adam(r_policy.network.parameters(), lr=_config["learning_rate"])

    evaluator = EvaluatorGT()

    loss_fn = globals()[f"loss_{_config['loss']}"]  # loss_l2 or loss_clipped_ratio

    def fill_buffer():
        r_policy.network.eval()
        buffer.reset()
        progress = tqdm.tqdm(total=buffer.buffer_size)
        progress.set_description("Collecting rollouts")

        # infinite loop of episodes
        for ep_index in itertools.count(start=1):
            ep_len = 0
            idx, start = next(train_ep_index_iter)

            # a single episode
            obs, done, e_info = train_env.reset(idx, start, demonstration=True)
            r_policy.reset()
            while not done:
                dem: np.ndarray = e_info["demonstration"]  # (C,)

                if type(obs) == Frame:  # state
                    act, logp_a, control, p_info = r_policy.init_action(obs)   # act: (C,)
                else:  # observation
                    act, logp_a, control, p_info = r_policy.step(obs)  # act: (C,)

                if control:
                    # add feats to buffer and end function if full
                    feat = p_info["features"]  # (F,)
                    buffer.add(feat, act, dem)
                    progress.update(1)
                    if buffer.full:
                        progress.close()
                        return

                    # max episode length
                    ep_len += 1
                    if ep_len == _config["horizon"]:
                        break

                # take step
                obs, rew, done, e_info = train_env.step(act, r_policy.latency, demonstration=True)

    def update_policy():
        r_policy.network.train()
        losses = utils.AverageMeter("Loss", ":.4e")
        for i in range(_config["passes_per_epoch"]):
            for batch in buffer.get(batch_size=_config["batch_size"]):
                pi = r_policy.forward(batch.features)  # BS=(B,) ES=(C,)
                loss = loss_fn(pi, batch.demonstrations)
                losses.update(loss.item())
                loss.backward()
                optimizer.step()
        return losses.avg

    def evaluate(epoch):
        @utils.timer.time_fn(name="evaluation")
        def evaluate_split(env, split):
            def evaluate_policy(policy: Actor,
                                oracle: bool):

                eval_ep_iter_index = iter(EvalEpIndexIterator(env.gt_state_device))

                for idx, start in eval_ep_iter_index:
                    ep_len = 0

                    # a single episode
                    obs, done, e_info = env.reset(idx, start, demonstration=True)
                    policy.reset()
                    while not done:
                        if type(obs) == Frame:  # state
                            p_act, logp_a, control, p_info = policy.init_action(obs)   # p_act: (C,)
                        else:  # observation
                            p_act, logp_a, control, p_info = policy.step(obs)  # p_act: (C,)

                        if control:
                            gt_act: np.ndarray = e_info["demonstration"]  # (C,)
                            actions = [p_act]

                            # oracle action
                            if oracle:
                                o_act: np.ndarray = utils.valid_curtain_behind_frontier(env.planner_min, gt_act)  # (C,)
                                actions.append(o_act)

                            evaluator.add(actions=actions, gt_action=gt_act)

                            # max episode length
                            ep_len += 1
                            if ep_len == _config["horizon"]:
                                break

                        # take step
                        obs, rew, done, e_info = env.step(p_act, policy.latency, demonstration=True)

                return evaluator.metrics(reset=True)

            p_metric, o_metric = evaluate_policy(r_policy, oracle=True)
            b_metric,          = evaluate_policy(b_policy, oracle=False)

            print(f"==========================================================================")
            print(f"Evaluation for Split={split} Epoch={epoch}")
            print(f"--------------------------------------------------------------------------")
            print(f"Residual policy: " + str(p_metric))
            print(f"Baseline policy: " + str(b_metric))
            print(f"Oracle   policy: " + str(o_metric))
            print(f"==========================================================================")

            # log metrics
            for attr, prefix in zip(["safety", "proximity", "accuracy"], ["saf", "prx", "acc"]):
                for metric, suffix in zip([p_metric, b_metric, o_metric], ["r", "b", "o"]):
                    ex.log_scalar(f"{split}_{prefix}_{suffix}", getattr(metric, attr), epoch)

            return p_metric, b_metric, o_metric

        valid_p_metric, _, _ = evaluate_split(valid_env, split="valid")
        _,              _, _ = evaluate_split(test_env,  split="test")

        # return accuracy on valid split
        return valid_p_metric.accuracy

    ####################################################################################################################
    # MAIN LOOP
    ####################################################################################################################
    if _config["eval_every"] > 0:
        evaluate(epoch=0)

    best_valid_accuracy = -float('inf')
    for epoch in range(1, _config["epochs"] + 1):
        # do not fill buffer multiple times if horizon is 1,
        # since training data is stationary.
        if epoch == 1 or _config["horizon"] > 1:
            fill_buffer()

        losses = utils.AverageMeter("Epoch Loss", ":.4e")
        for _ in range(_config["passes_per_epoch"]):
            losses.update(update_policy())
        print(f"EPOCH {epoch}: {str(losses)}")

        if _config["eval_every"] > 0:
            if epoch % _config["eval_every"] == 0 or epoch == _config["epochs"] + 1:
                valid_accuracy = evaluate(epoch=epoch)

                # save network weights
                if valid_accuracy > best_valid_accuracy:
                    best_valid_accuracy = valid_accuracy
                    print(f"Saving network weights for epoch {epoch} with accuracy {100 * valid_accuracy:.2f}%  ...")
                    torch.save(r_policy.network.state_dict(), "/tmp/weights.pth")
                    ex.add_artifact("/tmp/weights.pth", name="residual_weights")

        ex.log_scalar("loss", losses.avg, epoch)
    ####################################################################################################################
