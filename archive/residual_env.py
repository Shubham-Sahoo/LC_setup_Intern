import gym
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from envs.base_env import BaseEnv
from policies.actor import Actor
from policies.features import Featurizer, Features, CurtainInfo


class ResidualEnv(BaseEnv):
    """
    A wrapper on BaseEnv for residual policies.
    It takes a base policy as input, and creates an environment for a residual policy acting that uses the base policy.
    """
    def __init__(self,
                 base_policy: Actor,
                 debug: bool = False,
                 **dataset_kwargs):
        """
        Args:
            dataset_split (str): dataset split.
            base_policy (Actor): a base policy that the residual policy policy uses.
            debug (bool): whether to run the environment in debug mode.
            dataset_kwargs (dict): kwargs to pass to SynthiaGTState
        """
        super().__init__(debug=debug, **dataset_kwargs)

        self.base_policy = base_policy
        self.featurizer = Featurizer(self.thetas)

        # Spaces
        obs_dim = self.featurizer.feat_dim
        act_dim = self.featurizer.C
        self.observation_space = gym.spaces.Box(low=-20 * np.ones(obs_dim, dtype=np.float32),
                                                high=20 * np.ones(obs_dim, dtype=np.float32))
        self.action_space = gym.spaces.Box(low=-20 * np.ones(act_dim, dtype=np.float32),
                                           high=20 * np.ones(act_dim, dtype=np.float32))

    @dataclass
    class PrevOCA:
        o: BaseEnv.Observation  # observation at time t-1
        c: CurtainInfo          # curtain info from observations time t-1
        a: np.ndarray           # action of base policy at time t-1 using o

    ####################################################################################################################
    # region Env functions
    ####################################################################################################################

    def reset(self,
              idx: int,
              start: int = 0,
              demonstration: bool = False) -> Tuple[np.ndarray, bool, dict]:
        """Resets the state of the environment and returns an initial observation.

        Args:
            idx (int): video id.
            start (int): start frame of video.
            demonstration (bool): whether to provide a demonstration for this observation or not.

        Returns:
            obs (np.ndarray): observation.
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # initial state and action
        s0, done, info0 = super().reset(idx, start)
        if done:
            return np.empty(0, dtype=np.float32), True, {}
        a0 = self.base_policy.init_action(s0)

        # first step
        o1, rew, done, info1 = super().step(a0, self.base_policy.latency)
        if done:
            return np.empty(0, dtype=np.float32), True, {}
        c1 = self.featurizer.obs_to_curtain_info(o1)
        a1, _ = self.base_policy.step(o1)

        # second step
        o2, rew, done, info2 = super().step(a1, self.base_policy.latency, demonstration=demonstration)
        if done:
            return np.empty(0, dtype=np.float32), True, {}
        c2 = self.featurizer.obs_to_curtain_info(o2)
        a2, _ = self.base_policy.step(o2)

        f2 = self.featurizer.cinfos2feats(c1, c2, a2)
        self.prev_oca = self.PrevOCA(o2, c2, a2)

        info2["base_action"] = a2
        if demonstration:  # compute residual action
            d2 = info2["demonstration"] - a2
            info2["demonstration"] = d2

        return f2.to_numpy(), False, info2

    def step(self,
             action: np.ndarray,
             latency: float = 0.0,
             demonstration: bool = False) -> Tuple[np.ndarray, Optional[float], bool, dict]:
        """
        Args:
            action (np.ndarray, dtype=np.float32, shape=(C,)): Ranges of the light curtain (residual action).
            latency (float): Time take by the policy to compute this action, used by the simpy simulation.
            demonstration (bool): whether to provide a demonstration for this observation or not.

        Returns:
            observation (np.ndarray): featurized observation.
            reward (float) : Amount of reward returned after previous action
            done (bool): Whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        o_prev, c_prev, base_a_prev = self.prev_oca.o, self.prev_oca.c, self.prev_oca.a

        # add residual action
        a_prev = base_a_prev + action

        # total latency
        total_latency = self.base_policy.latency + latency

        # step through base env
        o_next, rew, done, info_next = super().step(a_prev, latency=total_latency, demonstration=demonstration)
        if done:
            return np.empty(0, dtype=np.float32), rew, done, {}

        # run base policy
        base_a_next, base_logp_a_next = self.base_policy.step(o_next)

        # compute features
        c_next = self.featurizer.obs_to_curtain_info(o_next)
        f_next = self.featurizer.cinfos2feats(c_prev,
                                              c_next,
                                              base_a_next)

        # store next oca
        self.prev_oca = self.PrevOCA(o_next, c_next, base_a_next)

        info_next["base_action"] = base_a_next
        if demonstration:
            d_next = info_next["demonstration"] - base_a_next
            info_next["demonstration"] = d_next

        return f_next.to_numpy(), rew, False, info_next

    def render(self, mode='human'):
        raise NotImplementedError

    # endregion
    ####################################################################################################################
