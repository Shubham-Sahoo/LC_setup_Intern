import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from sacred import cli_option

from data.synthia import Frame
from devices.synthia import EvalEpIndexIterator
from envs.base_env import BaseEnv
from policies.baseline import BaselineActor
from policies.nn.residual.actor import NNResidualActor
import utils

from kittiviewer.backend import vizstream

app = Flask("second")
CORS(app)


@cli_option('-l', '--load_weights_from_run')
def load_weights_from_run(args, run):
    run.info["run_id"] = int(args)


@cli_option('-H', '--horizon')
def horizon(args, run):
    run.info["horizon"] = int(args)


ex = utils.Experiment("backend", additional_cli_options=[load_weights_from_run, horizon])


class SecondBackend:
    def __init__(self):
        self.env = BaseEnv(debug=False,
                           split="mini_train",
                           cam=False,
                           pc=True,
                           preload="horizon" not in ex.info,
                           progress=True)
        self.actor = NNResidualActor(thetas=self.env.thetas,
                                     base_policy=BaselineActor())

        self.ep_index_iterator = EvalEpIndexIterator(self.env.gt_state_device)

        if "run_id" in ex.info:
            print(f"Loading weights from run {ex.info['run_id']} ...")
            file = utils.get_sacred_artifact_from_mongodb(run_id=ex.info["run_id"], name="residual_weights")
            state_dict = torch.load(file)
            self.actor.network.load_state_dict(state_dict)

        # Register device streams for visualization.
        vizstream(app, self.env.gt_state_device.stream, astype="scene_cloud")
        vizstream(app, self.env.light_curtain.stream,   astype="lc_cloud")

    def run_simulation(self, idx):
        horizon = ex.info["horizon"] if "horizon" in ex.info else float('inf')
        for vid, start in self.ep_index_iterator.single_video_iter(idx):
            ep_len = 0
            obs, done, info = self.env.reset(vid, start)
            self.actor.reset()

            # simulation loop
            while not done:
                if type(obs) == Frame:  # state
                    act, logp_a, control, p_info = self.actor.init_action(obs)  # act: (C,)
                else:  # observation
                    act, logp_a, control, p_info = self.actor.step(obs)  # act: (C,)

                # max episode length
                if control:
                    ep_len += 1
                    if ep_len == horizon:
                        break

                # take step
                obs, rew, done, e_info = self.env.step(act, self.actor.latency, demonstration=False)

    def stop_simulation(self):
        raise NotImplementedError


BACKEND = None


@app.route('/api/run_simulation', methods=['GET', 'POST'])
def run_simulation():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    video_idx = instance["video_idx"]
    enable_int16 = instance["enable_int16"]

    BACKEND.run_simulation(video_idx)

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}

    BACKEND.stop_simulation()

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@ex.automain
def main(port=16666):
    global BACKEND
    BACKEND = SecondBackend()

    app.run(host='127.0.0.1', threaded=True, port=port)
