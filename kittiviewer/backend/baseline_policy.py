from flask import Flask, jsonify, request
from flask_cors import CORS

from data.synthia import Frame
from envs.base_env import BaseEnv
from policies.baseline import BaselineActor
import utils

from kittiviewer.backend import vizstream

app = Flask("second")
CORS(app)

ex = utils.Experiment("backend")


class SecondBackend:
    def __init__(self):
        self.env = BaseEnv(debug=False,
                           split="mini_train",
                           cam=False,
                           pc=True,
                           preload=True,
                           progress=True)
        self.actor = BaselineActor()

        # Register device streams for visualization.
        vizstream(app, self.env.gt_state_device.stream, astype="scene_cloud")
        vizstream(app, self.env.light_curtain.stream,   astype="lc_cloud")

    def run_simulation(self, idx):
        obs, done, info = self.env.reset(idx)
        self.actor.reset()

        # simulation loop
        while not done:
            if type(obs) == Frame:  # state
                act, logp_a, control, p_info = self.actor.init_action(obs)  # act: (C,)
            else:  # observation
                act, logp_a, control, p_info = self.actor.step(obs)  # act: (C,)

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
