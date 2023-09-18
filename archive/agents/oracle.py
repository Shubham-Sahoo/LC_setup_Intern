import matplotlib.pyplot as plt
import numpy as np

from devices.device import Device
from utils import valid_curtain_behind_frontier
from planner import PlannerRT


class OracleAgent(Device):
    def __init__(self, env, manager, gt_state_device, light_curtain, latency=39.99, debug=False):
        super().__init__(env, capacity=1)
        self.latency = latency
        self._debug = debug

        self.manager = manager
        self.gt_state_device = gt_state_device
        self.light_curtain = light_curtain

        # options
        self._MAX_RANGE = 20
        self._MIN_RANGE = 1
        self._NODES_PER_RAY = 120  # 0.16m apart

        self.ranges = np.arange(1, self._NODES_PER_RAY + 1) / self._NODES_PER_RAY * self._MAX_RANGE  # (R,)
        self.thetas = self.light_curtain.lc_device.thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]
        self.R = len(self.ranges)
        self.C = self.light_curtain.lc_device.CAMERA_PARAMS["width"]  # number of camera rays
        self.planner_min = PlannerRT(self.light_curtain.lc_device, self.ranges, self.C, maximize=False)

    def process(self):
        while True:
            ############################################################################################################
            # Compute oracle front curtain behind GT safety envelope
            ############################################################################################################

            se_gt = self.gt_state_device.get_current_gt_safety_envelope()  # (C,)
            se_pred = valid_curtain_behind_frontier(self.planner_min, se_gt)  # (C,)

            ############################################################################################################
            # Timeout for computations
            ############################################################################################################

            # yield self.env.timeout(self.latency)

            ############################################################################################################
            # Call manager
            ############################################################################################################

            yield self.env.process(self.manager.service(se_pred))  # self.f_curtain is always the estimated se

            yield self.env.timeout(self.latency)

    def reset(self, env):
        super().reset(env)
