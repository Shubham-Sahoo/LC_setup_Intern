import matplotlib.pyplot as plt
import numpy as np

from devices.device import Device
from utils import valid_curtain_behind_frontier
from planner import PlannerRT

class SETBaselineFront(Device):
    def __init__(self, env, light_curtain, latency=10, debug=False):
        super().__init__(env, capacity=1)
        self.latency = latency
        self._debug = debug

        self.light_curtain = light_curtain

        # options
        self._MAX_RANGE           = 20
        self._NODES_PER_RAY       = 120  # 0.16m apart
        self._EXPANSION           = 0.3
        self._RECESSION           = 0.4
        self._SMOOTHNESS          = 0.05
        self._LC_INTENSITY_THRESH = 200

        self.ranges = np.arange(1, self._NODES_PER_RAY + 1) / self._NODES_PER_RAY * self._MAX_RANGE  # (R,)
        self.thetas = self.light_curtain.lc_device.thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]
        self.R = len(self.ranges)
        self.C = self.light_curtain.lc_device.CAMERA_PARAMS["width"]  # number of camera rays
        self.planner_min = PlannerRT(self.light_curtain.lc_device, self.ranges, self.C, maximize=False)
        
        # frontier is the estimated envelope, represented as range per camera ray
        self.frontier = np.ones([self.C], dtype=np.float32)  # (C,) initial frontier is 1m
    
    @property
    def GROUND_HEIGHT(self):
        return self.light_curtain.GROUND_HEIGHT
    
    ####################################################################################################################
    #region: Helper functions
    ####################################################################################################################

    def _design_pts_from_ranges(self, ranges):
        """
        Args:
            ranges: (np.ndarray, shape=(C,), dtype=np.float32) range per camera ray
        Returns:
            design_pts: (np.ndarray, shape=(C, 2), dtype=np.float32) design points corresponding to frontier.
                        - Axis 1 channels denote (x, z) in camera frame.
        """
        x = ranges * np.sin(np.deg2rad(self.thetas))
        z = ranges * np.cos(np.deg2rad(self.thetas))
        design_pts = np.hstack([x.reshape(-1, 1), z.reshape(-1, 1)])
        return design_pts

    def _hits_from_lc_image(self, lc_image):
        """
        Args:
            lc_image: (np.ndarray, dtype=np.float32, shape=(H, C, 4)) lc image.
                      Axis 2 corresponds to (x, y, z, i):
                        - x : x in cam frame.
                        - y : y in cam frame.
                        - z : z in cam frame.
                        - i : intensity of LC cloud, lying in [0, 255].
        Returns:
            hits: (np.ndarray, dtype=np.bool, shape=(C,)) whether there is a hit or not for every camera column.
        """
        hits = np.ones(lc_image.shape[:2], dtype=np.bool)  # (H, C)
            
        # mask out NaN values
        hits[np.isnan(lc_image).any(axis=2)] = 0  # (H, C)

        # mask out pixels below intensity threshold
        hits[lc_image[:, :, 3] < self._LC_INTENSITY_THRESH] = 0

        # mask out pixels that are below GROUND_HEIGHT (note that cam_y points downwards)
        hits[-lc_image[:, :, 1] < self.GROUND_HEIGHT] = 0

        # collect hits across camera columns
        hits = hits.any(axis=0)  # (C,)

        return hits

    #endregion
    ####################################################################################################################

    def process(self):
        while True:
            ############################################################################################################
            # Place light curtain and get return
            ############################################################################################################
            
            # get light curtain return from the design points
            design_pts = self._design_pts_from_ranges(self.frontier)
            yield self.env.process(self.light_curtain.service(design_pts))
            lc_image = self.light_curtain.stream[-1].data["lc_image"]  # (H, W, 4)
            assert lc_image.shape[1] == self.C

            ############################################################################################################
            # Compute hits on camera rays from lc image
            ############################################################################################################
            
            hits = self._hits_from_lc_image(lc_image)  # (C,)

            ############################################################################################################
            # Expand+Receed+Smooth frontier
            ############################################################################################################

            # Expansion + Recession
            self.frontier = self.frontier + (1 - hits) * self._EXPANSION
            self.frontier = self.frontier -      hits  * self._RECESSION

            # Enforce smoothness
            for i in range(len(self.frontier)):
                for j in range(len(self.frontier)):
                    if self.frontier[i] > self.frontier[j]:
                        self.frontier[i] = min(self.frontier[i], self.frontier[j] + self._SMOOTHNESS * abs(i-j))
            
            # Validate curtain
            self.frontier = valid_curtain_behind_frontier(self.planner_min, self.frontier, debug=self._debug)  # (C,)

            ############################################################################################################
            # Timeout for computations
            ############################################################################################################

            yield self.env.timeout(self.latency)

    
    def reset(self, env):
        super().reset(env)
        self.frontier = np.ones([self.C], dtype=np.float32)  # (C,) initial frontier is 1m
