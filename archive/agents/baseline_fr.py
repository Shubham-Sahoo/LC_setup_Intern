import matplotlib.pyplot as plt
import numpy as np

from devices.device import Device
from planner import PlannerRT
from utils import valid_curtain_behind_frontier

class BaselineFrontRandomAgent(Device):
    def __init__(self, env, manager, light_curtain, latency=10, debug=False):
        super().__init__(env, capacity=1)
        self.latency = latency
        self._debug = debug

        self.manager = manager
        self.light_curtain = light_curtain

        # options
        self._MAX_RANGE             = 20
        self._MIN_RANGE             = 1
        self._NODES_PER_RAY         = 120  # 0.16m apart
        self._EXPANSION             = 0.3
        self._RECESSION_F           = 0.4  # recession for front curtain
        self._RECESSION_R           = 1.0  # recession for random curtain
        self._SMOOTHNESS            = 0.05
        self._LC_INTENSITY_THRESH_F = 200
        self._LC_INTENSITY_THRESH_R = 200

        self.ranges = np.arange(1, self._NODES_PER_RAY + 1) / self._NODES_PER_RAY * self._MAX_RANGE  # (R,)
        self.thetas = self.light_curtain.lc_device.thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]
        self.R = len(self.ranges)
        self.C = self.light_curtain.lc_device.CAMERA_PARAMS["width"]  # number of camera rays
        self.planner_min = PlannerRT(self.light_curtain.lc_device, self.ranges, self.C, maximize=False)
        
        # frontier is the estimated envelope, represented as range per camera ray
        self.f_curtain = self._MIN_RANGE * np.ones([self.C], dtype=np.float32)  # (C,) initial frontier is 1m
    
    @property
    def GROUND_HEIGHT(self):
        return self.light_curtain.GROUND_HEIGHT
    
    ####################################################################################################################
    #region: Helper functions
    ####################################################################################################################

    def _hits_from_lc_image(self, lc_image, ithresh):
        """
        Args:
            lc_image: (np.ndarray, dtype=np.float32, shape=(H, C, 4)) lc image.
                      Axis 2 corresponds to (x, y, z, i):
                        - x : x in cam frame.
                        - y : y in cam frame.
                        - z : z in cam frame.
                        - i : intensity of LC cloud, lying in [0, 255].
            ithresh: (float) intensity threshold in [0, 255] above which returns will be considered as hits.
        Returns:
            hits: (np.ndarray, dtype=np.bool, shape=(C,)) whether there is a hit or not for every camera column.
        """
        hits = np.ones(lc_image.shape[:2], dtype=np.bool)  # (H, C)
            
        # mask out NaN values
        hits[np.isnan(lc_image).any(axis=2)] = 0  # (H, C)

        # mask out pixels below intensity threshold
        hits[lc_image[:, :, 3] < ithresh] = 0

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
            # Call manager
            ############################################################################################################
            
            yield self.env.process(self.manager.service(self.f_curtain))  # self.f_curtain is always the estimated se
            
            f_lc = self.manager.stream[-1]["data"]["f_lc"]
            self.f_curtain, f_lc_image = f_lc["lc_ranges"].copy(), f_lc["lc_image"]

            r_lc = self.manager.stream[-1]["data"]["r_lc"]
            r_curtain, r_lc_image = r_lc["lc_ranges"].copy(), r_lc["lc_image"]

            # Get hits
            f_hits = self._hits_from_lc_image(f_lc_image, ithresh=self._LC_INTENSITY_THRESH_F)  # (C,)
            r_hits = self._hits_from_lc_image(r_lc_image, ithresh=self._LC_INTENSITY_THRESH_R)  # (C,)

            ############################################################################################################
            # Expand+Receed+Smooth frontier
            ############################################################################################################

            n_hits = ~ (f_hits | r_hits)  # (C,) no hit from either front or random curtain

            # Expansion
            self.f_curtain[n_hits] += self._EXPANSION

            # Recession
            self.f_curtain[f_hits] -= self._RECESSION_F
            self.f_curtain[r_hits] = r_curtain[r_hits] - self._RECESSION_R

            # Enforce smoothness
            for i in range(len(self.f_curtain)):
                for j in range(len(self.f_curtain)):
                    if self.f_curtain[i] > self.f_curtain[j]:
                        self.f_curtain[i] = min(self.f_curtain[i], self.f_curtain[j] + self._SMOOTHNESS * abs(i-j))
            
            # Curtain should lie between max and min ranges.
            self.f_curtain = self.f_curtain.clip(min=self._MIN_RANGE, max=self._MAX_RANGE)
            
            # Validate curtain
            self.f_curtain = valid_curtain_behind_frontier(self.planner_min, self.f_curtain, debug=self._debug)  # (C,)

            ############################################################################################################
            # Timeout for computations
            ############################################################################################################

            yield self.env.timeout(self.latency)

    
    def reset(self, env):
        super().reset(env)
        self.f_curtain = self._MIN_RANGE * np.ones([self.C], dtype=np.float32)  # (C,) initial frontier is 1m
