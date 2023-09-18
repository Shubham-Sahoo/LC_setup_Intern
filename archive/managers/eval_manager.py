import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict

from devices.managers.manager import Manager
from utils import valid_curtain_behind_frontier
from planner import PlannerRT

class EvalManager(Manager):
    """Manager that computes GT and SS rewards and stores them in a list"""
    def __init__(self, env, gt_state_device, light_curtain, oracle=False, debug=False):
        super().__init__(env, gt_state_device, light_curtain, debug)

        self._oracle = oracle  # whether to compute rewards of the oracle

        self.BINS_GT        = 200  # number of bins in histograms for gt_reward
        self.BINS_SS        = 200  # number of bins in histograms for ss_reward
        self._MAX_RANGE     = 20
        self._NODES_PER_RAY = 120  # 0.16m apart

        if self._oracle:
            self.ranges = np.arange(1, self._NODES_PER_RAY + 1) / self._NODES_PER_RAY * self._MAX_RANGE  # (R,)
            self.planner_min = PlannerRT(self.light_curtain.lc_device, self.ranges, self.C, maximize=False)

        self.gt_reward_stream = []
        self.ss_reward_stream = []
    
    def reset(self, env):
        """Overlead device.reset to reset reward streams too"""
        super().reset(env)
        self.gt_reward_stream.clear()
        self.ss_reward_stream.clear()

    ####################################################################################################################
    #region: Functions to compute rewards
    ####################################################################################################################

    def gt_reward(self, lc_data):
        """
        Computes supervised reward for a single Manager.service call.

        Args:
            lc_data:
                {
                    "f_lc": Return of the front light curtain, placed using "se_pred".
                        {
                            "lc_ranges": (np.ndarray, dtype=np.float32, shape=(C,)) ranges per camera ray of the light
                                         curtain that was actually placed by the light curtain device.
                            "lc_image": (np.ndarray, dtype=np.float32, shape=(H, C, 4)) lc image.
                                        Axis 2 corresponds to (x, y, z, i):
                                                - x : x in cam frame.
                                                - y : y in cam frame.
                                                - z : z in cam frame.
                                                - i : intensity of LC cloud, lying in [0, 255].
                            "lc_cloud": (np.ndarray, dtype=np.float32, shape=(N, 4)) lc cloud.
                                        Axis 2 corresponds to (x, y, z, i):
                                                - x : x in velo frame.
                                                - y : y in velo frame.
                                                - z : z in velo frame.
                                                - i : intensity of LC cloud, lying in [0, 1].
                        }
                    "r_lc": Return of the random light curtain that is placed behind "se_pred".
                            Format is the same as "f_lc".
                }
        
        Appends: hists
            {
                "agent": (np.ndarray, dtype=np.float32, shape=(BINS_GT,)) histogram of the ratio of distance between the
                         predicted safety envelope (se_pred) and the true safety envelope (se_gt), for the agent.
                "oracle": (optional) same as above, but for oracle.
            }
        """
        # se_pred
        # we don't care about r_lc here
        se_pred_agent = lc_data["f_lc"]["lc_ranges"]  # (C,)
        if se_pred_agent.max() >= 21:
            print("ASSERTION ERROR: se_pred.max() has exceeded 20")
            import pdb; pdb.set_trace()

        # se_gt
        se_gt = self.gt_state_device.get_current_gt_safety_envelope()  # (C,)

        if self._debug:
            design_pts = self._design_pts_from_ranges(se_gt)  # (C, 2)
            plt.plot(design_pts[:, 0], design_pts[:, 1], c='b')
            design_pts = self._design_pts_from_ranges(se_pred_agent)  # (C, 2)
            plt.plot(design_pts[:, 0], design_pts[:, 1], c='r')
            plt.ylim(0, 21)
            plt.title("blue: se_gt, red: se_pred")
            plt.show()
        
        def get_ratio_hist(se_pred):
            ratio = (se_pred / se_gt).clip(min=0, max=2)  # (C,)
            hist = np.histogram(ratio, range=(0, 2), bins=self.BINS_GT)[0]  # (BINS_GT,)
            return hist
        
        hist_agent = get_ratio_hist(se_pred_agent)  # (BINS_GT,)
        hists = {"agent": hist_agent}

        if self._oracle:
            # plan a curtain right behind the safety envelope
            se_pred_oracle = valid_curtain_behind_frontier(self.planner_min, se_gt, debug=self._debug)  # (C,)
            hist_oracle = get_ratio_hist(se_pred_oracle)  # (BINS_GT,)
            hists["oracle"] = hist_oracle
        
        return hists


    def ss_reward(self, lc_data):
        """
        Computes self-supervised reward for a single Manager.service call.
        Histograms are computed for max intensities along each camera column.

        Args:
            lc_data:
                {
                    "f_lc": Return of the front light curtain, placed using "se_pred".
                        {
                            "lc_ranges": (np.ndarray, dtype=np.float32, shape=(C,)) ranges per camera ray of the light
                                         curtain that was actually placed by the light curtain device.
                            "lc_image": (np.ndarray, dtype=np.float32, shape=(H, C, 4)) lc image.
                                        Axis 2 corresponds to (x, y, z, i):
                                                - x : x in cam frame.
                                                - y : y in cam frame.
                                                - z : z in cam frame.
                                                - i : intensity of LC cloud, lying in [0, 255].
                            "lc_cloud": (np.ndarray, dtype=np.float32, shape=(N, 4)) lc cloud.
                                        Axis 2 corresponds to (x, y, z, i):
                                                - x : x in velo frame.
                                                - y : y in velo frame.
                                                - z : z in velo frame.
                                                - i : intensity of LC cloud, lying in [0, 1].
                        }
                    "r_lc": Return of the random light curtain that is placed behind "se_pred".
                            Format is the same as "f_lc".
                }
        
        Appends: hists
            {
                "agent_f": (np.ndarray, dtype=np.float32, shape=(BINS_SS,)) histogram of front curtain rets.
                "agent_r": (np.ndarray, dtype=np.float32, shape=(BINS_SS,)) histogram of random curtain rets.
                "oracle_f": (optional) same as "agent_f" but for oracle.
            }
        """
        hists = {}

        def get_max_intensities_hist_from_image(image_):  # image_ is (H, C, 4)
            return_ = image_[:, :, 3].copy()  # (H, C)

            # mask out pixels that are below GROUND_HEIGHT (note that cam_y points downwards)
            return_[-image_[:, :, 1] < self.GROUND_HEIGHT] = 0
            return_ /= 255.0  # rescale intensities to [0, 1]

            # take maximum intensity along camera columns
            return_ = return_.max(axis=0)  # (C,)
            
            # convert to histogram
            hist_ = np.histogram(return_, range=(0, 1), bins=self.BINS_SS)[0]  # (BINS_SS,)

            return hist_

        ################################################################################################################
        # Compute histogram of agent's front curtain return
        ################################################################################################################
        
        af_image = lc_data["f_lc"]["lc_image"]
        hists["agent_f"] = get_max_intensities_hist_from_image(af_image)

        ################################################################################################################
        # Compute histogram of agents' random curtain return
        ################################################################################################################
        
        ar_image = lc_data["r_lc"]["lc_image"]
        hists["agent_r"] = get_max_intensities_hist_from_image(ar_image)

        if self._oracle:

            # get f_curtain for oracle = curtain that lies behind se_gt
            se_gt = self.gt_state_device.get_current_gt_safety_envelope()  # (C,)
            of_curtain = valid_curtain_behind_frontier(self.planner_min, se_gt, debug=self._debug)  # (C,)

            ############################################################################################################
            # Compute histogram of oracle's front curtain return
            ############################################################################################################
            
            design_pts = self._design_pts_from_ranges(of_curtain)
            of_image, of_cloud = self.light_curtain.get_lc_return_from_current_state(design_pts)
            assert of_image.shape[1] == self.C
            hists["oracle_f"] = get_max_intensities_hist_from_image(of_image)

            ############################################################################################################
            # Compute histogram of oracle's random curtain return
            ############################################################################################################
            
            or_curtain = self._random_curtain(of_curtain)
            design_pts = self._design_pts_from_ranges(or_curtain)
            or_image, or_cloud =self.light_curtain.get_lc_return_from_current_state(design_pts)
            assert or_image.shape[1] == self.C
            hists["oracle_r"] = get_max_intensities_hist_from_image(or_image)

        return hists

    #endregion
    ####################################################################################################################

    def service(self, se_pred):
        """
        Args:
            se_pred: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray of the estimate of the safety
                        envelope. This will be used to place the front curtain.
                        NOTE: it is required that se_pred correspond to a valid light curtain! Any device calling this
                        service must ensure this.
        Publishes:
            {
                "f_lc": Return of the front light curtain, placed using "se_pred".
                    {
                        "lc_ranges": (np.ndarray, dtype=np.float32, shape=(C,)) ranges per camera ray of the light
                                        curtain that was actually placed by the light curtain device.
                        "lc_image": (np.ndarray, dtype=np.float32, shape=(H, C, 4)) lc image.
                                    Axis 2 corresponds to (x, y, z, i):
                                            - x : x in cam frame.
                                            - y : y in cam frame.
                                            - z : z in cam frame.
                                            - i : intensity of LC cloud, lying in [0, 255].
                        "lc_cloud": (np.ndarray, dtype=np.float32, shape=(N, 4)) lc cloud.
                                    Axis 2 corresponds to (x, y, z, i):
                                            - x : x in velo frame.
                                            - y : y in velo frame.
                                            - z : z in velo frame.
                                            - i : intensity of LC cloud, lying in [0, 1].
                    }
                "r_lc": Return of the random light curtain that is placed behind "se_pred".
                        Format is the same as "f_lc".
            }
        """
        data = yield self.env.process(self.get_lc_data(se_pred))
        
        # Compute and append gt reward to stream
        gt_r = self.gt_reward(data)
        self.gt_reward_stream.append(edict(timestamp=self.env.now, data=gt_r))

        # Compute and append ss reward to stream
        ss_r = self.ss_reward(data)
        self.ss_reward_stream.append(edict(timestamp=self.env.now, data=ss_r))

        self.publish(data)
