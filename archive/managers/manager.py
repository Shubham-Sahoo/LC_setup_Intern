import matplotlib.pyplot as plt
import numpy as np

from devices.device import Device

class Manager(Device):
    """
    A manager provides a service to an agent whenever it wants to report an estimate of the safety envelope.
        - See manager.service() for more detais.
        - The service takes as input "se_pred" which is the agent's estimate of the safety envelope.
          "se_pred" is required to be a valid curtain because the manager will place the front curtain using "se_pred".
          The onus is on the agent to ensure this.
        - The service returns the light curtain return for the front curtain (se_pred) as well as a random curtain
          placed behind the front curtain. The agent need not place front and random curtains. It may however choose to
          place additional curtains.
        - The manager may perform other bookkeeping tasks like computing rewards, evaluation and RL.
          But this will be assumed to take no time in the simulation.
        - The total time taken by the manager's service is twice the time to run the light curtain device.
    """
    def __init__(self, env, gt_state_device, light_curtain, debug=False):
        super().__init__(env, capacity=1)
        self._debug = debug

        self.light_curtain = light_curtain
        self.gt_state_device = gt_state_device  # for supervised rewards
    
    @property
    def thetas(self):
        return self.light_curtain.lc_device.thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]

    @property
    def C(self):
        return self.light_curtain.lc_device.CAMERA_PARAMS["width"]  # number of camera rays

    @property
    def GROUND_HEIGHT(self):
        return self.gt_state_device.GROUND_HEIGHT
    
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

    
    def _validate_curtain_using_lc_image(self, lc_image):
        """Extract the valid curtain that was physically placed via the lc_image

        Args:
            lc_image: (np.ndarray, dtype=float32, shape=(H, C, 4))) output of LC device.
                        - Channels denote (x, y, z, intensity).
                        - Pixels that aren't a part of LC return will have NaNs in one of
                        the 4 channels.
                        - Intensity ranges from 0. to 255.
        Returns:
            curtain: (np.ndarray, dtype=np.float32, shape=(C',)) range per camera ray that correpsonds to a
                      valid curtain.
            mask: (np.ndarray, dtype=np.bool, shape=(C,)) subset of the camera rays for which a return was observed.
        """
        mask = np.logical_not(np.isnan(lc_image).any(axis=(0, 2)))  # (C,)
        
        xz = lc_image[:, mask, :][:, :, [0, 2]]  # (H, C', 2)
        assert np.all(xz[[0], :, :] == xz)  # consistency along column
        xz = xz[0]  # (C', 2)
        curtain = np.linalg.norm(xz, axis=1)  # (C',)

        return curtain, mask


    def _random_curtain(self, frontier):
        """Computes a random curtain that lies behind the current frontier
        
        Args:
            frontier: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that may not correpsond to a
                       valid curtain.
        Returns:
            curtain: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that correpsonds to a
                      valid curtain.
        """
        low = np.ones_like(frontier) * 0.5  # (C,)
        curtain = np.random.uniform(low=low, high=frontier)  # (C,)
        return curtain


    def _debug_visualize_curtains(self, f_curtain, r_curtain):
        design_pts = self._design_pts_from_ranges(f_curtain)
        x, z = design_pts[:, 0], design_pts[:, 1]
        plt.plot(x, z, c='b')

        design_pts = self._design_pts_from_ranges(r_curtain)
        x, z = design_pts[:, 0], design_pts[:, 1]
        plt.plot(x, z, c='r')

        plt.ylim(0, 21)
        plt.show()
    
    #endregion
    ####################################################################################################################
    
    def get_lc_data(self, se_pred):
        """
        Args:
            se_pred: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray of the estimate of the safety
                      envelope. This will be used to place the front curtain.
                      NOTE: it is required that se_pred correspond to a valid light curtain! Any device calling this
                      service must ensure this.
        Returns:
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
        ############################################################################################################
        # Place random light curtain and get return
        ############################################################################################################
        
        f_curtain = se_pred.copy()  # (C,) front curtain will be placed on se_pred.

        # get light curtain return from the design points
        r_curtain = self._random_curtain(f_curtain)
        design_pts = self._design_pts_from_ranges(r_curtain)
        yield self.env.process(self.light_curtain.service(design_pts))
        r_lc = self.light_curtain.stream[-1].data
        r_lc_image = r_lc["lc_image"]  # (H, W, 4)
        assert r_lc_image.shape[1] == self.C

        # Update curtain
        curtain, mask = self._validate_curtain_using_lc_image(r_lc_image)
        r_curtain[mask] = curtain
        r_lc["lc_ranges"] = r_curtain

        # Random curtain should be behind front curtain
        if not np.all(r_curtain <= f_curtain + 4):
            # debug this
            print("ASSERTION ERROR: r_curtain <= f_curtain + 4")
            import pdb; pdb.set_trace()

        ############################################################################################################
        # Place front light curtain and get return
        ############################################################################################################
        
        # get light curtain return from the design points
        design_pts = self._design_pts_from_ranges(f_curtain)
        yield self.env.process(self.light_curtain.service(design_pts))
        f_lc = self.light_curtain.stream[-1].data
        f_lc_image = f_lc["lc_image"]  # (H, W, 4)
        assert f_lc_image.shape[1] == self.C

        # Update and add curtain
        curtain, mask = self._validate_curtain_using_lc_image(f_lc_image)
        f_curtain[mask] = curtain
        f_lc["lc_ranges"] = f_curtain

        if self._debug:
            self._debug_visualize_curtains(f_curtain, r_curtain)
        
        data = dict(f_lc=f_lc, r_lc=r_lc)
        return data

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
        self.publish(data)
