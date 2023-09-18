import numpy as np
import sim
from typing import NamedTuple

from devices.device import Device
import utils

ingr = utils.Ingredient('lc_device')
ingr.add_config("config/lc_device.yaml")

########################################################################################################################
# region LCReturn class
########################################################################################################################


class LCReturn(NamedTuple):
    """
    Ranges per camera ray of the light curtain that was actually placed by the light curtain device.
        - (np.ndarray, dtype=np.float32, shape=(C,)) 
    """
    lc_ranges: np.ndarray  # (np.ndarray, dtype=np.float32, shape=(C,)) 

    """
    Light curtain image.
        - (np.ndarray, dtype=np.float32, shape=(H, C, 4)).
        - Axis 2 corresponds to (x, y, z, i):
                - x : x in cam frame.
                - y : y in cam frame.
                - z : z in cam frame.
                - i : intensity of LC cloud, lying in [0, 255].
    """
    lc_image: np.ndarray  # (np.ndarray, dtype=np.float32, shape=(H, C, 4))

    """
    Light curtain point cloud.
        - (np.ndarray, dtype=np.float32, shape=(N, 4))
        - Axis 2 corresponds to (x, y, z, i):
                - x : x in velo frame.
                - y : y in velo frame.
                - z : z in velo frame.
                - i : intensity of LC cloud, lying in [0, 1].
    """
    lc_cloud: np.ndarray  # (np.ndarray, dtype=np.float32, shape=(N, 4))


# endregion
########################################################################################################################

class LightCurtain(Device):
    @ingr.capture
    def __init__(self, env, gt_state_device, latency):
        super().__init__(env, capacity=1)
        self.gt_state_device = gt_state_device
        self.latency = latency

        # hardcoded lc_device for synthia
        self.lc_device = sim.LCDevice(
            CAMERA_PARAMS={
                'width': 640,
                'height': 480,
                'fov': 39.32012056540195,
                'matrix': np.array([[895.6921997070312, 0.0              , 320.0],
                                    [0.0              , 895.6921997070312, 240.0],
                                    [0.0              , 0.0              , 1.0  ]], dtype=np.float32),
                'distortion': [0, 0, 0, 0, 0]
            },
            LASER_PARAMS={
                'y': -0.2  # place laser 3m to the right of camera
            }
        )
        self._MAX_DEPTH = 80.0  # only consider points within this depth

    @property
    def thetas(self):
        return self.lc_device.thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]

    @staticmethod
    def _validate_curtain_using_lc_image(lc_image: np.ndarray):
        """Extract the actual valid curtain that was physically placed via the lc_image

        Args:
            lc_image (np.ndarray, dtype=float32, shape=(H, C, 4))):  output of LC device.
                        - Channels denote (x, y, z, intensity).
                        - Pixels that aren't a part of LC return will have NaNs in one of
                        the 4 channels.
                        - Intensity ranges from 0. to 255.
        Returns:
            curtain (np.ndarray, dtype=np.float32, shape=(C',)): range per camera ray that correpsonds to a
                      valid curtain.
            mask (np.ndarray, dtype=np.bool, shape=(C,)): subset of the camera rays for which a return was observed.
        """
        mask = np.logical_not(np.isnan(lc_image).any(axis=(0, 2)))  # (C,)

        xz = lc_image[:, mask, :][:, :, [0, 2]]  # (H, C', 2)
        assert np.all(xz[[0], :, :] == xz)  # consistency along column
        xz = xz[0]  # (C', 2)
        curtain = np.linalg.norm(xz, axis=1)  # (C',)

        return curtain, mask

    def get_lc_return_from_current_state(self,
                                         ranges: np.ndarray) -> LCReturn:
        """
        Args:
            ranges (np.ndarray, shape=(C,), dtype=np.float32): range per camera ray.

        Returns:
            lc_return (LCReturn): Light curtain return.
        """
        ranges = ranges.copy()
        design_pts = utils.design_pts_from_ranges(ranges, self.thetas)  # (C, 2)

        # get latest depth image at the time of publication
        if len(self.gt_state_device.stream) == 0:
            raise Exception("Light Curatin Device: gt_state_device stream empty at the time of LC publication!")
        depth_image = self.gt_state_device.stream[-1].data.depth  # (H, W)

        lc_image = self.lc_device.get_return(depth_image, design_pts)  # (H, W, 4)
        lc_cloud = lc_image.reshape(-1, 4)  # (N, 4)
        # Remove points which are NaNs.
        non_nan_mask = np.all(np.isfinite(lc_cloud), axis=1)
        lc_cloud = lc_cloud[non_nan_mask]  # (N, 4)
        # Convert lc_cloud to velo frame.
        lc_cloud_xyz1 = np.hstack((lc_cloud[:, :3], np.ones([len(lc_cloud), 1], dtype=np.float32)))
        lc_cloud_xyz1 = lc_cloud_xyz1 @ self.lc_device.TRANSFORMS["cam_to_world"].T
        lc_cloud[:, :3] = lc_cloud_xyz1[:, :3]  # (N, 4)
        # Rescale LC return to [0, 1].
        lc_cloud[:, 3] /= 255.  # (N, 4)

        # Update curtain
        curtain, mask = self._validate_curtain_using_lc_image(lc_image)
        ranges[mask] = curtain
        lc_ranges = ranges

        lc_return = LCReturn(lc_ranges, lc_image, lc_cloud)
        return lc_return

    def service(self,
                ranges: np.ndarray) -> LCReturn:
        """
        Args:
            ranges (np.ndarray, shape=(C,), dtype=np.float32): range per camera ray.

        Publishes:
            lc_return (LCReturn): light curtain return.
        """
        yield self.env.timeout(self.latency)
        lc_return = self.get_lc_return_from_current_state(ranges)
        self.publish(lc_return)
