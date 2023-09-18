import datetime
import functools
import gridfs
import matplotlib.pyplot as plt
import numpy as np
import pymongo
import sacred
import time

from data.synthia import Frame, append_xyz_to_depth_map
from devices.synthia import GROUND_HEIGHT

########################################################################################################################
# region: Sacred
########################################################################################################################


INGREDIENTS = []


class Ingredient(sacred.Ingredient):
    def __init__(self, path):
        super().__init__(path, ingredients=INGREDIENTS)
        INGREDIENTS.append(self)


class Experiment(sacred.Experiment):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, ingredients=INGREDIENTS, **kwargs)
        INGREDIENTS.append(self)

# endregion
########################################################################################################################
# region: Timer
########################################################################################################################

"""
Usage:
    from utils import timer

    # this may be in a loop
    with timer.time_as("loading images"):
        /* code that loads images */
    
    # this may be in a loop
    with timer.time_as("computation"):
        /* code that does computation */
    
    timer.print_stats()
"""


class Timer:
    def __init__(self):
        self.ttime = {}  # total time for every key
        self.ttimesq = {}  # total time-squared for every key
        self.titer = {}  # total number of iterations for every key

    def _add(self, key, time_):
        if key not in self.ttime:
            self.ttime[key] = 0
            self.ttimesq[key] = 0
            self.titer[key] = 0

        self.ttime[key] += time_
        self.ttimesq[key] += time_ * time_
        self.titer[key] += 1

    def print_stats(self):
        print("TIMER STATS:")
        word_len = max([len(k) for k in self.ttime.keys()]) + 8
        for key in self.ttime:
            ttime_, ttimesq_, titer_ = self.ttime[key], self.ttimesq[key], self.titer[key]
            mean = ttime_ / titer_
            std = np.sqrt(ttimesq_ / titer_ - mean * mean)
            interval = 1.96 * std
            print(f"{key.rjust(word_len)}: {mean:.3f}s ± {interval:.3f}s")

    class TimerContext:
        def __init__(self, timer, key):
            self.timer = timer
            self.key = key

            self._stime = 0
            self._etime = 0

        def __enter__(self):
            self._stime = time.time()

        def __exit__(self, type, value, traceback):
            self._etime = time.time()
            time_ = self._etime - self._stime
            self.timer._add(self.key, time_)

    def time_as(self, key):
        return Timer.TimerContext(self, key)

    def time_fn(self, name):
        def decorator(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                with self.time_as(name):
                    ret = function(*args, **kwargs)
                ttime_   = self.ttime[name]
                titer_   = self.titer[name]
                avg_time = ttime_ / titer_
                print(f"Avg. {name} time is {datetime.timedelta(seconds=round(avg_time))}s")
                return ret
            return wrapper
        return decorator


timer = Timer()

# endregion
########################################################################################################################
# region: Meters
########################################################################################################################


# copied from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# endregion
########################################################################################################################
# region: Light curtain utils
########################################################################################################################


def valid_curtain_behind_frontier(planner_min, frontier, debug=False):
    """Computes a valid curtain that lies strictly behind the current frontier using the planner
    
    Args:
        planner_min: (lcsim.python.planner.PlannerRT) PlannerRT initialized with ranges that performs minimization.
        frontier: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that may not correpsond to a
                    valid curtain.
    Returns:
        curtain: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that correpsonds to a
                    valid curtain.
    """
    # construct cmap
    ranges = planner_min.ranges  # (R,)
    safe_mask = ranges.reshape(-1, 1) <= frontier  # (R, C)
    distances = np.abs(ranges.reshape(-1, 1) - frontier)  # (R, C)

    # cost map = distance in safety region and infinity outside
    cmap = np.inf * np.ones_like(distances)  # (R, C)
    safe_mask_i, safe_mask_j = np.where(safe_mask)  # both are (N,)
    cmap[safe_mask_i, safe_mask_j] = distances[safe_mask_i, safe_mask_j]  # (R, C)

    design_pts = planner_min.get_design_points(cmap)  # (C, 2)
    assert design_pts.shape == (planner_min.num_camera_angles, 2)

    if debug:
        unsafe_mask_i, unsafe_mask_j = np.where(np.logical_not(safe_mask))
        cmap[unsafe_mask_i, unsafe_mask_j] = 0
        cmap[safe_mask_i, safe_mask_j] = 100 - cmap[safe_mask_i, safe_mask_j]
        planner_min._visualize_curtain_rt(cmap, design_pts, show=False)
        new_x, new_z = design_pts[:, 0], design_pts[:, 1]  # (C,)
        thetas = np.arctan2(new_z, new_x)
        old_x, old_z = frontier * np.cos(thetas), frontier * np.sin(thetas)
        plt.plot(old_x, old_z, c='r', linewidth=0.5)
        plt.ylim(0, 30)
        plt.xlim(-10, 10)
        plt.show()

    # compute curtain from design points
    curtain = np.linalg.norm(design_pts, axis=1)  # (C,)

    # assert that curtain lies completely behind planner
    if not np.all(curtain <= frontier + 1e-3):
        # debug this
        raise AssertionError("planner doesn't place curtain completely behind frontier.")

    return curtain


def design_pts_from_ranges(ranges: np.ndarray,
                           thetas: np.ndarray):
    """
        Args:
            ranges (np.ndarray, shape=(C,), dtype=np.float32): range per camera ray
            thetas (np.ndarray, shape=(C,), dtype=np.float32): in degrees and in increasing order in [-fov/2, fov/2]
        Returns:
            design_pts: (np.ndarray, shape=(C, 2), dtype=np.float32) design points corresponding to frontier.
                        - Axis 1 channels denote (x, z) in camera frame.
        """
    x = ranges * np.sin(np.deg2rad(thetas))
    z = ranges * np.cos(np.deg2rad(thetas))
    design_pts = np.hstack([x.reshape(-1, 1), z.reshape(-1, 1)])
    return design_pts


def hits_from_lc_image(lc_image: np.ndarray,
                       ithresh: float) -> np.ndarray:
    """
        Args:
            lc_image (np.ndarray, dtype=np.float32, shape=(H, C, 4)): lc image.
                      Axis 2 corresponds to (x, y, z, i):
                        - x : x in cam frame.
                        - y : y in cam frame.
                        - z : z in cam frame.
                        - i : intensity of LC cloud, lying in [0, 255].
            ithresh: (float) intensity threshold in [0, 255] above which returns will be considered as hits.

        Returns:
            hits (np.ndarray, dtype=np.bool, shape=(C,)): whether there is a hit or not for every camera column.
        """
    hits = np.ones(lc_image.shape[:2], dtype=np.bool)  # (H, C)

    # mask out NaN values
    hits[np.isnan(lc_image).any(axis=2)] = 0  # (H, C)

    # mask out pixels below intensity threshold
    hits[lc_image[:, :, 3] < ithresh] = 0

    # mask out pixels that are below GROUND_HEIGHT (note that cam_y points downwards)
    hits[-lc_image[:, :, 1] < GROUND_HEIGHT] = 0

    # collect hits across camera columns
    hits = hits.any(axis=0)  # (C,)

    return hits


def safety_envelope(frame: Frame) -> np.ndarray:
    """
    Computes ground truth safety envelope from the ground truth depth map in the frame.

    Args:
        frame (Frame): frame containing ground truth depth.

    Returns:
        se_gt: (np.ndarray, dtype=np.float32, shape=(C,)) the ranges of the ground truth safety envelope, one per
                camera ray.
    """
    depth = frame.depth.copy()  # (H, C)

    # maximum depth is 20m, we do not care about object beyond 20m
    depth = depth.clip(max=20)  # (H, C)

    # append x, y, z to depth
    P2 = frame.calib["P2"][:3, :3]  # (3, 3)
    cam_xyz = append_xyz_to_depth_map(depth[:, :, None], P2)  # (H, C, 3); axis 2 is (x, y, z) in cam frame

    # pixels that are below GROUND_HEIGHT are assumed to be infinitely far away (note that cam_y points downwards)
    depth[-cam_xyz[:, :, 1] < GROUND_HEIGHT] = 20

    se_gt = depth.min(axis=0)  # (C,)
    return se_gt

# endregion
########################################################################################################################
# region: Sacred utils
########################################################################################################################


def get_sacred_artifact_from_mongodb(run_id, name):
    """
    Get artifact from MongoDB

    Args:
        run_id (int): id of the run
        name (string): name of the artifact saved

    Returns:
        file: a file-like object with a read() function
    """
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    fs = gridfs.GridFS(client.sacred)

    runs = client.sacred.runs
    run_entry = runs.find_one({'_id': run_id})
    artifacts = run_entry["artifacts"]
    artifacts = list(filter(lambda entry: entry["name"] == "residual_weights", artifacts))
    assert len(artifacts) == 1,\
        str(f"Number of artifacts with run_id={run_id} and name={name} is {len(artifacts)} instead of 1")
    file_id = artifacts[0]['file_id']
    file = fs.get(file_id)  # this is a file-like object that has a read() method

    return file

# endregion
########################################################################################################################
