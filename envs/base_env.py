import gym
import matplotlib.pyplot as plt
import numpy as np
import simpy
from typing import Optional, Tuple

from data.synthia import Frame
from devices.synthia import SynthiaGTState
from devices.light_curtain import LightCurtain, LCReturn
from planner import PlannerRT
import utils


class BaseEnv(gym.Env):
    Observation = Tuple[LCReturn, LCReturn]

    """
    An OpenAI Gym wrapper of a simpy simulation of safety envelope tracking.
    """

    def __init__(self,
                 debug: bool = False,
                 **dataset_kwargs):
        """
        Args:
            debug (bool): whether to run the environment in debug mode.
            dataset_kwargs (dict): kwargs to pass to SynthiaGTState
        """
        self._debug = debug

        # options
        self._MIN_RANGE     = 1
        self._MAX_RANGE     = 20
        self._NODES_PER_RAY = 120  # 0.16m apart

        # Simpy environment
        self.simpy_env = simpy.Environment()

        # Set up devices
        self.gt_state_device = SynthiaGTState(self.simpy_env, **dataset_kwargs)
        self.light_curtain = LightCurtain(self.simpy_env, self.gt_state_device)

        # Set up utilities
        ranges = np.arange(1, self._NODES_PER_RAY + 1, dtype=np.float32) / self._NODES_PER_RAY * self._MAX_RANGE  # (R,)
        self.planner_min = PlannerRT(self.light_curtain.lc_device, ranges, self.C, maximize=False)

        self.gt_proc_done_event = None

    @property
    def thetas(self):
        return self.light_curtain.thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]

    @property
    def C(self):
        return self.light_curtain.lc_device.CAMERA_PARAMS["width"]  # number of camera rays

    ####################################################################################################################
    # region Env functions
    ####################################################################################################################

    def reset(self,
              idx: int,
              start: int = 0,
              demonstration: bool = False) -> Tuple[Frame, bool, dict]:
        """Resets the state of the environment and returns an initial observation.

        Args:
            idx (int): video id.
            start (int): start frame of video.
            demonstration (bool): whether to provide a demonstration for this observation or not

        Returns:
            state (Frame): the initial state.
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Create new simpy env and reset all devices with it.
        self.simpy_env = simpy.Environment()
        self.gt_state_device.reset(self.simpy_env)
        self.light_curtain.reset(self.simpy_env)

        # Start gt_state_device's process.
        self.gt_proc_done_event = self.simpy_env.process(self.gt_state_device.process(idx, start))

        self.gt_state_device.init(idx, start)
        init_frame: Frame = self.gt_state_device.stream[-1].data

        info = {}
        if demonstration:
            info["demonstration"] = utils.safety_envelope(init_frame)

        return init_frame, False, info

    def step(self,
             action: np.ndarray,
             latency: float = 0.0,
             demonstration: bool = False) -> Tuple[Observation, Optional[float], bool, dict]:
        """
        Args:
            action (np.ndarray, dtype=np.float32, shape=(C,)): Ranges of the light curtain.
            latency (float): Time take by the policy to compute this action, used by the simpy simulation.
            demonstration (bool): whether to provide a demonstration for this observation or not

        Returns:
            observation (Tuple[LCReturn, LCReturn]): agent's observation of the current environment. This is the return
                        from the front and random light curtains respectively.
            reward (float) : Amount of reward returned after previous action
            done (bool): Whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # launch lc process
        lc_proc_done_event = self.simpy_env.process(self.lc_process(action, latency))

        # wait for lc process to get lc returns
        condition_value = self.simpy_env.run(until=lc_proc_done_event | self.gt_proc_done_event)
        triggered_events = condition_value.events

        done = (self.gt_proc_done_event in triggered_events) or (lc_proc_done_event not in triggered_events)
        if done:
            return None, None, True, {}
        else:
            obs = lc_proc_done_event.value

            info = {}
            if demonstration:
                curr_frame: Frame = self.gt_state_device.stream[-1].data
                info["demonstration"]: np.ndarray = utils.safety_envelope(curr_frame)  # (C,)
            return obs, None, False, info

    def render(self, mode='human'):
        raise NotImplementedError

    # endregion
    ####################################################################################################################
    # region Light curtain process
    ####################################################################################################################

    def lc_process(self,
                   action: np.ndarray,
                   latency: float = 0.0) -> Observation:
        """Places forward (and optionally random) light curtains based on actions ONCE, and returns the results"""
        yield self.simpy_env.timeout(latency)  # policy's latency to compute action; timeout for this duration

        # get a valid front curtain from the action
        f_curtain = action.clip(min=self._MIN_RANGE, max=self._MAX_RANGE)  # (C,)

        # validate curtain to be behind
        f_curtain = utils.valid_curtain_behind_frontier(self.planner_min, f_curtain, debug=self._debug)  # (C,)

        r_return = yield self.simpy_env.process(self.get_r_return(f_curtain))
        f_return = yield self.simpy_env.process(self.get_f_return(f_curtain))
        obs = (f_return, r_return)

        return obs

    # endregion
    ####################################################################################################################
    # region Helper functions
    ####################################################################################################################

    def _debug_visualize_curtains(self, f_curtain, r_curtain):
        design_pts = utils.design_pts_from_ranges(f_curtain, self.thetas)
        x, z = design_pts[:, 0], design_pts[:, 1]
        plt.plot(x, z, c='b')

        design_pts = utils.design_pts_from_ranges(r_curtain, self.thetas)
        x, z = design_pts[:, 0], design_pts[:, 1]
        plt.plot(x, z, c='r')

        plt.ylim(0, 21)
        plt.show()

    def _random_curtain(self,
                        frontier: np.ndarray) -> np.ndarray:
        """Computes a random curtain that lies behind the current frontier
        
        Args:
            frontier: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that may not correpsond to a
                       valid curtain.
        Returns:
            curtain: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that correpsonds to a
                      valid curtain.
        """
        low = np.ones_like(frontier) * 0.5 * self._MIN_RANGE  # (C,)
        curtain = np.random.uniform(low=low, high=frontier)  # (C,)
        return curtain

    def get_f_return(self,
                     f_curtain: np.ndarray) -> LCReturn:
        """
            Args:
                f_curtain (np.ndarray, dtype=np.float32, shape=(C,)): Range per camera ray of the estimate of the safety
                       envelope. This will be used to place the front curtain.
                       NOTE: it is required that se_pred correspond to a valid light curtain! Any device calling this
                       service must ensure this.
            Returns:
                f_return (LCReturn): Return of the front light curtain, placed using "action".
            """
        f_curtain = f_curtain.copy()  # (C,) front curtain will be placed on se_pred.
        yield self.simpy_env.process(self.light_curtain.service(f_curtain))
        f_return: LCReturn = self.light_curtain.stream[-1].data
        return f_return

    def get_r_return(self,
                     f_curtain: np.ndarray) -> LCReturn:
        """
            Args:
                f_curtain (np.ndarray, dtype=np.float32, shape=(C,)): Range per camera ray of the estimate of the safety
                       envelope. This will be used to place the front curtain.
                       NOTE: it is required that se_pred correspond to a valid light curtain! Any device calling this
                       service must ensure this.
            Returns:
                r_return (LCReturn): Return of the random light curtain, placed behind the "action" curtain.
            """
        f_curtain = f_curtain.copy()  # (C,) front curtain will be placed on se_pred.
        r_curtain = self._random_curtain(f_curtain)
        yield self.simpy_env.process(self.light_curtain.service(r_curtain))
        r_return: LCReturn = self.light_curtain.stream[-1].data

        # Random curtain should be behind front curtain
        if not np.all(r_return.lc_ranges <= f_curtain + 4):
            raise AssertionError("r_curtain <= f_curtain + 4")

        return r_return

    # endregion
    ####################################################################################################################
