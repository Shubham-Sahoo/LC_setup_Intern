from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Any, Optional, Tuple

from data.synthia import Frame


class Actor(ABC):
    @property
    def latency(self) -> int:
        """Latency in integer number of milliseconds"""
        raise NotImplementedError

    @abstractmethod
    def init_action(self,
                    state: Frame) -> Tuple[np.ndarray, Optional[np.ndarray], bool, dict]:
        """
        Args:
            state (Frame): the state of the world represented by a Frame from the gt_state_device.

        Returns:
            action (np.ndarray, dtype=np.float32, shape=(C,)): action, in terms of ranges for each camera ray.
            logp_a (Optional[np.ndarray]): log-probability of the sampled actions. None if sampling is deterministic.
            control (bool): whether this policy had control of taking the action at this timestep.
                            - this will be used to determine whether to evaluate the policy at this frame.
                            - another eg. is that these are the timesteps when nn-based policies will run the network.
            info (dict): auxiliary info generated by policy, such as a vector representation of the observation while
                         generating the action.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self,
             obs: Any) -> Tuple[np.ndarray, Optional[np.ndarray], bool, dict]:
        """
        Args:
            obs (Any): observation
        
        Returns:
            act (np.ndarray, dtype=np.float32, shape=(C,)): sampled actions.
            logp_a (Optional[np.ndarray]): log-probability of the sampled actions. None if sampling is deterministic.
            control (bool): whether this policy had control of taking the action at this timestep.
                            - this will be used to determine whether to evaluate the policy at this frame.
                            - another eg. is that these are the timesteps when nn-based policies will run the network.
            info (dict): auxiliary info generated by policy, such as a vector representation of the observation while
                         generating the action.
        """
        raise NotImplementedError

    def forward(self,
                obs: torch.Tensor) -> torch.distributions.Distribution:
        """
        Args:
            obs (torch.Tensor): observation in torch tensors.
        
        Returns:
            pi (torch.distributions.Distribution): predicted action distribution of the policy.
        """
        raise NotImplementedError

    def evaluate_actions(self,
                         obs: torch.Tensor,
                         act: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs (torch.Tensor): observation in torch tensors.
            act (torch.Tensor): actions in torch tensors.
        
        Returns:
            logp_a (torch.Tensor): log probability of taking actions "act" by the actor under "obs", as a torch tensor.
        """
        raise NotImplementedError

    def reset(self):
        """
        Generally, actors can maintain a history of observations from the environment, and actions can be a function of
        the entire history. Reset is used to clear any such histories.
        """
        pass
