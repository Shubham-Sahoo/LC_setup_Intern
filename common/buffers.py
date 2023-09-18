from dataclasses import dataclass
from typing import Optional, Generator
import numpy as np

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


@dataclass
class ImitationRolloutBufferSamples:
    features: np.ndarray
    actions: np.ndarray
    demonstrations: np.ndarray


class ImitationRolloutBuffer(RolloutBuffer):
    def reset(self) -> None:
        self.features = np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.demonstrations = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def add(self,
            feat: np.ndarray,
            act : np.ndarray,
            dem : np.ndarray) -> None:
        """
        :param feat: (np.ndarray) Features from observations
        :param act : (np.ndarray) Action
        :param dem : (np.ndarray) Demonstration
        """
        self.features[self.pos] = np.array(feat).copy()
        self.actions[self.pos] = np.array(act).copy()
        self.demonstrations[self.pos] = np.array(dem).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[ImitationRolloutBufferSamples, None, None]:
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['features', 'actions', 'demonstrations']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> ImitationRolloutBufferSamples:
        feat: np.ndarray = self.features[batch_inds]
        act : np.ndarray = self.actions[batch_inds]
        dem : np.ndarray = self.demonstrations[batch_inds]
        return ImitationRolloutBufferSamples(feat, act, dem)
