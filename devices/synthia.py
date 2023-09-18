import random
from abc import ABC, abstractmethod

import tqdm

from data.synthia import SynthiaVideoDataset, Frame
from devices.device import Device

# LC return for height < self.GROUND_HEIGHT is treated as coming from the ground.
# Height in velo frame is +Z and cam frame is -Y.
GROUND_HEIGHT = -1.2


class SynthiaGTState(Device):
    def __init__(self, env, split, latency=40, cam=False, pc=False, preload=True, progress=True):
        """
        Args:
            env: (simply.Environment) simulation environment.
            split: (string) dataset split ["train"/"test"].
            latency: (float) latency of the Synthia dataset (25fps).
            cam: (bool) whether to publish camera image
            pc: (bool) whether to load point cloud
            preload (bool): whether to prelaod the entire video
            progress (bool): whether to show tqdm's progress bar
        """
        super().__init__(env, capacity=1)  # synthia will only expose the most recent ground truth
        self.dataset = SynthiaVideoDataset(split, cam, pc)
        self.latency = latency
        self._preload = preload
        self._progress = progress

    def init(self, idx, start=0):
        video = self.dataset[idx]
        frame_gt = video[start]
        self.publish(frame_gt)

    def process(self, idx, start=0):
        self.stream.clear()  # empty stream
        prange = tqdm.trange if self._progress else range

        video = self.dataset[idx]
        if self._preload:
            if self._progress: print("Preloading video ...")
            video = [video[i] for i in prange(len(video))]

        if self._progress: print("Streaming video ...")
        for i in prange(start, len(video)):
            frame_gt = video[i]
            self.publish(frame_gt)  # first publication is at t=0
            yield self.env.timeout(self.latency)


########################################################################################################################
# region EpisodeIndexIterator
########################################################################################################################

class EpIndexIterator(ABC):
    """Samples a sequence of (idx, start) pairs where idx is the video id, and start is the start frame index"""
    def __init__(self,
                 gt_state_device: SynthiaGTState):
        """
        Args:
            gt_state_device (SynthiaGTState): the ground truth state device.
        """
        self.gt_state_device = gt_state_device
        self.dataset = self.gt_state_device.dataset
        self.num_videos = len(self.dataset)

    @abstractmethod
    def __iter__(self):
        return NotImplementedError


class TrainEpIndexIterator(EpIndexIterator):
    def __init__(self, gt_state_device):
        super().__init__(gt_state_device)

        # all pairs of video idx and start frame idx in video
        self.tuples = [(vid, start) for vid in range(self.num_videos) for start in range(len(self.dataset[vid]))]

    def __iter__(self):
        i = 0
        while True:
            if i == len(self.tuples):
                i = 0  # reset to beginning creating infinitely many rounds
            if i == 0:
                random.shuffle(self.tuples)  # shuffle at the beginning of every round
            yield self.tuples[i]
            i += 1


class EvalEpIndexIterator(EpIndexIterator):
    def single_video_iter(self, vid):
        video_no, video_info = self.dataset.video_infos[vid]
        fid = 0  # always start with the first frame

        while fid < len(self.dataset[vid]):
            yield vid, fid

            frame: Frame = self.gt_state_device.stream[-1].data

            gt_state_vid = frame.metadata["vid"]
            gt_state_video_no = frame.metadata["video_no"]
            gt_state_fid = frame.metadata["fid"]

            assert (vid == gt_state_vid and video_no == gt_state_video_no), \
                "gt_state_device used must exclusively use the vid provided by EpIndexIterator"

            assert gt_state_fid >= fid, \
                "gt_state_device cannot go backwards in the episode"

            # next fid
            fid = gt_state_fid + 1

    def __iter__(self):
        for vid_ in range(self.num_videos):
            for vid, fid in self.single_video_iter(vid_):
                yield vid, fid


class DeprecatedEpisodeIndexIterator:
    def __init__(self,
                 gt_state_device: SynthiaGTState,
                 shuffle: bool,
                 partial: bool,
                 loop: bool):
        """
        Samples a sequence of (idx, start) pairs where idx is the video id, and start is the start frame index.

        Args:
            gt_state_device (SynthiaGTState): the ground truth state device.
            shuffle (bool): whether to randomly permute the index pairs or not.
            partial (bool): whether to use partial videos (start can be any value) or full videos (start is always 0).
            loop    (bool): whether to loop over the set of index pairs indefinitely, or return after 1 loop.
        """
        # save options
        self.shuffle = shuffle
        self.loop = loop
        self.partial = partial

        dataset = gt_state_device.dataset
        num_videos = len(dataset)

        if self.partial:
            self.tuples = [(idx, start) for idx in range(num_videos) for start in range(len(dataset[idx]))]
        else:
            self.tuples = [(idx, 0) for idx in range(num_videos)]

    def __len__(self):
        if self.loop:
            raise ValueError("length of EpisodeIndexIterator with loop=True is undefined")
        else:
            return len(self.tuples)

    def __iter__(self):
        i = 0
        while True:
            if i == len(self.tuples):
                if self.loop:
                    i = 0  # reset to beginning creating infinitely many rounds
                else:
                    return  # break out of loop creating just a single round
            if i == 0 and self.shuffle:
                random.shuffle(self.tuples)  # shuffle at the beginning of every round
            yield self.tuples[i]
            i += 1

# endregion
########################################################################################################################

