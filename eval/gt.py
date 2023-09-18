from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, List

import utils
from data.synthia import Frame


@dataclass
class MetricsGT:
    safety: float
    proximity: float
    accuracy: float

    def __str__(self):
        s, p, a = self.safety * 100, self.proximity * 100, self.accuracy * 100
        return f"| safety: {s:06.2f} | proximity: {p:06.2f} | accuracy: {a:06.2f} |"


class EvaluatorGT:
    """Performs ground truth evaluation based on per-frame actions"""
    def __init__(self,
                 n_bins: int = 1000,
                 debug: bool = False):
        """
        Args:
            n_bins (int): number of histogram bins.
            debug (bool): whether to debug or not
        """
        self._n_bins = n_bins  # denoted as BINS
        self._debug = debug

        self._store = None

        # Pre-computations that will help get metrics
        bin_edges = np.histogram_bin_edges(np.empty(0), range=(0, 2), bins=self._n_bins)
        self.bins_l = bin_edges[:-1]  # left values for each bin
        self.bins_r = bin_edges[1:]  # right values for each bin
        self.bins_m = 0.5 * (self.bins_l + self.bins_r)  # mid values for each bin

        # This is the index that 1 lies in (we want to include this).
        # Note that np.hist bins are left closed and right open.
        self.one_index = np.where((self.bins_l <= 1) & (1 < self.bins_r))[0][0]
        self.n_safe_bins = self.one_index + 1  # number of bins that are safe

    def _create_store(self, n_policies):
        assert self._store is None, "Cannot create store without calling EvaluatorGT.reset()"
        self._store = [np.zeros(self._n_bins, dtype=np.float32) for i in range(n_policies)]

    def _compute_hist(self,
                      action: np.ndarray,
                      gt_action: np.ndarray):
        ratio = (action / gt_action).clip(min=0, max=2)  # (C,)
        hist = np.histogram(ratio, range=(0, 2), bins=self._n_bins)[0]  # (BINS,)
        return hist

    def _hist_to_metrics(self,
                         hist: np.ndarray) -> MetricsGT:
        """
        Compute metrics of a single policy.

        Args:
            hist (np.ndarray, dtype=np.float32, shape=(BINS,)): hist of a single policy

        Returns:
            metrics (MetricsGT): ground truth metrics.
        """
        hist = hist / hist.sum()  # (BINS)  convert hist into a probability distribution

        # safety
        p_safe = hist[:self.n_safe_bins].sum()  # probability of safe rays
        safety = p_safe  # fraction of rays that are safe

        # proximity
        if p_safe == 0:
            proximity = 0
        else:
            proximity = (self.bins_m * hist)[:self.n_safe_bins].sum() / p_safe  # conditional mean of ratio given safe

        # accuracy
        accuracy = safety * proximity

        metrics = MetricsGT(safety=safety, proximity=proximity, accuracy=accuracy)

        if self._debug:
            sns.distplot(self.bins_l, bins=self._n_bins, kde=False, hist_kws={"weights": hist})
            plt.xlabel("distance ratio", fontsize='xx-large')
            plt.ylabel("% rays", fontsize='xx-large')
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            title = str(metrics)
            plt.title(title, fontsize='xx-large')
            plt.tight_layout()
            plt.show()

        return metrics

    ####################################################################################################################
    # region API
    ####################################################################################################################

    def reset(self):
        self._store = None

    def add(self,
            actions: List[np.ndarray],
            gt_action: Optional[np.ndarray] = None,
            frame: Optional[Frame] = None):
        """
        Args:
            actions (List[np.ndarray, dtype=float32, shape=(C,)]): an action (ranges) for each policy to evaluate.
            gt_action (Optional[np.ndarray, dtype=float32, shape=(C,)]: ground truth ranges (safety envelope).
            frame (Optional[Frame]): frame from GT state device.
        """
        if self._store is None:
            self._create_store(n_policies=len(actions))

        assert len(actions) == len(self._store), \
            f"Number of actions ({len(actions)}) does not match expected number ({len(self._store)}). " + \
            f"Use the same number of actions consistently."

        # get gt_action
        if gt_action is None:
            assert frame is not None, "Should provide either the ground truth action or the Frame"
            gt_action = utils.safety_envelope(frame)  # (C,)

        # accumulate hists
        for i, action in enumerate(actions):
            hist = self._compute_hist(action, gt_action)  # (BINS,)
            self._store[i] += hist

    def metrics(self,
                reset: bool = False) -> List[MetricsGT]:
        """
        Args:
            reset (bool): whether to reset at the end or not. Equivalent to calling EvaluatorGT.reset() at the end.

        Returns:
            List[MetricsGT]: metrics for each policy.
        """

        ret = []
        for hist in self._store:
            metrics = self._hist_to_metrics(hist)
            ret.append(metrics)

        if reset:
            self.reset()

        return ret

    # region
    ####################################################################################################################
