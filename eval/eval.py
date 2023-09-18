import matplotlib.pyplot as plt
from sacred import Experiment
import seaborn as sns
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import pickle
import simpy

from devices.synthia import SynthiaGTState
from devices.light_curtain import LightCurtain
from devices.managers.eval_manager import EvalManager

# agents
from devices.agents.baseline_fr import BaselineFrontRandomAgent
from devices.agents.oracle import OracleAgent


def bpc(pc, dec):
    num = np.round(pc * 100, dec)
    return r"$\bf{{{}\%}}$".format(num)

class Store:
    def __init__(self, name):
        """
        self.data is going to be indexed as
            data[video_id][type_name] = [ histogram ]
        """
        self.name = name
        self.data = {}
    
    def add_hist(self, video_id, type_name, hist):
        if video_id not in self.data:
            self.data[video_id] = {}
        
        if type_name not in self.data[video_id]:
            self.data[video_id][type_name] = []
        
        self.data[video_id][type_name].append(hist)
    
    def save(self):
        pickle_path = Path(f"store_{self.name}.pkl")
        print(f"Saving to {pickle_path} ...")

        with open(str(pickle_path), "wb") as f:
            pickle.dump(self.data, f)

    def load(self):
        pickle_path = Path(f"store_{self.name}.pkl")
        print(f"Loading from {pickle_path} ...")

        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

    def __getitem__(self, type_name):
        """
        Returns:
            all_hists = (np.ndarray, dtype=np.float32, shape=(N, BINS)) 2D array of histograms.
        """
        # video_id doesn't matter
        ret = []
        for video_data in self.data.values():
            if type_name not in video_data: continue
            ret.extend(video_data[type_name])
        
        # ret is now a single list of histograms
        ret = np.vstack(ret)  # (N, BINS)
        return ret

class GTMetrics:
    def __init__(self, hists, name):
        """
        Args:
            hists: (np.ndarray, dtype=np.float32, shape=(N, BINS_GT)) collection of histograms of distance ratios.
        """
        self.name = name

        # comine histograms across frames
        # this contains counts of rays
        self.hist = hists.sum(axis=0)  # (BINS_GT,)

        self.BINS, MIN, MAX = len(self.hist), 0, 2
        bin_edges = np.histogram_bin_edges(np.empty(0), range=(MIN, MAX), bins=self.BINS)
        self.bins_l = bin_edges[:-1]                     # left values for each bin
        self.bins_r = bin_edges[1:]                      # right values for each bin
        self.bins_m = 0.5 * (self.bins_l + self.bins_r)  # mid values for each bin

        # This is the index that 1 lies in (we want to include this).
        # Note that np.hist bins are left closed and right open.
        self.one_index = np.where((self.bins_l <= 1) & (1 < self.bins_r))[0][0]
        self.num_safe_bins = self.one_index + 1  # number of bins that are safe

        self.pcentile_order = np.hstack([np.arange(self.BINS-1, self.one_index, -1),
                                         np.arange(self.num_safe_bins)])
    
    def evaluate(self, pcentile=1.):
        hist = self._pcentile2hist(pcentile)
        return self._hist2metrics(hist, prefix=f"GTR ({self.name}) (worst {int(pcentile*100)}%)\n")
    
    def _pcentile2hist(self, pcentile):
        """Percentile at the ray level: chose the worst p% rays"""
        assert 0 <= pcentile <= 1

        ret_hist = np.zeros_like(self.hist)

        count = int(pcentile * self.hist.sum())  # we need to include these many rays
        for bin_ind in self.pcentile_order:
            if count == 0:
                break
            rays_in_bin = self.hist[bin_ind]
            rays_to_move = min(count, rays_in_bin)
            ret_hist[bin_ind] = rays_to_move
            count -= rays_to_move
        
        return ret_hist

    def _hist2metrics(self, hist, prefix=""):
        """
        Args:
            hist: (np.ndarray, dtype=np.float32, shape=(BINS_GT,)) histogram of distance ratios,
                   contains counts of rays.
        """
        hist = hist / hist.sum()  # (BINS_GT)  convert it into a probability distribution

        p_safe = hist[:self.num_safe_bins].sum()  # probability of safe rays
        
        # metrics
        safety = p_safe  # fraction of rays that are safe
        proximity = (self.bins_m * hist)[:self.num_safe_bins].sum() / p_safe  # conditional mean of ratio given safe
        accuracy = safety * proximity

        metrics = dict(safety=safety, proximity=proximity, accuracy=accuracy)

        sns.distplot(self.bins_l, bins=self.BINS, kde=False, hist_kws={"weights": hist})
        plt.xlabel("distance ratio", fontsize='xx-large')
        plt.ylabel("% rays",  fontsize='xx-large')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        title = prefix + f"Safety: {bpc(safety, 2)}, Proximity: {bpc(proximity, 2)}, Accuracy: {bpc(accuracy, 2)}"
        plt.title(title, fontsize='xx-large')
        plt.tight_layout()
        plt.show()

        return metrics

class SSMetrics:
    def __init__(self, f_hists, r_hists, name):
        self.f_hists = f_hists
        self.r_hists = r_hists
        self.name = name

        self.BINS, MIN, MAX = self.f_hists.shape[1], 0, 1
        bin_edges = np.histogram_bin_edges(np.empty(0), range=(MIN, MAX), bins=self.BINS)
        self.bins_l = bin_edges[:-1]                     # left values for each bin
        self.bins_r = bin_edges[1:]                      # right values for each bin
        self.bins_m = 0.5 * (self.bins_l + self.bins_r)  # mid values for each bin
        
        self.f_dist = self.f_hists.sum(axis=0)
        self.f_dist = self.f_dist / self.f_dist.sum()
        self.r_dist = self.r_hists.sum(axis=0)
        self.r_dist = self.r_dist / self.r_dist.sum()

        self.beta = (self.bins_m * self.f_dist).sum() / (self.bins_m * self.r_dist).sum()
        print(f"SSMetrics: beta that equalizes f and r intensities: {self.beta:.4f}")

        def means_per_frame(hists):  # hists is (N, BINS_SS)
            dist_per_frame = hists / hists.sum(axis=1).reshape(-1, 1)  # (N, BINS_SS)
            mean_per_frame = (dist_per_frame * self.bins_m).sum(axis=1)  # (N,)
            # mask = self.bins_m.copy()
            # mask[mask < 0.9] = 0
            # mean_per_frame = (dist_per_frame * mask).sum(axis=1)  # (N,)
            return mean_per_frame
        
        f_mean_per_frame = means_per_frame(self.f_hists)  # (N,)
        r_mean_per_frame = means_per_frame(self.r_hists)  # (N,)
        # from worst to better: f - beta * r should be high for good placements
        self.pcentile_order = np.argsort(f_mean_per_frame - self.beta * r_mean_per_frame)
        # self.pcentile_order = np.argsort(-r_mean_per_frame)

        self.R_THRESH = 0.0
        # self.R_THRESH = 0.9
    
    def evaluate(self, pcentile=1.):
        f_hists, r_hists = self._pcentile2hists(pcentile)
        return self._hists2metrics(f_hists, r_hists, prefix=f"SSR ({self.name}) (worst {int(pcentile*100)}%)\n")
    
    def _pcentile2hists(self, pcentile):
        """Percentile at the frame level: chose the worst p% rays"""
        assert 0 <= pcentile <= 1

        count = int(pcentile * len(self.f_hists))
        worst_inds = self.pcentile_order[:count]  # (N')

        ret_f_hists = self.f_hists[worst_inds]  # (N')
        ret_r_hists = self.r_hists[worst_inds]  # (N')

        return ret_f_hists, ret_r_hists

    def _hists2metrics(self, f_hists, r_hists, prefix=""):
        """
        Args:
            f_hists: (np.ndarray, dtype=np.float32, shape=(N, BINS_GT)) collection of histograms of max intensities
                     across columns for the front curtain.
            r_hists: (np.ndarray, dtype=np.float32, shape=(N, BINS_GT)) collection of histograms of max intensities
                     across columns for the random curtain.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        # front curtain
        # average intensity histograms across frams
        f_hists = f_hists.sum(axis=0)  # (BINS_SS,)
        f_hists = f_hists / f_hists.sum()  # (BINS_SS,)
        
        # front curtain histogram
        f_mean = (self.bins_m * f_hists).sum()
        sns.distplot(self.bins_l[1:], bins=self.BINS, kde=False, hist_kws={"weights": f_hists[1:]}, color='b', ax=ax1)

        # random curtain
        # average intensity histograms across frams
        r_hists = r_hists.sum(axis=0)  # (BINS_SS,)
        r_hists = r_hists / r_hists.sum()  # (BINS_SS,)

        # random curtain histogram
        # set random curtain threshold
        r_mean = (self.bins_m * r_hists)[self.bins_m > self.R_THRESH].sum()
        sns.distplot(self.bins_l[1:], bins=self.BINS, kde=False, hist_kws={"weights": r_hists[1:]}, color='r', ax=ax2)
        
        for ax in [ax1, ax2]:
            ax.set_xlabel("max-intensity", fontsize='xx-large')
            ax.tick_params(axis="both", labelsize='x-large')
        ax1.set_ylabel("% rays",  fontsize='xx-large')
        
        title = f"{prefix}Front mean-max-return: {bpc(f_mean, 4)}, Random mean-max-return: {bpc(r_mean, 4)}"
        plt.suptitle(title, fontsize='xx-large')
        plt.tight_layout(rect=[0, 0, 0, 0.1])
        plt.show()


ex = Experiment("eval")

@ex.config
def eval_config():
    agent_name = "BaselineFrontRandomAgent"
    oracle = False


class EvalEngine:
    @ex.capture
    def __init__(self, agent_name, oracle):
        self.oracle = oracle
        self.env = simpy.Environment()
        
        # devices
        self.gt_state_device = SynthiaGTState(self.env, "mini_train", cam=True, colored_pc=True)
        self.light_curtain = LightCurtain(self.env, self.gt_state_device)
        self.manager = EvalManager(self.env, self.gt_state_device, self.light_curtain, oracle=self.oracle)

        if agent_name == "BaselineFrontRandomAgent":
            self.agent = BaselineFrontRandomAgent(self.env, self.manager, self.light_curtain)
        elif agent_name == "OracleAgent":
            self.agent = OracleAgent(self.env, self.manager, self.gt_state_device, self.light_curtain)
        else:
            raise Exception("agent name must be one of [BaselineFrontRandomAgent, OracleAgent]")

        self.devices = [self.gt_state_device, self.light_curtain, self.manager, self.agent]
    
    def clear_simulation(self):
        del self.env
        self.env = simpy.Environment()
        for device in self.devices:
            device.reset(self.env)

    def run_simulation(self, idx):
        """Returns the reward streams for this simulation"""
        # create process
        gt_state_process = self.env.process(self.gt_state_device.process(idx, preload=True))
        agent_process = self.env.process(self.agent.process())
        
        # run simulation
        self.env.run(until=gt_state_process)

        # collect rewards
        gtr_stream = self.manager.gt_reward_stream.copy()
        ssr_stream = self.manager.ss_reward_stream.copy()

        # clear simulation
        self.clear_simulation()

        return gtr_stream, ssr_stream
    
    def evaluate(self):
        store_name = self.agent.__class__.__name__
        if self.oracle:
            store_name += "_Oracle"
        store = Store(store_name)
        
        try:
            store.load()
        except:
            num_videos = len(self.gt_state_device.dataset)

            for idx in range(num_videos):
                print(f"Evaluating {idx + 1}/{num_videos} videos ...")
                gtr_stream, ssr_stream = self.run_simulation(idx)
                
                # gtr stream
                for elem in gtr_stream:
                    store.add_hist(idx, "agent_gtr", elem.data["agent"])
                    if self.oracle:
                        store.add_hist(idx, "oracle_gtr", elem.data["oracle"])
                
                # ssr stream
                for elem in ssr_stream:
                    store.add_hist(idx, "agent_ssr_f", elem.data["agent_f"])
                    store.add_hist(idx, "agent_ssr_r", elem.data["agent_r"])
                    if self.oracle:
                        store.add_hist(idx, "oracle_ssr_f", elem.data["oracle_f"])
                        store.add_hist(idx, "oracle_ssr_r", elem.data["oracle_r"])
            
            store.save()
        
        pcentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

        # GTR reports
        gt_agent_metrics = GTMetrics(store["agent_gtr"], "agent")
        for p in pcentiles: gt_agent_metrics.evaluate(p)
        if self.oracle:
            gt_oracle_metrics = GTMetrics(store["oracle_gtr"], "oracle")
            for p in pcentiles: gt_oracle_metrics.evaluate(p)

        # SSR reports
        ss_agent_metrics = SSMetrics(store["agent_ssr_f"], store["agent_ssr_r"], "agent")
        for p in pcentiles: ss_agent_metrics.evaluate(p)
        if self.oracle:
            ss_oracle_metrics = SSMetrics(store["oracle_ssr_f"], store["oracle_ssr_r"], "oracle")
            for p in pcentiles: ss_oracle_metrics.evaluate(p)

@ex.automain
def evaluate():
    eval_engine = EvalEngine()
    eval_engine.evaluate()
