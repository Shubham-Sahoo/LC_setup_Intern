from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
import simpy

from envs.base_env import BaseEnv
from policies.actor import Actor

from data.synthia import Frame
import utils

import setcpp
from devices.light_curtain import LightCurtain
from policies.nn.absolute.features import Features, Featurizer, CurtainInfo
from devices.device import Device

ingr = utils.Ingredient("baseline")
ingr.add_config("config/baseline.yaml")


class arrow(Device):
    def __init__(self, env):
        super().__init__(env, capacity=1)
        
    def pub(self,acc):
        self.publish(acc.velo)
    
    @property
    def streaming(self):
        return self.stream
    
    def reset(self,env):
        self.env = env


class VelocityActor(Actor):
    @ingr.capture
    def __init__(self, transform_w2c, thetas: np.ndarray, latency):
        self._latency = latency

        # parameters
        self._EXPANSION             = 0.3
        self._RECESSION_F           = 0.4  # recession for front curtain
        self._RECESSION_R           = 1.0  # recession for random curtain
        self._SMOOTHNESS            = 0.05
        self._LC_INTENSITY_THRESH_F = 200
        self._LC_INTENSITY_THRESH_R = 200
        self.GROUND_HEIGHT          = -1.2
        self._MAX_RANGE             = 20
        self._publish                = {"tails":np.array([]),"heads":np.array([])}
        self.history: List[NNActor.PrevOCA] = []
        self.featurizer = Featurizer(thetas)
        self.transform_w2c          = transform_w2c
        

    @dataclass
    class PrevOCA:
        o: BaseEnv.Observation  # observation at time t-1
        c: CurtainInfo  # curtain info from observations time t-1
        a: np.ndarray  # action of base policy at time t-1 using o

    @property
    def latency(self) -> float:
        return self._latency

    @property
    def velo(self):
        return self._publish


    def _enforce_smoothness(self, ranges: np.ndarray):
        """NOTE: this is an in-place operation!"""
        for i in range(len(ranges)):
            for j in range(len(ranges)):
                if ranges[i] > ranges[j]:
                    ranges[i] = min(ranges[i], ranges[j] + self._SMOOTHNESS * abs(i - j))

    def init_action(self,
                    state: Frame) -> Tuple[np.ndarray, Optional[np.ndarray], bool, dict]:
        """
        Simulates convergence of this policy in a frozen frame where the front curtain converges to a smoothed out\
        version of the safety envelope
        """
        # compute safety envelope
        ranges = utils.safety_envelope(state)  # (C,)

        # enforce smoothness
        self._enforce_smoothness(ranges)

        return ranges, None, True, {}

    def _process_lc_cloud(self, lc_cloud):
        """
        Perform ground subtraction and intensity thresholding.
        
        Args:
            lc_cloud: (np.ndarray, dtype=np.float32, shape=(N, 4)) lc cloud.
                      Axis 2 corresponds to (x, y, z, i):
                        - x : x in velo frame.
                        - y : y in velo frame.
                        - z : z in velo frame.
                        - i : intensity of LC cloud, lying in [0, 1].
        Returns:
            lc_cloud: (same as above) processed lc cloud.
        """
        # remove points below intensity threshold
        lc_cloud = lc_cloud[lc_cloud[:, 3] >= self._LC_INTENSITY_THRESH_F / 255]  # (N, 4)

        # remove points that are below GROUND_Y (note that velo_z points upwards)
        lc_cloud = lc_cloud[lc_cloud[:, 2] >= self.GROUND_HEIGHT]  # (N, 4)

        return lc_cloud

    def frontier_behind_forecasted_points(self, points,ranges,thetas,C):
        """
        Given a set of points forecasted using velocities, compute the frontier that lies behind these points.
        
        Args:
            points: (np.ndarray, shape=(N, 2), dtype=np.float32) forecasted points.
                    - Axis 1 channels denote (x, z) in camera frame.
        Returns:
            frontier: (np.ndarray, shape=(C,), dtype=np.float32) frontier that lies behind points.
        """
        frontier = self._MAX_RANGE * np.ones((C,), dtype=np.float32)  # (C,)
        x, z = points[:, 0], points[:, 1]
        r = np.linalg.norm(points, axis=1)  # (N,)
        θ = np.rad2deg(np.arctan2(x, z))  # (N,) in degrees
        
        R = len(ranges)
        # Subtract 1 since we want the indexed range (r_inds) to be less than the point's range r.
        r_inds = (ranges.searchsorted(r) - 1).clip(0, R-1)  # (N,)
        θ_inds = thetas.searchsorted(θ).clip(0, C-1)  # (N,)

        sort_inds = np.argsort(r_inds)[::-1]  # (N,) sorting of r_inds in descending order i.e. decreasing r
        r_inds, θ_inds = r_inds[sort_inds], θ_inds[sort_inds]  # (N,) sorted in decreasing order of r
        frontier[θ_inds] = ranges[r_inds]

        #if self._debug:
        #    plt.scatter(x, z, c='r')
        #    fx = frontier * np.sin(np.deg2rad(thetas))
        #    fz = frontier * np.cos(np.deg2rad(thetas))
        #    plt.plot(fx, fz)
        #    plt.show()

        return frontier


    def step(self,

             obs: BaseEnv.Observation) ->Tuple[np.ndarray, Optional[np.ndarray], bool, dict]:

        """
        Args:
            obs (BaseEnv.Observation): observations

        Returns:
            act (np.ndarray, dtype=np.float32, shape=(C,)): sampled actions.
            logp_a (Optional[np.ndarray]): log-probability of the sampled actions. None if sampling is deterministic.
            control (bool): whether this policy had control of taking the action at this timestep.
                            - this will be used to determine whether to evaluate the policy at this frame.
                            - another eg. is that these are the timesteps when nn-based policies will run the network.
            info (dict): auxiliary info generated by policy, such as a vector representation of the observation while
                         generating the action.
        """

        # convert obs to cinfo and get base policy action
        # all these quantities are for current timestep
        cinfo = self.featurizer.obs_to_curtain_info(obs)
        oca = VelocityActor.PrevOCA(obs, cinfo, None)

        f_return, r_return = obs
        if len(self.history) != 0: 
            b_return = self.history[0].o[0]

        else:
            b_return = f_return
            self.history.append(oca)

        
        self.history[0] = oca  # update history

        # Get hits on front curtain
        f_lc_cloud = f_return.lc_cloud.copy()  # (N,4)
        
        f_lc_cloud = self._process_lc_cloud(f_lc_cloud)  # (M, 4)
        #print(f_lc_cloud)

        b_lc_cloud = b_return.lc_cloud.copy()  # (N,4)
        
        b_lc_cloud = self._process_lc_cloud(b_lc_cloud)  # (M, 4)

        # Get hits on random curtain
        r_curtain = r_return.lc_ranges.copy()  # (N,4)
        

        r_hits = utils.hits_from_lc_image(r_return.lc_image, ithresh=self._LC_INTENSITY_THRESH_R)  # (C,)

        feat = self.featurizer.obs_to_curtain_info(obs)
        thetas = feat.t[0]
        ranges = feat.r[0]
        C = self.featurizer.C
        

        seg_points_f = setcpp.euclidean_cluster(f_lc_cloud[:, :3])		# (A,3)
        seg_points_b = setcpp.euclidean_cluster(b_lc_cloud[:, :3])		# (B,3)       


        #if len(b_lc_cloud) == 0:
            #tail = head = forecasted_point = np.ones([0, 3], dtype=np.float32)
            #self.publish(dict(tails=tail, heads=head))
        #self.publish=dict(tails=[1,2,5], heads=[7,9,12])
        

        if ((seg_points_f[0] == -1 or seg_points_b[0] == -1) or (len(seg_points_f)<3000 or len(seg_points_b)<3000)):	# if one point return

            f_centroid = f_lc_cloud[:, :3].mean(axis=0)  # (3,)
            b_centroid = b_lc_cloud[:, :3].mean(axis=0)  # (3,)

            # publish single displacement arrow 
            #self.publish=dict(tails=[1,2,5], heads=[7,9,12])

            self._publish["tails"] = np.array(f_centroid)
            self._publish["heads"] = np.array(b_centroid)

            velocity = (b_centroid - f_centroid) / (self.latency)  # (3,)
            forecasted_point = b_centroid + velocity * self.latency  # (3,)

            # convert point to camera frame
            velo2cam = self.transform_w2c
            forecasted_point = (np.hstack([forecasted_point, 1]) @ velo2cam.T)[:3]  # (3,)
            forecasted_point = forecasted_point.reshape(1, 3)  # (1, 3)
            

        else:

            seg_points_f = np.resize(seg_points_f,(-1,500,3))          # (,500,3)
            seg_points_b = np.resize(seg_points_b,(-1,500,3))	       # (,500,3)
            points_f = []
            points_b = [] 
            for i in range(seg_points_f.shape[0]):
                points_f.append(seg_points_f[i][seg_points_f[i]!=0])
                points_f[i] = np.reshape(points_f[i],(-1,3))
            for i in range(seg_points_b.shape[0]):
                points_b.append(seg_points_b[i][seg_points_b[i]!=0])
                points_b[i] = np.reshape(points_b[i],(-1,3))
            
                        
            f_centroids = []
            b_centroids = []
            
            for j in range(seg_points_f.shape[0]):
                p_f = points_f[j]
                f_centroids.append(p_f.mean(axis=0))  # (K,3)
            for j in range(seg_points_b.shape[0]):
                p_b = points_b[j]
                b_centroids.append(p_b.mean(axis=0))  # (L,3)

            velocity = np.zeros((max(seg_points_f.shape[0],seg_points_b.shape[0]),3))                     # Max (K,L)
            forecasted_point = np.zeros((max(seg_points_f.shape[0],seg_points_b.shape[0]),3))             # Max (K,L)
            for i in range(min(seg_points_f.shape[0],seg_points_b.shape[0])):                             # Min (K,L)
                min_dist = 0.2                                                                     # For velocity < 72 kmph sidewards
                for j in range(seg_points_b.shape[0]):
                    val = b_centroids[j] - f_centroids[i]
                    dist = val[0]*val[0]+val[1]*val[1]+val[2]*val[2]
                    if (min_dist>abs(dist)):
                        min_dist = abs(dist)                                                       # Speed constraint of object
                        min_point = b_centroids[j] - f_centroids[i]
                        min_ind = j
                    
                if min_dist>=0.2:
                    velocity[i] = 0.0
                    forecasted_point[i] = f_centroids[i] - self._RECESSION_F
                else:
                    velocity[i] = min_point / (self.latency) # (N,3)
                    forecasted_point[i] = b_centroids[min_ind] + velocity[i] * self.latency  # (N,3)
                    
                f_ind = i
          
            if f_ind != seg_points_f.shape[0]-1:                                                   # Remaining clusters in front curtain
                for i in range(f_ind+1,seg_points_f.shape[0]):
                    forecasted_point[i] = f_centroids[i] - self._RECESSION_F

            elif f_ind != seg_points_b.shape[0]-1:                                                 # Remaining clusters in back curtain
                for i in range(f_ind+1,seg_points_b.shape[0]):
                    forecasted_point[i] = b_centroids[i] - self._RECESSION_F
            #self.publish=dict(tails=f_centroids, heads=b_centroids)
            self._publish["tails"] = np.array(f_centroids)
            self._publish["heads"] = np.array(b_centroids)
            
            # convert point to camera frame
            velo2cam = self.transform_w2c
            for i in range(max(seg_points_f.shape[0],seg_points_b.shape[0])):
                
                forecasted_point[i] = (np.hstack([forecasted_point[i], 1]) @ velo2cam.T)[:3]  # (N,3,)
                forecasted_point[i] = forecasted_point[i].reshape(-1, 3)  # (N, 3)

        forecasted_frontier = self.frontier_behind_forecasted_points(forecasted_point,ranges,thetas,C)  # (C,)
        
        b_curtain = self.frontier_behind_forecasted_points(b_lc_cloud[:,:2],ranges,thetas,C)
        forecasted_frontier = np.minimum(forecasted_frontier,b_curtain)
        #expanded_frontier = f_curtain + self._EXPANSION  # (C,)
        
        #print(self.publish)

        ############################################################################################################
        # Expand+Recede+Smooth frontier
        ############################################################################################################
        

        n_hits = ~ (r_hits)  # (C,) no hit from either front or random curtain
        # Expansion
        forecasted_frontier[n_hits] += self._EXPANSION

        # Recession
        forecasted_frontier[r_hits] -= self._RECESSION_F
        forecasted_frontier[r_hits] = r_curtain[r_hits] - self._RECESSION_R
        r_curtain[n_hits] = forecasted_frontier[n_hits] + 1        

        forecasted_frontier = np.minimum(forecasted_frontier, r_curtain)  # (C,)

        # Enforce smoothness
        self._enforce_smoothness(forecasted_frontier)


        action, logp_a, control, info = forecasted_frontier, None, True, {}  # logp_a is None since this policy is deterministic
        return action, logp_a, control, info

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
        pass
