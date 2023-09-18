from devices.set import gnn_train as gnn
import matplotlib.pyplot as plt
import numpy as np

from devices.device import Device
from planner import PlannerRT

class SETBaselineFront(Device):
    def __init__(self, env, light_curtain, latency=10, debug=False):
        super(SETBaselineFront, self).__init__(env, capacity=1)
        self.latency = latency
        self._debug = debug

        self.light_curtain = light_curtain

        # options
        self._MAX_RANGE           = 20
        self._NODES_PER_RAY       = 120  # 0.16m apart
        self._EXPANSION           = 0.3
        self._RECESSION           = 0.4
        self._SMOOTHNESS          = 0.05
        self._GROUND_Y            = 1.2 # LC return below _GROUND_Y is treated as coming from the ground
        self._MAX_REWARD          = 100
        self._LC_INTENSITY_THRESH = 200

        self.ranges = np.arange(1, self._NODES_PER_RAY + 1) / self._NODES_PER_RAY * self._MAX_RANGE  # (R,)
        self.thetas = self.light_curtain.lc_device.thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]
        self.R = len(self.ranges)
        self.C = self.light_curtain.lc_device.CAMERA_PARAMS["width"]  # number of camera rays
        self.H = self.light_curtain.lc_device.CAMERA_PARAMS["height"]  # height of camera rays
        self.planner = PlannerRT(self.light_curtain.lc_device, self.ranges, self.C)
        
        # frontier is the estimated envelope, represented as range per camera ray
        self.frontier = np.ones([self.C], dtype=np.float32)  # (C,) initial frontier is 1m
        self.x_prev = np.zeros([self.C], dtype=np.float32)
        self.z_prev = np.zeros([self.C], dtype=np.float32)
        self.r_prev = np.zeros([self.C], dtype=np.float32)
        self.theta_prev = np.zeros([self.C], dtype=np.float32)
        self.lc_image_prev = np.zeros([self.H,self.C,4], dtype=np.float32)
        self.policy_prev = np.zeros([self.C], dtype=np.float32)
        self.edge_prev = np.zeros([self.C,4], dtype=np.float32)
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

    def _hits_from_lc_image(self, lc_image):
        """
        Args:
            lc_image: (np.ndarray, dtype=np.float32, shape=(H, C, 4)) lc image.
                      Axis 2 corresponds to (x, y, z, i):
                        - x : x in cam frame.
                        - y : y in cam frame.
                        - z : z in cam frame.
                        - i : intensity of LC cloud, lying in [0, 255].
        Returns:
            hits: (np.ndarray, dtype=np.bool, shape=(C,)) whether there is a hit or not for every camera column.
        """
        hits = np.ones(lc_image.shape[:2], dtype=np.bool)  # (H, C)
            
        # mask out NaN values
        hits[np.isnan(lc_image).any(axis=2)] = 0  # (H, C)

        # mask out pixels below intensity threshold
        hits[lc_image[:, :, 3] < self._LC_INTENSITY_THRESH] = 0

        # mask out pixels that are below GROUND_Y (note that cam_y points downwards)
        hits[lc_image[:, :, 1] > self._GROUND_Y] = 0

        # collect hits across camera columns
        hits = hits.any(axis=0)  # (C,)

        return hits

    def _validate_curtain(self, frontier):
        """Computes a valid curtain that lies strictly behind the current frontier using the planner
        
        Args:
            frontier: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that may not correpsond to a
                       valid curtain.
        Returns:
            curtain: (np.ndarray, dtype=np.float32, shape=(C,)) range per camera ray that correpsonds to a
                      valid curtain.
        """
        # construct umap
        safety_mask = self.ranges.reshape(-1, 1) <= frontier  # (R, C)
        distances = np.abs(self.ranges.reshape(-1, 1) - frontier)  # (R, C)
        safe_reward = self._MAX_REWARD - distances  # (R, C)
        umap = safety_mask * safe_reward  # (R, C)

        design_pts = self.planner.get_design_points(umap)  # (C, 2)
        assert design_pts.shape == (self.C, 2)

        if self._debug:
            self.planner._visualize_curtain_rt(umap, design_pts, show=False)
            new_x, new_z = design_pts[:, 0], design_pts[:, 1]  # (C,)
            thetas = np.arctan2(new_z, new_x)
            r_mod = np.sqrt(new_z*new_z+new_x*new_x)
            old_x, old_z = frontier * np.cos(thetas), frontier * np.sin(thetas)
            plt.plot(old_x, old_z, c='r')
            plt.ylim(0, 30)
            plt.xlim(-10, 10)
            plt.pause(1e-4)
            plt.clf()

        # compute curtain from design points
        curtain = np.linalg.norm(design_pts, axis=1)  # (C,)
        return curtain


    #endregion
    ####################################################################################################################

    def process(self):
        while True:
            ############################################################################################################
            # Place light curtain and get return
            ############################################################################################################
            
            # get light curtain return from the design points
            design_pts = self._design_pts_from_ranges(self.frontier)
            new_x, new_z = design_pts[:, 0], design_pts[:, 1]  # (C,)
            new_theta = np.arctan2(new_z, new_x)
            new_r = np.sqrt(new_z*new_z+new_x*new_x)
            delta_x = new_x - self.x_prev
            delta_z = new_z - self.z_prev
            delta_r = new_r - self.r_prev
            delta_theta = new_theta - self.theta_prev
            self.x_prev = new_x
            self.z_prev = new_z
            self.r_prev = new_r
            self.theta_prev = new_theta
            yield self.env.process(self.light_curtain.service(design_pts))
            lc_image = self.light_curtain.stream[-1].data["lc_image"]  # (H, W, 4)
            assert lc_image.shape[1] == self.C
            #print(lc_image.shape)
            ############################################################################################################
            # Compute hits on camera rays from lc image
            ############################################################################################################
            
            hits = self._hits_from_lc_image(lc_image)  # (C,)
            #print(hits.shape)
            #hits = graph_pred
            #print(hits)
            ############################################################################################################
            # Expand+Receed+Smooth frontier
            ############################################################################################################

            # Expansion + Recession
            self.frontier = self.frontier + (1 - hits) * self._EXPANSION
            self.frontier = self.frontier -      hits  * self._RECESSION

            # Enforce smoothness
            for i in range(len(self.frontier)):
                for j in range(len(self.frontier)):
                    if self.frontier[i] > self.frontier[j]:
                        self.frontier[i] = min(self.frontier[i], self.frontier[j] + self._SMOOTHNESS * abs(i-j))
            
            # Validate curtain
            self.frontier = self._validate_curtain(self.frontier)
            #print(self.frontier.shape)
            res_policy = self.frontier - self.policy_prev
            edge_features_t = np.vstack((delta_x,delta_z,delta_r,delta_theta)).T
            lc_image = np.nan_to_num(lc_image)
            #print(res_policy)
            x_t1 = lc_image[:,:,1]
            z_t1 = lc_image[:,:,2]
            r_t1 = np.sqrt(x_t1*x_t1+z_t1*z_t1)
            theta_t1 = np.arctan2(z_t1, x_t1)
            delta_xt1 = [x_t1[:,i]-x_t1[:,i+1] for i in range(x_t1.shape[1]-1)]
            delta_zt1 = [z_t1[:,i]-z_t1[:,i+1] for i in range(z_t1.shape[1]-1)]
            delta_rt1 = [r_t1[:,i]-r_t1[:,i+1] for i in range(r_t1.shape[1]-1)]
            delta_thetat1 = [theta_t1[:,i]-theta_t1[:,i+1] for i in range(theta_t1.shape[1]-1)]         

            edge_t1 = np.vstack((delta_xt1,delta_zt1,delta_rt1,delta_thetat1))

            x_t2 = self.lc_image_prev[:,:,1]
            z_t2 = self.lc_image_prev[:,:,2]
            r_t2 = np.sqrt(x_t2*x_t2+z_t2*z_t2)
            theta_t2 = np.arctan2(z_t2, x_t2)
            delta_xt2 = [x_t2[:,i]-x_t2[:,i+1] for i in range(x_t2.shape[1]-1)]
            delta_zt2 = [z_t2[:,i]-z_t2[:,i+1] for i in range(z_t2.shape[1]-1)]
            delta_rt2 = [r_t2[:,i]-r_t2[:,i+1] for i in range(r_t2.shape[1]-1)]
            delta_thetat2 = [theta_t2[:,i]-theta_t2[:,i+1] for i in range(theta_t2.shape[1]-1)]         

            edge_t2 = np.vstack((delta_xt2,delta_zt2,delta_rt2,delta_thetat2))


            graph_pred = gnn.lc_train(lc_image[:,:,3],self.lc_image_prev[:,:,3],res_policy,edge_features_t,edge_t1,edge_t2)
            #graph_pred = np.reshape(graph_pred,hits.shape)
            self.lc_image_prev = lc_image
            self.policy_prev = self.policy_prev + graph_pred
            self.edge_prev = edge_features_t
            #self.frontier = self.policy_prev + graph_pred
            print("New Updated")
            #print(self.frontier)
            ############################################################################################################
            # Timeout for computations
            ############################################################################################################

            yield self.env.timeout(self.latency)

    
    def reset(self, env):
        super(SETBaselineFront, self).reset(env)
        self.frontier = np.ones([self.C], dtype=np.float32)  # (C,) initial frontier is 1m
