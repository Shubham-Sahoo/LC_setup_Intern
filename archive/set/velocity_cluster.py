import matplotlib.pyplot as plt
import numpy as np
import setcpp

from devices.device import Device
from utils import valid_curtain_behind_frontier
from planner import PlannerRT

class SETVelocity(Device):
    def __init__(self, env, light_curtain, latency=10, debug=False):
        super().__init__(env, capacity=1)
        self.latency = latency
        self._debug = debug

        self.light_curtain = light_curtain

        # options
        self._PAIR_SEPARATION     = 0.5  # 0.5m apart
        self._MAX_RANGE           = 20
        self._NODES_PER_RAY       = 120  # 0.16m apart
        self._EXPANSION           = 0.3
        self._SMOOTHNESS          = 0.05
        self._LC_INTENSITY_THRESH = 200

        self.ranges = np.arange(1, self._NODES_PER_RAY + 1) / self._NODES_PER_RAY * self._MAX_RANGE  # (R,)
        self.thetas = self.light_curtain.lc_device.thetas  # (C,) in degrees and in increasing order in [-fov/2, fov/2]
        self.R = len(self.ranges)
        self.C = self.light_curtain.lc_device.CAMERA_PARAMS["width"]  # number of camera rays
        self.planner_min = PlannerRT(self.light_curtain.lc_device, self.ranges, self.C, maximize=False)
        
        # each curtain is represented as range per camera ray
        self.f_curtain = np.ones([self.C], dtype=np.float32)  # (C,) initial front frontier is at 1m
        self.b_curtain = self.f_curtain - self._PAIR_SEPARATION  # (C,) inital back curtain
    
    @property
    def GROUND_HEIGHT(self):
        return self.light_curtain.GROUND_HEIGHT
    
    ####################################################################################################################
    #region: Helper functions
    ####################################################################################################################
    
    def _update_design_pts_from_lc_image(self, lc_image, design_pts):
        """
        Args:
            lc_image: (np.ndarray, dtype=float32, shape=(H, C, 4))) output of LC device.
                        - Channels denote (x, y, z, intensity).
                        - Pixels that aren't a part of LC return will have NaNs in one of
                        the 4 channels.
                        - Intensity ranges from 0. to 255.
            design_pts: (np.ndarray, dtype=np.float32, shape=(W, 2)) design points that produced this lc_image.
        """
        mask_non_nan_cols = np.logical_not(np.isnan(lc_image).any(axis=(0, 2)))  # (W,)
        
        xz = lc_image[:, mask_non_nan_cols, :][:, :, [0, 2]]  # (H, W', 2)
        assert np.all(xz[[0], :, :] == xz)  # consistency along column
        xz = xz[0]  # (W', 2)

        # update design points
        design_pts[mask_non_nan_cols] = xz


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
        lc_cloud = lc_cloud[lc_cloud[:, 3] >= self._LC_INTENSITY_THRESH / 255]  # (N, 4)

        # remove points that are below GROUND_Y (note that velo_z points upwards)
        lc_cloud = lc_cloud[lc_cloud[:, 2] >= self.GROUND_HEIGHT]  # (N, 4)

        return lc_cloud


    def forecast_points(self, f_lc_cloud, f_lc_timestamp, b_lc_cloud, b_lc_timestamp, forecast_duration):
        assert b_lc_timestamp > f_lc_timestamp

        f_lc_cloud = self._process_lc_cloud(f_lc_cloud)  # (M, 4)
        b_lc_cloud = self._process_lc_cloud(b_lc_cloud)  # (N, 4)

        # print(f"Number of points on back curtain: {len(b_lc_cloud)}")
        seg_points_f = setcpp.euclidean_cluster(f_lc_cloud[:, :3])		# (A,3)
        seg_points_r = setcpp.euclidean_cluster(b_lc_cloud[:, :3])		# (B,3)       

        if len(b_lc_cloud) == 0:
            tail = head = forecasted_point = np.ones([0, 3], dtype=np.float32)
            self.publish(dict(tails=tail, heads=head))
        
        elif ((seg_points_f[0] == -1 or seg_points_r[0] == -1) or (len(seg_points_f)<3000 or len(seg_points_r)<3000)):	# if one point return
            f_centroid = f_lc_cloud[:, :3].mean(axis=0)  # (3,)
            b_centroid = b_lc_cloud[:, :3].mean(axis=0)  # (3,)

            # publish single displacement arrow 
            self.publish(dict(tails=f_centroid, heads=b_centroid))

            velocity = (b_centroid - f_centroid) / (b_lc_timestamp - f_lc_timestamp)  # (3,)
            forecasted_point = b_centroid + velocity * forecast_duration  # (3,)

            # convert point to camera frame
            velo2cam = self.light_curtain.lc_device.TRANSFORMS["wTc"]
            forecasted_point = (np.hstack([forecasted_point, 1]) @ velo2cam.T)[:3]  # (3,)
            forecasted_point = forecasted_point.reshape(1, 3)  # (1, 3)
            

        else:
            seg_points_f = np.resize(seg_points_f,(-1,500,3))          # (,500,3)
            seg_points_r = np.resize(seg_points_r,(-1,500,3))	       # (,500,3)
            points_f = []
            points_r = [] 
            for i in range(seg_points_f.shape[0]):
                points_f.append(seg_points_f[i][seg_points_f[i]!=0])
                points_f[i] = np.reshape(points_f[i],(-1,3))
            for i in range(seg_points_r.shape[0]):
                points_r.append(seg_points_r[i][seg_points_r[i]!=0])
                points_r[i] = np.reshape(points_r[i],(-1,3))
            
                        
            f_centroids = []
            b_centroids = []
            
            for j in range(min(seg_points_f.shape[0],seg_points_r.shape[0])):
                p_f = points_f[j]
                f_centroids.append(p_f.mean(axis=0))  # (N,3)
            for j in range(min(seg_points_f.shape[0],seg_points_r.shape[0])):
                p_r = points_r[j]
                b_centroids.append(p_r.mean(axis=0))  # (N,3)
            
            #print(f_centroids)
            # publish single displacement arrow 
            self.publish(dict(tails=np.array(f_centroids), heads=np.array(b_centroids)))
            
            velocity = np.zeros((min(seg_points_f.shape[0],seg_points_r.shape[0]),3))
            forecasted_point = np.zeros((min(seg_points_f.shape[0],seg_points_r.shape[0]),3))
            for i in range(min(seg_points_f.shape[0],seg_points_r.shape[0])):
                velocity[i] = (b_centroids[i] - f_centroids[i]) / (b_lc_timestamp - f_lc_timestamp) # (N,3)
                forecasted_point[i] = b_centroids[i] + velocity[i] * forecast_duration  # (N,3)

            # convert point to camera frame
            velo2cam = self.light_curtain.lc_device.TRANSFORMS["wTc"]
            for i in range(min(seg_points_f.shape[0],seg_points_r.shape[0])):
                forecasted_point[i] = (np.hstack([forecasted_point[i], 1]) @ velo2cam.T)[:3]  # (N,3,)
                forecasted_point[i] = forecasted_point[i].reshape(-1, 3)  # (N, 3)
        
        return forecasted_point


    def frontier_behind_forecasted_points(self, points):
        """
        Given a set of points forecasted using velocities, compute the frontier that lies behind these points.
        
        Args:
            points: (np.ndarray, shape=(N, 2), dtype=np.float32) forecasted points.
                    - Axis 1 channels denote (x, z) in camera frame.
        Returns:
            frontier: (np.ndarray, shape=(C,), dtype=np.float32) frontier that lies behind points.
        """
        frontier = self._MAX_RANGE * np.ones((self.C,), dtype=np.float32)  # (C,)
        x, z = points[:, 0], points[:, 1]
        r = np.linalg.norm(points, axis=1)  # (N,)
        θ = np.rad2deg(np.arctan2(x, z))  # (N,) in degrees

        # Subtract 1 since we want the indexed range (r_inds) to be less than the point's range r.
        r_inds = (self.ranges.searchsorted(r) - 1).clip(0, self.R-1)  # (N,)
        θ_inds = self.thetas.searchsorted(θ).clip(0, self.C-1)  # (N,)

        sort_inds = np.argsort(r_inds)[::-1]  # (N,) sorting of r_inds in descending order i.e. decreasing r
        r_inds, θ_inds = r_inds[sort_inds], θ_inds[sort_inds]  # (N,) sorted in decreasing order of r
        frontier[θ_inds] = self.ranges[r_inds]

        if self._debug:
            plt.scatter(x, z, c='r')
            fx = frontier * np.sin(np.deg2rad(self.thetas))
            fz = frontier * np.cos(np.deg2rad(self.thetas))
            plt.plot(fx, fz)
            plt.show()

        return frontier
    
    def curtain_to_design_pts(self, curtain):
        """
        Args:
            curtain: (np.ndarray, shape=(C,), dtype=np.float32) range per camera ray
        Returns:
            design_pts: (np.ndarray, shape=(C, 2), dtype=np.float32) design points corresponding to frontier.
                        - Axis 1 channels denote (x, z) in camera frame.
        """
        x = curtain * np.sin(np.deg2rad(self.thetas))
        z = curtain * np.cos(np.deg2rad(self.thetas))
        design_pts = np.hstack([x.reshape(-1, 1), z.reshape(-1, 1)])
        return design_pts
    
    #endregion
    ####################################################################################################################

    def process(self):
        while True:
            ############################################################################################################
            # CALL LIGHT CURTAIN DEVICE: to physically sense front and back curtain
            ############################################################################################################
            
            # Call front light curtain.
            f_design_pts = self.curtain_to_design_pts(self.f_curtain)
            yield self.env.process(self.light_curtain.service(f_design_pts))
            f_lc_timestamp = self.env.now
            f_lc_image = self.light_curtain.stream[-1].data["lc_image"]  # (H, W, 4)
            f_lc_cloud = self.light_curtain.stream[-1].data["lc_cloud"]  # (N, 4)
            assert f_lc_image.shape[1] == self.C
            self._update_design_pts_from_lc_image(f_lc_image, f_design_pts)
            self.f_curtain = np.linalg.norm(f_design_pts, axis=1)  # (C,)
            
            # Call back light curtain.
            b_design_pts = self.curtain_to_design_pts(self.b_curtain)
            yield self.env.process(self.light_curtain.service(b_design_pts))
            b_lc_timestamp = self.env.now
            b_lc_image = self.light_curtain.stream[-1].data["lc_image"]  # (H, W, 4)
            b_lc_cloud = self.light_curtain.stream[-1].data["lc_cloud"]  # (N, 4)
            assert b_lc_image.shape[1] == self.C
            self._update_design_pts_from_lc_image(b_lc_image, b_design_pts)
            self.b_curtain = np.linalg.norm(b_design_pts, axis=1)  # (C,)

            ############################################################################################################
            # Forecast points by computing velocity and compute corresponding frontier
            ############################################################################################################
            
            forecasted_points = self.forecast_points(f_lc_cloud, f_lc_timestamp,
                                                     b_lc_cloud, b_lc_timestamp,
                                                     forecast_duration=self.latency)  # (N, 3)
            
            forecasted_frontier = self.frontier_behind_forecasted_points(forecasted_points)  # (C,)
            
            ############################################################################################################
            # Update front curtain based on forecasted points + expansion
            ############################################################################################################

            expanded_frontier = self.f_curtain + self._EXPANSION  # (C,)
            self.f_curtain = np.minimum(forecasted_frontier, expanded_frontier)  # (C,)

            ############################################################################################################
            # Enforce smoothness in front curtain
            ############################################################################################################

            for i in range(len(self.f_curtain)):
                for j in range(len(self.f_curtain)):
                    if self.f_curtain[i] > self.f_curtain[j]:
                        self.f_curtain[i] = min(self.f_curtain[i],
                                                   self.f_curtain[j] + self._SMOOTHNESS * abs(i-j))

            # Validate curtain
            self.f_curtain = valid_curtain_behind_frontier(self.planner_min, self.f_curtain, debug=self._debug)  # (C,)

            ############################################################################################################
            # Compute back curtain behind front curtain
            ############################################################################################################

            # here we won't use the planner, but just the design points to be epsilon behind the front curtain            
            self.b_curtain = self.f_curtain - self._PAIR_SEPARATION  # (C,)

            ############################################################################################################
            # Timeout for computations
            ############################################################################################################

            yield self.env.timeout(self.latency)


    def reset(self, env):
        super().reset(env)
        self.f_curtain = np.ones([self.C], dtype=np.float32)  # (C,) initial front frontier is at 1m
        self.b_curtain = self.f_curtain - self._PAIR_SEPARATION  # (C,) inital back curtain
