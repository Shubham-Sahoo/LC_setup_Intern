#include "planning.h"

using namespace lc;

// ----------- UTILS -----------
float deg2rad(float deg) {
    return deg * PI / 180;
}
float rad2deg(float rad) {
    return rad * 180 / PI;
}
// -----------------------------

template<bool MAX>
Planner<MAX>::Planner(std::shared_ptr<DatumProcessor> datumProcessor,
                      const std::vector<float>& ranges,
                      std::shared_ptr<Interpolator> interpolator,
                      bool debug)
                      : datumProcessor_(datumProcessor), ranges_(ranges), interpolator_(interpolator), debug_(debug){
    std::shared_ptr<Datum> c_datum = datumProcessor_->getCDatum("camera01");  // assume only one camera named camera01
    Laser laser = c_datum->laser_data["laser01"];  // assume only one camera named camera01

    camera_angles_ = datumProcessor_->getCDatum("camera01")->valid_angles;
    num_camera_rays_ = camera_angles_.size();
    num_nodes_per_ray_ = ranges_.size();
    max_d_las_angle_ = laser.laser_limit * laser.laser_timestep;
    laser_to_cam_ = laser.laser_to_cam;

    if (debug_) {
        std::cout << std::setprecision(4)
                  << "PYLC_PLANNER: Max change in laser angle: " << max_d_las_angle_ << "°" << std::endl;
    }

    constructGraph();
}

template<bool MAX>
void Planner<MAX>::constructGraph() {
    // Add nodes in the graph.
    for (int ray_i = 0; ray_i < num_camera_rays_; ray_i++)
        for (int range_i = 0; range_i < num_nodes_per_ray_; range_i++) {
            float r = ranges_[range_i];
            float theta_cam = camera_angles_[ray_i];
            float x = r * std::sin(deg2rad(theta_cam));
            float z = r * std::cos(deg2rad(theta_cam));

            // Compute laser angle.
            Eigen::Vector4f xyz1_cam(x, 0.0f, z, 1.0f);
            Eigen::Vector4f xyz1_las = laser_to_cam_ * xyz1_cam;
            float x_las = xyz1_las(0), z_las = xyz1_las(2);
            float theta_las = rad2deg(std::atan2(x_las, z_las));

            std::pair<int, int> k = interpolator_->getCmapIndex(x, z, r, theta_cam, theta_las, ray_i, range_i);
            int ki = k.first, kj = k.second;

            graph_[ray_i][range_i].fill(x, z, r, theta_cam, theta_las, ki, kj);
        }

    // Add edges in the graph.
    for (int ray_i = 0; ray_i < num_camera_rays_ - 1; ray_i++) {
        Node* ray_prev = graph_[ray_i];
        Node* ray_next = graph_[ray_i + 1];

        for (int prev_i = 0; prev_i < num_nodes_per_ray_; prev_i++) {
            Node &node_prev = ray_prev[prev_i];
            for (int next_i = 0; next_i < num_nodes_per_ray_; next_i++) {
                Node &node_next = ray_next[next_i];

                float d_theta_las = node_next.theta_las - node_prev.theta_las;
                bool is_neighbor = (-max_d_las_angle_ <= d_theta_las) && (d_theta_las <= max_d_las_angle_);
                if (is_neighbor)
                    node_prev.edges.emplace_back(ray_i + 1, next_i);
            }
        }
    }
}

template<bool MAX>
std::vector<std::pair<float, float>> Planner<MAX>::optimizedDesignPts(Eigen::MatrixXf cmap) {
    // Check if cmap shape is as expected.
    if (!interpolator_->isCmapShapeValid(cmap.rows(), cmap.cols()))
        throw std::invalid_argument(std::string("PYLC_PLANNER: Unexpected cmap shape (")
                                    + std::to_string(cmap.size())
                                    + std::string(")."));

    // Backward pass.
    for (int ray_i = num_camera_rays_ - 1; ray_i >= 0; ray_i--) {
        for (int range_i = 0; range_i < num_nodes_per_ray_; range_i++) {
            Node* pNode = &(graph_[ray_i][range_i]);

            if (ray_i == num_camera_rays_ - 1) {
                // For last ray, the trajectory starts and ends at the same node.
                dp_[ray_i][range_i] = Trajectory<MAX>(pNode, cmap);
            } else {
                // For non-last ray, iterate over all its valid neighbors to select best sub-trajectory.
                for (int edge_i = 0; edge_i < pNode->edges.size(); edge_i++) {
                    std::pair<int, int>& edge = pNode->edges[edge_i];
                    Trajectory<MAX>* pSubTraj = &(dp_[edge.first][edge.second]);
                    Trajectory<MAX> traj(pNode, pSubTraj, cmap);
                    if (edge_i == 0 || traj > dp_[ray_i][range_i])
                        dp_[ray_i][range_i] = traj;
                }
            }
        }
    }

    // Select overall best trajectory.
    Trajectory<MAX> best_traj = dp_[0][0];
    for (int range_i = 1; range_i < num_nodes_per_ray_; range_i++)
        if (dp_[0][range_i] > best_traj)
            best_traj = dp_[0][range_i];

    if (debug_) {
        std::cout << std::fixed << std::setprecision(3)
                  << "PYLC_PLANNER: Optimal cost         : " << best_traj.cost << std::endl
                  << "              Optimal laser penalty: " << best_traj.las << std::endl
                  ;
    }

    // Forward pass.
    std::vector<std::pair<float, float>> design_pts;
    while (true) {
        // Current design point.
        design_pts.emplace_back(best_traj.pNode->x, best_traj.pNode->z);

        if (!best_traj.pSubTraj)  // trajectory ends here
            break;

        best_traj = *(best_traj.pSubTraj);
    }

    return design_pts;
}

template <bool MAX>
std::vector<std::vector<std::pair<Node, int>>> Planner<MAX>::getVectorizedGraph() {
    std::vector<std::vector<std::pair<Node, int>>> m;

    // Copy 2D array to matrix.
    for (int ray_i = 0; ray_i < num_camera_rays_; ray_i++) {
        m.emplace_back();
        for (int range_i = 0; range_i < num_nodes_per_ray_; range_i++) {
            Node& node = graph_[ray_i][range_i];
            m[ray_i].emplace_back(node, node.edges.size());
        }
    }
    return m;
}

template <bool MAX>
Planner<MAX>::~Planner() = default;

Node::Node() = default;

Node::~Node() {
    edges.clear();
}

void Node::fill(float x_, float z_, float r_, float theta_cam_, float theta_las_, long ki_, long kj_) {
    x = x_;
    z = z_;
    r = r_;
    theta_cam = theta_cam_;
    theta_las = theta_las_;
    ki = ki_;
    kj = kj_;

    edges = std::vector<std::pair<int, int>>();
}

template<bool MAX>
Trajectory<MAX>::Trajectory() {
    pNode = nullptr;
    pSubTraj = nullptr;
    cost = MAX ? -INF : INF;  // for maximization: init with -INF; for minimization: init with +INF
    las = 0.0f;
}

template<bool MAX>
Trajectory<MAX>::Trajectory(Node* pNode_, const Eigen::MatrixXf& cmap) {
    // Start node.
    pNode = pNode_;

    // Sub-trajectory.
    pSubTraj = nullptr;

    // Cost.
    if ((pNode->ki != -1) && (pNode->kj != -1))
        cost = cmap(pNode->ki, pNode->kj);
    else
        cost = MAX ? -INF : INF;

    // Laser penalty.
    las = 0.0f;
}

template<bool MAX>
Trajectory<MAX>::Trajectory(Node* pNode_, Trajectory* pSubTraj_, const Eigen::MatrixXf& cmap)
                           : Trajectory(pNode_, cmap) {
    // Start Node : delegated.

    // Sub-trajectory.
    pSubTraj = pSubTraj_;

    // Uncertainty.
    // Initialized from delegation.
    cost += pSubTraj->cost;

    // Laser angle penalty : sum of squares of laser angle changes.
    float d_theta_cam = pSubTraj->pNode->theta_las - pNode->theta_las;
    las = d_theta_cam * d_theta_cam + pSubTraj->las;
}

// Traj1 < Traj2 means that Traj1 is WORSE that Traj2
template<bool MAX>
bool Trajectory<MAX>::operator<(const Trajectory& t) {
    if (cost == t.cost)
        return las > t.las;

    if (MAX)
        return cost < t.cost;
    else
        return cost > t.cost;
}

// Traj1 > Traj2 means that Traj1 is BETTER that Traj2
template<bool MAX>
bool Trajectory<MAX>::operator>(const Trajectory& t) {
    if (cost == t.cost)
        return las < t.las;

    if (MAX)
        return cost > t.cost;
    else
        return cost < t.cost;
}

template<bool MAX>
Trajectory<MAX>::~Trajectory() = default;

CartesianNNInterpolator::CartesianNNInterpolator(int cmap_w, int cmap_h,
                                                 float x_min, float x_max, float z_min, float z_max) {
    cmap_w_ = cmap_w;
    cmap_h_ = cmap_h;
    x_min_ = x_min;
    x_max_ = x_max;
    z_min_ = z_min;
    z_max_ = z_max;
}

std::pair<int, int> CartesianNNInterpolator::getCmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const {
    // This function assumes that the cost map is a 2D grid with evenly spaced xs and zs.
    // It also assumes that X and Z are increasing with a fixed increment.

    float x_incr = (x_max_ - x_min_) / float(cmap_h_ - 1);
    float z_incr = (z_max_ - z_min_) / float(cmap_w_ - 1);

    // Convert to pixel coordinates.
    long ki = std::lround((x - x_min_) / x_incr);
    long kj = std::lround((z - z_min_) / z_incr);

    if (ki < 0 || ki >= cmap_h_)
        ki = -1;  // means that this is outside the cmap grid

    if (kj < 0 || kj >= cmap_w_)
        kj = -1;  // means that this is outside the cmap grid

    return {ki, kj};
}

bool CartesianNNInterpolator::isCmapShapeValid(int nrows, int ncols) const {
    return (nrows == cmap_h_) && (ncols == cmap_w_);
}

PolarIdentityInterpolator::PolarIdentityInterpolator(int num_camera_rays, int num_ranges)
    : num_camera_rays_(num_camera_rays), num_ranges_(num_ranges) {}

std::pair<int, int> PolarIdentityInterpolator::getCmapIndex(float x, float z, float r, float theta_cam, float theta_las, int ray_i, int range_i) const {
    return {range_i, ray_i};
}

bool PolarIdentityInterpolator::isCmapShapeValid(int nrows, int ncols) const {
    return (nrows == num_ranges_) && (ncols == num_camera_rays_);
}
