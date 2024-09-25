#include <lstm_layers_array/lstm_layers_array.h>
#include <pluginlib/class_list_macros.h>

// #include <boost/thread.hpp> // (not needed) to access /usr/include/boost/geometry/strategies/strategy_transform.hpp/transform method
#include <tf2_geometry_msgs/tf2_geometry_msgs.h> // to handle tf2 exceptions

PLUGINLIB_EXPORT_CLASS(lstm_layer_array_namespace::LstmLayerArray, costmap_2d::Layer)

// #include <math.h>
// #include <angles/angles.h>
// #include <geometry_msgs/PointStamped.h>
// #include <algorithm>
// #include <string>

using costmap_2d::NO_INFORMATION;
using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::FREE_SPACE;

namespace lstm_layer_array_namespace
{

void LstmLayerArray::onInitialize() {
    ros::NodeHandle nh("~/" + name_), g_nh;

    current_ = true; // true when all the data in the layer is up to date.
    first_time_ = true;
    predicted_trajectory_sub_ = nh.subscribe("/predicted_trajectory", 1, &LstmLayerArray::predictedTrajectoryCallback, this);

    footprint_sub_ = g_nh.subscribe("/footprint", 1, &LstmLayerArray::footprintCallback, this);

    dsrv_ = new dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>(nh);
    dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>::CallbackType cb = boost::bind(
      &LstmLayerArray::reconfigureCB, this, _1, _2);
    dsrv_->setCallback(cb);
}

void LstmLayerArray::reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level) {
  enabled_ = config.enabled;
}

void LstmLayerArray::footprintCallback(const geometry_msgs::PolygonStamped::ConstPtr& msg)
{
    boost::recursive_mutex::scoped_lock lock(lock_);
    current_footprint_ = *msg;
    footprint_received_ = true;
}

void LstmLayerArray::predictedTrajectoryCallback(const hrii_person_tracker::PathArray& msg) {
    predicted_path_global_frame_PathArray_.path.clear(); 
    predicted_path_global_frame_PathArray_.header = msg.header;

    // Copy the received PathArray into the member variable
    for (const auto& path : msg.path) {
        predicted_path_global_frame_PathArray_.path.push_back(path);
    }
}

void LstmLayerArray::updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y,
                               double* max_x, double* max_y) {
    boost::recursive_mutex::scoped_lock lock(lock_);
    std::string global_frame = layered_costmap_->getGlobalFrameID();
    predicted_path_global_frame_.clear();  // Avoid accumulation of old data in the costmap

    // Iterate through the trajectories (paths)
    for (const auto& path : predicted_path_global_frame_PathArray_.path) {
        // Iterate through the poses within the current path
        for (const auto& pose_stamped : path.poses) {
            geometry_msgs::PointStamped pos_global_frame;
            pos_global_frame.header = pose_stamped.header;
            pos_global_frame.point = pose_stamped.pose.position;

            predicted_path_global_frame_.push_back(pos_global_frame);
        }
    }
    
    updateBoundsFromPredictedTrajectory(min_x, min_y, max_x, max_y);

    if (first_time_) {
        last_min_x_ = *min_x;
        last_min_y_ = *min_y;
        last_max_x_ = *max_x;
        last_max_y_ = *max_y;
        first_time_ = false;
    } else {
        double a = *min_x, b = *min_y, c = *max_x, d = *max_y;
        *min_x = std::min(last_min_x_, *min_x);
        *min_y = std::min(last_min_y_, *min_y);
        *max_x = std::max(last_max_x_, *max_x);
        *max_y = std::max(last_max_y_, *max_y);
        last_min_x_ = a;
        last_min_y_ = b;
        last_max_x_ = c;
        last_max_y_ = d;
    }
}

void LstmLayerArray::updateBoundsFromPredictedTrajectory(double* min_x, double* min_y, double* max_x, double* max_y) {
    for (const auto& point_stamped : predicted_path_global_frame_) {
        const geometry_msgs::Point& point = point_stamped.point;

        // Update the bounds based on the position of each point in the path
        *min_x = std::min(*min_x, point.x);
        *min_y = std::min(*min_y, point.y);
        *max_x = std::max(*max_x, point.x);
        *max_y = std::max(*max_y, point.y);
    }
}

void LstmLayerArray::updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j) {
    boost::recursive_mutex::scoped_lock lock(lock_);
    if (predicted_path_global_frame_PathArray_.path.empty()) return;

    costmap_2d::Costmap2D* costmap = layered_costmap_->getCostmap();
    double res = costmap->getResolution();

    // Iterate through each trajectory (path) in the PathArray
    for (const auto& path : predicted_path_global_frame_PathArray_.path) {
        bool discard_points = false;  // Flag to discard points after footprint intersection

        auto it = path.poses.begin();
        auto end = path.poses.end();
        
        while (it != end) {
            // Get the current point's position
            double cx = it->pose.position.x, cy = it->pose.position.y;
            unsigned int mx1, my1;

            // Check if the point is in the robot's footprint
            geometry_msgs::PointStamped point_in_map;
            point_in_map.header = path.header;
            point_in_map.point = it->pose.position;

            if (isPointInFootprint(point_in_map.point.x, point_in_map.point.y)) {
                discard_points = true;  // Set the flag to discard the remaining points in the trajectory
                break;  // Stop processing this trajectory since the footprint has been intersected
            }

            if (!costmap->worldToMap(cx, cy, mx1, my1)) {
                ++it;  // Skip point if conversion fails
                continue;
            }

            // Process the current trajectory by connecting consecutive points
            auto next_it = std::next(it);
            while (next_it != end && !discard_points) {
                double nx = next_it->pose.position.x, ny = next_it->pose.position.y;
                unsigned int mx2, my2;
                if (isPointInFootprint(next_it->pose.position.x, next_it->pose.position.y)) {
                    discard_points = true;  // Discard points after this one
                    break;
                }

                // Convert the next point to map coordinates
                if (!costmap->worldToMap(nx, ny, mx2, my2)) {
                    ++next_it;  // Skip point if conversion fails
                    continue;
                }

                // Draw a line between the current and next point in the same trajectory
                bresenhamLine(costmap, mx1, my1, mx2, my2, min_i, min_j, max_i, max_j);

                // Move to the next pair of points
                mx1 = mx2;
                my1 = my2;
                ++next_it;
            }

            // Move to the next point in the current trajectory
            it = next_it;
        }
    }
}


bool LstmLayerArray::isPointInFootprint(double x, double y) {
    if (!footprint_received_ || current_footprint_.polygon.points.size() < 3) {
        return false;  // No footprint data available, or invalid polygon
    }

    bool inside = false;
    const auto& points = current_footprint_.polygon.points;
    
    // Ray-casting algorithm to check if the point is inside the polygon
    for (size_t i = 0, j = points.size() - 1; i < points.size(); j = i++) {
        double xi = points[i].x, yi = points[i].y;
        double xj = points[j].x, yj = points[j].y;

        bool intersect = ((yi > y) != (yj > y)) &&
                         (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) {
            inside = !inside;
        }
    }

    return inside;
}


void LstmLayerArray::bresenhamLine(costmap_2d::Costmap2D* costmap, int x0, int y0, int x1, int y1, int min_i, int min_j, int max_i, int max_j) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1; 
    int err = dx + dy, e2;

    while (true) {
        if (x0 >= min_i && x0 < max_i && y0 >= min_j && y0 < max_j) {
            costmap->setCost(x0, y0, costmap_2d::LETHAL_OBSTACLE);
        }

        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

// void LstmLayerArray::updateCosts_old(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j) {
//     boost::recursive_mutex::scoped_lock lock(lock_);
//     if (predicted_path_global_frame_PathArray_.path.empty()) return;

//     costmap_2d::Costmap2D* costmap = layered_costmap_->getCostmap();
//     double res = costmap->getResolution();

//     // Iterate through each trajectory (path) in the PathArray
//     for (const auto& path : predicted_path_global_frame_PathArray_.path) {
//         auto it = path.poses.begin();
//         auto end = path.poses.end();
        
//         while (it != end) {
//             // Get the current point's position
//             double cx = it->pose.position.x, cy = it->pose.position.y;
//             unsigned int mx1, my1;

//             // Convert the current point to map coordinates
//             if (!costmap->worldToMap(cx, cy, mx1, my1)) {
//                 ++it;  // Skip point if conversion fails
//                 continue;
//             }

//             // Process the current trajectory by connecting consecutive points
//             auto next_it = std::next(it);
//             while (next_it != end) {
//                 double nx = next_it->pose.position.x, ny = next_it->pose.position.y;
//                 unsigned int mx2, my2;

//                 // Convert the next point to map coordinates
//                 if (!costmap->worldToMap(nx, ny, mx2, my2)) {
//                     ++next_it;  // Skip point if conversion fails
//                     continue;
//                 }

//                 // Draw a line between the current and next point in the same trajectory
//                 bresenhamLine(costmap, mx1, my1, mx2, my2, min_i, min_j, max_i, max_j);

//                 // Move to the next pair of points
//                 mx1 = mx2;
//                 my1 = my2;
//                 ++next_it;
//             }

//             // Move to the next point in the current trajectory
//             it = next_it;
//         }
//     }
// }
};  // namespace lstm_layer_array_namespace



// #pragma region print_PointStamped
//Convert the path to a string for logging
// std::ostringstream oss;
// for (const auto& pose_stamped : predicted_path_global_frame_navmsgs_Path_.poses){
//     const auto& position = pos_global_frame.point; // pose_stamped.pose.position;
//     oss << " [x: " << position.x << ", y: " << position.y << ", z: " << position.z << "]";}
// Log the message
// ROS_ERROR("%s", oss.str().c_str());
// #pragma endregion

// #pragma region print_PointStamped
// ROS_ERROR("global_frame Position: x=%f, y=%f, z=%f", pos_global_frame.point.x, pos_global_frame.point.y, pos_global_frame.point.z);
// #pragma endregion