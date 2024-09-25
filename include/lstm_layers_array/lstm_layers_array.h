#ifndef LSTM_LAYER_ARRAY_H_
#define LSTM_LAYER_ARRAY_H_

#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>

#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <hrii_person_tracker/PathArray.h>

#include <dynamic_reconfigure/server.h>
#include <costmap_2d/GenericPluginConfig.h>

#include <geometry_msgs/PolygonStamped.h>


typedef nav_msgs::Path MyPath;
// typedef std::vector<geometry_msgs::Point> MyPath; // this choice requires handling things with std::vector functions
                                                     // for example, one cannot simply do predicted_path_global_frame_navmsgs_Path_ = msg, but fill
                                                     // the array with a for loop and the function push_back().

namespace lstm_layer_array_namespace
{

class LstmLayerArray : public costmap_2d::Layer
{
public:
  LstmLayerArray() : footprint_received_(false) {
    layered_costmap_ = NULL;
  }

  void onInitialize();
  void updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y, double* max_x,
                     double* max_y);
  void updateBoundsFromPredictedTrajectory(double* min_x, double* min_y, double* max_x, double* max_y);
  void updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j);
  void updateCosts_old(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j);
  void bresenhamLine(costmap_2d::Costmap2D* costmap, int x0, int y0, int x1, int y1, int min_i, int min_j, int max_i, int max_j);


private:
    void reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level); // these two make sure updateBounds and updateCosts are run 
    dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig> *dsrv_;         // these two make sure updateBounds and updateCosts are run 
    
    void predictedTrajectoryCallback(const hrii_person_tracker::PathArray& msg);
    ros::Subscriber predicted_trajectory_sub_;
    boost::recursive_mutex lock_;
    
    hrii_person_tracker::PathArray predicted_path_global_frame_PathArray_;
    MyPath predicted_path_global_frame_navmsgs_Path_; // we can use either a nav_msgs::Path or a std::vector<geometry_msgs::Point>
    std::vector<geometry_msgs::PointStamped> predicted_path_global_frame_; // i should have used std::list< > to be compliant with ProxemicLayer.
                                                                           // however, since predicted_path_global_frame_ is supposed to work on
                                                                           // sequential data it makes more sense to use vector instead of list.
                                                                           // to know why, ask ChatGPT to compare the two.

    bool first_time_;
    double last_min_x_, last_min_y_, last_max_x_, last_max_y_;


    bool isPointInFootprint(double x, double y);
    void footprintCallback(const geometry_msgs::PolygonStamped::ConstPtr& msg);
    ros::Subscriber footprint_sub_;
    geometry_msgs::PolygonStamped current_footprint_;
    bool footprint_received_;
};
} // namespace lstm_layer_namespace

#endif // LSTM_LAYER_ARRAY_H