import sys

from utilities import Logger

from rclpy.time import Time

from utilities import euler_from_quaternion, calculate_angular_error, calculate_linear_error
from rclpy.node import Node
from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry as odom

from sensor_msgs.msg import Imu
from kalman_filter import kalman_filter

from rclpy import init, spin, spin_once, qos

import numpy as np
import message_filters

rawSensors=0
kalmanFilter=1
odom_qos=QoSProfile(reliability=2, durability=2, history=1, depth=10)

motion_type = "point"
Q_CONST = 0.1
R_CONST = 0.9

class localization(Node):
    
    def __init__(self, type, dt, loggerName="robotPose.csv", loggerHeaders=["imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w","kf_x", "kf_y","stamp"]):

        super().__init__("localizer")
        
        data = f"V2-{motion_type}-Q{Q_CONST}-R{R_CONST}".replace(".", "")
        self.loc_logger=Logger(f"CSVs/robot_pose-{data}.csv", loggerHeaders)

        self.pose=None
        
        if type==rawSensors:
            self.initRawSensors()
        elif type==kalmanFilter:
            self.initKalmanfilter(dt)
        else:
            print("We don't have this type for localization", sys.stderr)
            return  

    def initRawSensors(self):
        self.create_subscription(odom, "/odom", self.odom_callback, qos_profile=odom_qos)
        
    def initKalmanfilter(self, dt):
        
        # TODO Part 3: Set up the quantities for the EKF (hint: you will need the functions for the states and measurements)
        x= [0, 0, 0, 0, 0, 0] # Initial state
        
        Q= Q_CONST * np.array([[1,0,0,0,0,0], # Q array is Q constant multiplied by 6x6 I matrix from state vector x=[x,y,th,w,v,vdot]
                            [0,1,0,0,0,0],
                            [0,0,1,0,0,0],
                            [0,0,0,1,0,0],
                            [0,0,0,0,1,0],
                            [0,0,0,0,0,1]])

        R= R_CONST * np.array([[1,0,0,0], # R array is R constant multiplied by 4x4 I matrix from odometry and IMU z=[v,w,ax,ay]
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1],])
        
        P = Q # Initially set P equal to Q
        
        self.kf=kalman_filter(P,Q,R, x, dt) # Initialize Kalman Filter

        TurtleBot = 4 # Set TurtleBot type
        if TurtleBot == 3: # If simulation, set appropriate QoS profile
            odom_qos=QoSProfile(
                reliability = qos.ReliabilityPolicy.RELIABLE,
                durability = qos.DurabilityPolicy.VOLATILE,
                history = qos.HistoryPolicy.KEEP_LAST, # terminal output says UNKNOWN, I think we defaulted to this last time
                depth = 10
            )
        else: # If physical model, set appropriate QoS profile
            print("TB4")
            odom_qos=QoSProfile(
                reliability = qos.ReliabilityPolicy.BEST_EFFORT,
                durability = qos.DurabilityPolicy.VOLATILE,
                history = qos.HistoryPolicy.KEEP_LAST,
                depth = 10
            )
        
        # TODO Part 3: Use the odometry and IMU data for the EKF
        self.odom_sub=message_filters.Subscriber(self, odom, "/odom", qos_profile=odom_qos) # Create odometry subscriber
        self.imu_sub=message_filters.Subscriber(self, Imu, "/imu", qos_profile=odom_qos) # Create IMU subscriber 
        
        time_syncher=message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size=10, slop=0.1)
        time_syncher.registerCallback(self.fusion_callback)
    
    def fusion_callback(self, odom_msg: odom, imu_msg: Imu):
        # TODO Part 3: Use the EKF to perform state estimation
        # Take the measurements
        # your measurements are the linear velocity and angular velocity from odom msg
        # and linear acceleration in x and y from the imu msg
        # the kalman filter should do a proper integration to provide x,y and filter ax,ay
        z=[
            np.sqrt(odom_msg.twist.twist.linear.x**2 + odom_msg.twist.twist.linear.y**2), # v
            odom_msg.twist.twist.angular.z, # w
            imu_msg.linear_acceleration.x, #ax
            imu_msg.linear_acceleration.y, #ay
        ]
        
        # Implement the two steps for estimation
        self.kf.predict() # Prediction step
        self.kf.update(z) # Correction step
        
        # Get the estimate
        xhat=self.kf.get_states() # Get next state estimate 

        # Update the pose estimate to be returned by getPose
        self.pose=np.array([xhat[0], xhat[1], xhat[2], odom_msg.header.stamp])

        # "imu_ax", "imu_ay", "kf_ax", "kf_ay",
        # "kf_vx","kf_w","kf_x", "kf_y","stamp"

        # x, y, th, w, v, vdot=self.x

        # TODO Part 4: log your data
        self.loc_logger.log_values([
                            z[2], # ax
                            z[3], # ay
                            xhat[5], # EKF ax
                            xhat[4]*xhat[3], # EKF ay
                            xhat[4]*np.cos(xhat[2]), # EKF v
                            xhat[3], # EKF w
                            xhat[0], # EKF x
                            xhat[1], # EKF y
                            odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec*1e-9 # Time stamp
                            ])
      
    def odom_callback(self, pose_msg):
        
        self.pose=[ pose_msg.pose.pose.position.x,
                    pose_msg.pose.pose.position.y,
                    euler_from_quaternion(pose_msg.pose.pose.orientation),
                    pose_msg.header.stamp]

    # Return the estimated pose
    def getPose(self):
        return self.pose


if __name__=="__main__":
    
    init()
    
    LOCALIZER=localization()
    
    spin(LOCALIZER)
