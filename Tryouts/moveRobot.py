#!/usr/bin/env python3
import time
import numpy as np
import os
import random
import rtde_control
import rtde_receive
from math import pi
import math
from visual_kinematics.RobotSerial import *
from visual_kinematics.RobotTrajectory import *

def moveRobot(rotation):
    
    ACCELERATION = 0.1  # Robot acceleration value
    VELOCITY = 0.1 # Robot speed value



    rtde_c = rtde_control.RTDEControlInterface("192.168.2.177")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.2.177")
  
    np.set_printoptions(precision=3, suppress=True)

    dh_params = np.array([[0.089159, .0,  pi/2, .0],
                      [.0, -0.42500, .0, .0],
                      [.0, -0.39225, .0, .0],
                      [0.10915, .0,  pi/2, .0],
                      [0.09465, .0, -pi/2, .0],
                      [0.0823, .0, .0, .0]])

    robot = RobotSerial(dh_params)
    
    robot_startposition = (math.radians(0),
                    math.radians(-90),
                    math.radians(90),
                    math.radians(-90),
                    math.radians(-90),
                    math.radians(0))
    
    theta = rtde_c.getForwardKinematics(robot_startposition)
    f = robot.forward(robot_startposition)
    print("end frame xyz:")
    xyz = f.t_3_1.reshape([3, 1])
    abc = f.r_3
    
    
    vector = [round(xyz[0][0],4), round(xyz[1][0],4), round(xyz[2][0],4)-0.19, round(abc[0],4), round(abc[1],4), round(abc[2],4)]
    print(vector)
    
    rtde_c.moveJ(robot_startposition, ACCELERATION, VELOCITY )
    rtde_c.moveL(vector, VELOCITY, ACCELERATION)
    robot.show()
    input()
    exit()
        
    xyz = np.array([[0.000], [-0.350], [0.500]]) # x, y, z
    abc = np.array(rotation) #(rotation um z achse, rotation y achse, rotation x achse )
    end = Frame.from_euler_3(abc, xyz)
    robot.inverse(end)
    if (robot.is_reachable_inverse):
        if(-50>(robot.axis_values[1]*180/pi)>-130):
            print(" axis values: {0}".format(robot.axis_values*180/pi))
            if(rtde_c.moveJ(robot.axis_values, VELOCITY, ACCELERATION)):
                pass # do something
        else:
            print("Robot out of range")
    else:
        print("Point not reachable")
    if(rtde_r.getRobotStatus() != 3):
        print(rtde_r.getRobotStatus())
        return
    
    
moveRobot([1.61, 1.40, 0.17])