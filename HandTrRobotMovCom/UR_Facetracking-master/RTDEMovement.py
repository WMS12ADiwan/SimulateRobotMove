import URBasic
import math
import numpy as np
import sys
import time
from math import pi
import math
from visual_kinematics.RobotSerial import *
from visual_kinematics.RobotTrajectory import *

ACCELERATION = 0.1  # Robot acceleration value
VELOCITY = 0.1 # Robot speed value


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
                math.radians(0),
                math.radians(0))

f = robot.forward(robot_startposition)
xyz = f.t_3_1.reshape([3, 1])
abc = f.r_3


vector = [round(xyz[0][0],4), round(xyz[1][0],4), round(xyz[2][0],4)-0.19, round(abc[0],4), round(abc[1],4), round(abc[2],4)]
print(vector)

#rtde_c.moveJ(robot_startposition, ACCELERATION, VELOCITY )
#rtde_c.moveL(vector, VELOCITY, ACCELERATION)
robot.show()

time.sleep(0.2)




"""FACE TRACKING LOOP ____________________________________________________________________"""

# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
ROBOT_IP = '192.168.2.177'
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

# Move Robot to the midpoint of the lookplane
#robot.movej(q=robot_startposition, a= ACCELERATION, v= VELOCITY )


robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
time.sleep(1) # just a short wait to make sure everything is initialised

try:
    print("starting loop")
    #while True:
    robot.set_realtime_pose(vector)

    print("exiting loop")
except KeyboardInterrupt:
    print("closing robot connection")
    # Remember to always close the robot connection, otherwise it is not possible to reconnect
    robot.close()

except:
    robot.close()