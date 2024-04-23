#!/usr/bin/env python3
import rtde_control
import rtde_receive
import time
import numpy as np
import math
import zmq
import os

#Connect to Socket to receive Data from displayMovement
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.subscribe("")

#Connection To UR Robot
rtde_c = rtde_control.RTDEControlInterface("10.40.70.196", 500.0)
rtde_r = rtde_receive.RTDEReceiveInterface("10.40.70.196", 500.0)  

#Robot Arm Parameters for servo control
velocity = 0.1
acceleration = 0.1
dt = 1.0/500  # 2ms
lookahead_time = 0.2
gain = 300


def getAngel():
    return (socket.recv_string())


def move_robot(ang):
    joint_q = (math.radians(180),
                        math.radians(ang[2]),
                        math.radians(ang[1]),
                        math.radians(ang[0]),
                        math.radians(-90),
                        math.radians(0))

    t_start = rtde_c.initPeriod()
    rtde_c.servoJ(joint_q, velocity, acceleration, dt, lookahead_time, gain)
    rtde_c.waitPeriod(t_start)
    print("Angel: ",ang)
    return

        
def main():
    #Init Position
    joint_q = (math.radians(0),
                        math.radians(-90),
                        math.radians(90),
                        math.radians(-180),
                        math.radians(-90),
                        math.radians(0))
    # Move to initial joint position with a regular moveJ
    rtde_c.moveJ(joint_q)
    ang = 0
    angels = [0,0,0]
    counter = 0
    try:
        while True:
            os.system('clear') 
            ang = getAngel()
            print("Angel: ", ang)
            time.sleep(0.002)
            strAngles = ang.split("[")[1].split("]")[0].split(",")
            for i in range(3):
                angels[i] = int(strAngles[i])
            if (counter >= 20):
                move_robot(angels)
                counter = 0
            counter += 1
             
    except KeyboardInterrupt:
        rtde_c.servoStop()
        rtde_c.stopScript()


#print("im Steady:", rtde_c.isSteady())
#print("im Running:", rtde_c.isProgramRunning())
#print("Connected:", rtde_r.isConnected())


if __name__ == "__main__":
   main()
