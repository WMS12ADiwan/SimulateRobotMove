#!/usr/bin/env python3
import time
import cv2
import numpy as np
import mediapipe as mp
# for visualizing results
from mediapipe.framework.formats import landmark_pb2
import os
import random
from scipy.spatial.transform import Rotation as R



import rtde_control
import rtde_receive
from math import pi
import math
import math3d as m3d
from visual_kinematics.RobotSerial import *
from visual_kinematics.RobotTrajectory import *
import asyncio
from multiprocessing import Process, Value, Array





def moveRobot(n):
    
    ACCELERATION = 0.2  # Robot acceleration value
    VELOCITY = 0.2 # Robot speed value

    rotation = n.value

    rtde_c = rtde_control.RTDEControlInterface("192.168.2.54")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.2.54")
  
    np.set_printoptions(precision=3, suppress=True)

    dh_params = np.array([[0.089159, 0.000000,  pi/2, 0.],
                      [0.000000, -0.42500, 0.000, 0.],
                      [0.000000, -0.39225, 0.000, 0.],
                      [0.109150, 0.000000,  pi/2, 0.],
                      [0.094650, 0.000000, -pi/2, 0.],
                      [0.082300, 0.000000, 0.000, 0.]])

    robot = RobotSerial(dh_params)

        
        
    xyz = np.array([[0.000], [-0.350], [0.500]]) # x, y, z
    abc = np.array(rotation) #(rotation um z achse, rotation y achse, rotation x achse )
    end = Frame.from_euler_3(abc, xyz)
    robot.inverse(end)
    if (robot.is_reachable_inverse):
        if(-50>(robot.axis_values[1]*180/pi)>-130 and (robot.axis_values[4]*180/pi)>30):
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






























class landmarker_and_result():
   def __init__(self):
      self.result = mp.tasks.vision.HandLandmarkerResult
      self.landmarker = mp.tasks.vision.HandLandmarker
      self.createLandmarker()
   
   def createLandmarker(self):
      # callback function
      def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
         self.result = result

      # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
      options = mp.tasks.vision.HandLandmarkerOptions( 
         base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         num_hands = 1, # track both hands
         min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
         min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
         min_tracking_confidence = 0.3, # lower than value to get predictions more often
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   def detect_async(self, frame):
      # convert np frame to mp image
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # detect landmarks
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   def close(self):
      # close landmarker
      self.landmarker.close()

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         handedness_list = detection_result.handedness
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())

         return annotated_image
   except:
      return rgb_image
   

def count_fingers_raised(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult, maxdist, draw, num):
   """Iterate through each hand, checking if fingers (and thumb) are raised.
   Hand landmark enumeration (and weird naming convention) comes from
   https://developers.google.com/mediapipe/solutions/vision/hand_landmarker."""
   try:
      # Get Data
      hand_landmarks_list = detection_result.hand_landmarks
      # Counter
      numRaised = 0
      # for each hand...
      risingHands = (len(hand_landmarks_list))
      thumb = []
      indexF = []
      middleF = []
      ringF = []
      pinkyF = []
      wrist = []
      currentdist = 0
      greifen = False
      for idx in range(len(hand_landmarks_list)):
         # hand landmarks is a list of landmarks where each entry in the list has an x, y, and z in normalized image coordinates
         hand_landmarks = hand_landmarks_list[idx]
         # for each fingertip... (hand_landmarks 4, 8, 12, and 16)
         for i in range(8,21,4):
            # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
            tip_y = hand_landmarks[i].y
            dip_y = hand_landmarks[i-1].y
            pip_y = hand_landmarks[i-2].y
            mcp_y = hand_landmarks[i-3].y
            if tip_y < min(dip_y,pip_y,mcp_y):
               numRaised += 1
            if (i == 8):
               indexF = [hand_landmarks[i].x, hand_landmarks[i].y, hand_landmarks[i].z]
            if (i == 12):
               middleF = [hand_landmarks[i].x, hand_landmarks[i].y, hand_landmarks[i].z]
            if (i == 16):
               ringF = [hand_landmarks[i].x, hand_landmarks[i].y, hand_landmarks[i].z]
            if (i == 20):
               pinkyF = [hand_landmarks[i].x, hand_landmarks[i].y, hand_landmarks[i].z]
         # for the thumb
         # use direction vector from wrist to base of thumb to determine "raised"
         thumb = [hand_landmarks[4].x, hand_landmarks[4].y, hand_landmarks[4].z]
         wrist = [hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z]
         tip_x = hand_landmarks[4].x
         dip_x = hand_landmarks[3].x
         pip_x = hand_landmarks[2].x
         mcp_x = hand_landmarks[1].x
         palm_x = hand_landmarks[0].x
         

         
         if mcp_x > palm_x:
            if tip_x > max(dip_x,pip_x,mcp_x):
               numRaised += 1
         else:
            if tip_x < min(dip_x,pip_x,mcp_x):
               numRaised += 1

         os.system('cls' if os.name == 'nt' else 'clear')
         if (pinkyF[0] > thumb[0]):
            print("Right Rising Hand")  
         else:
            print("Left Rising Hand")
            
            
         if (numRaised == 0):
            draw.clear()   
            
         print("Thumb: x: {:.2f}%, y: {:.2f}%, z: {:.2f}%".format(thumb[0]*100, thumb[1]*100, thumb[2]*100))
         print("index: x: {:.2f}%, y: {:.2f}%, z: {:.2f}%".format(indexF[0]*100, indexF[1]*100, indexF[2]*100))
         print("middle: x: {:.2f}%, y: {:.2f}%, z: {:.2f}%".format(middleF[0]*100, middleF[1]*100, middleF[2]*100))
         print("ring: x: {:.2f}%, y: {:.2f}%, z: {:.2f}%".format(ringF[0]*100, ringF[1]*100, ringF[2]*100))
         print("pinky: x: {:.2f}%, y: {:.2f}%, z: {:.2f}%".format(pinkyF[0]*100, pinkyF[1]*100, pinkyF[2]*100))
         print("wrist: x: {:.2f}%, y: {:.2f}%, z: {:.2f}%".format(wrist[0]*100, wrist[1]*100, wrist[2]*100))
         

         # Example usage:
         handlandmarks = [wrist, indexF]
         # Assuming hand landmarks are provided as [bottom of hand, finger tip, other points...]

         # Calculate the vectors representing the hand orientation
         palm_vector = [a - b for a, b in zip(handlandmarks[1], handlandmarks[0])]# Vector from bottom of hand to finger tip
         x_axis = np.array([1.0, .0, .0])  # Reference x-axis

         # Calculate rotation around x-axis
         rx = np.arccos(np.dot(palm_vector, x_axis) / (np.linalg.norm(palm_vector) * np.linalg.norm(x_axis)))

         # Calculate the normal vector of the hand plane
         normal_vector = np.cross(palm_vector, x_axis)
         normal_vector /= np.linalg.norm(normal_vector)

         # Calculate rotation around y-axis and z-axis using the normal vector
         ry = np.arccos(np.dot(normal_vector, np.array([.0, 1.0, 0.0])))
         rz = np.arccos(np.dot(normal_vector, np.array([.0, .0, 1.0])))

         # Convert rotation angles to degrees
         #rx = np.degrees(rx)
         #ry = np.degrees(ry)
         #rz = np.degrees(rz)
         print("rotation: rx: {:.2f}, ry: {:.2f}, rz: {:.2f}".format(rx, ry, rz))
         
         num = Value(rx, ry, rz)
         
         
         
         
         
         
         
         
         
         
         
         currentdist = math.sqrt(pow(thumb[0]-indexF[0],2)+pow(thumb[1]-indexF[1],2))
         if(maxdist < currentdist):
            maxdist = currentdist
         print("currentdist: x: {:.2f}, maxdist: {:.2f}, dist: {:.2f}%".format(currentdist, maxdist, (currentdist/maxdist)*100))
         print((maxdist/currentdist)*28)
         if((currentdist) < 0.05):
            print("Jetzt Greifen")
            maxdist = currentdist
            greifen = True
            draw.append([indexF[0], indexF[1]])
         else:
            greifen = False
            
         
            
      # display number of fingers raised on the image
      annotated_image = np.copy(rgb_image)
      height, width, _ = annotated_image.shape
      #print("height: x: {}, width: {}".format(height, width))
      
      
      
      
      
      text_x = int(hand_landmarks[0].x * width) - 100
      text_y = int(hand_landmarks[0].y * height) + 50
      cv2.putText(img = annotated_image, text = str(numRaised) + " Fingers Raised",
                        org = (text_x, text_y), fontFace = cv2.FONT_HERSHEY_DUPLEX,
                        fontScale = 1, color = (0,0,255), thickness = 2, lineType = cv2.LINE_4)
      

      if(greifen == True):
         for x in range(0, len(draw)):
            if(x == 0):                                     
               cv2.circle(annotated_image, (int(draw[x][0]*width),int(draw[x][1]*height)), 10, (0,0,255),-1)
            else:
               cv2.line(annotated_image,(int(draw[x-1][0]*width),int(draw[x-1][1]*height)), (int(draw[x][0]*width),int(draw[x][1]*height)), (255,0,0), 5)

            
      return annotated_image, maxdist
   except:
      return rgb_image, maxdist

def main():
   # access webcam
   cap = cv2.VideoCapture(0)
   #maxdist = [0, 28, 0] # max distance between thumb and index finger, base distance between hand and cam, 
   maxdist = 0
   draw = []
   x = 0
   y = 0
   
   num = Value(.0,.0,.0)
   p = Process(target=moveRobot, args=(num))
   p.start()
   # create landmarker
   hand_landmarker = landmarker_and_result()
   while True:
      # pull frame
      ret, frame = cap.read()
      
      # mirror frame
      frame = cv2.flip(frame, 1)
      # update landmarker results
      hand_landmarker.detect_async(frame)
      # draw landmarks on frame
      frame = draw_landmarks_on_image(frame,hand_landmarker.result)

      # count number of fingers raised
      height, width, _ = frame.shape


      


      frame, maxdist = count_fingers_raised(frame,hand_landmarker.result, maxdist, draw, num)

      # display image
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) == ord('q'):
         break
   
   # release everything
   hand_landmarker.close()
   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()