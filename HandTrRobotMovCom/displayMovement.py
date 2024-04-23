#!/usr/bin/env python3
import threading
import rtde_control
import rtde_receive
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import os
import math
import zmq

angel  = 90


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
   

def count_fingers_raised(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult, socket):
    angel = 90
    try:
        # Get Data
        hand_landmarks_list = detection_result.hand_landmarks
        thumb = []
        wrist = []
        currentdist = 0
        greifen = False
        #for each hand
        for idx in range(len(hand_landmarks_list)):
            # hand landmarks is a list of landmarks where each entry in the list has an x, y, and z in normalized image coordinates
            hand_landmarks = hand_landmarks_list[idx]

            thumb = [hand_landmarks[4].x, hand_landmarks[4].y, hand_landmarks[4].z]
            wrist = [hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z]

        annotated_image = np.copy(rgb_image)
        cv2.line(annotated_image, (0,1000),(1000,1000) , (0, 255, 0) , 9)
        cv2.line(annotated_image, (500,500), (500,1000), (255, 0, 0) , 9) 
        if(len(wrist) == 0):
            #print(len(wrist))
            cv2.line(annotated_image, (500,0), (500,500), (0, 0, 255) , 9)
        else:
            try:
                cv2.line(annotated_image, (int(wrist[0]*1000),int(wrist[1]*1000)), (500,500), (0, 0, 255) , 9) 

                v1 = [wrist[0]*1000-500, wrist[1]*1000-500]
                v2 = [0,-500]
                x = np.array(v1)
                y = np.array(v2)
                unit_x = x / np.linalg.norm(x)
                unit_y = y / np.linalg.norm(y)
                angle_rad = np.arccos(np.dot(unit_x, unit_y))
                angle_deg = np.degrees(angle_rad)
                neg = 1
                if (wrist[0]*1000 >= 500):
                    neg = -1
                angel = angle_deg*neg
                #print(angel)

            except ValueError:
                print("Oops!  That was no valid number.  Try again...")
        return annotated_image,angel
    except:
        return rgb_image,angel


        
def main():
    # access webcam
    cap = cv2.VideoCapture(0)
    # create landmarker
    hand_landmarker = landmarker_and_result()
    img = np.ones((1000, 1000, 3), dtype = np.uint8)
    img = 255* img

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")

    
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

        frame,angel = count_fingers_raised(img, hand_landmarker.result, socket)
        print(angel)
        socket.send_string(str(int(angel)))
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
   
   
   
   
   










