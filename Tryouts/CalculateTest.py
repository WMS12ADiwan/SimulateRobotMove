import math
import cv2
import numpy as np

def printAngle(a,b,c):  
      
    # Square of lengths be a2, b2, c2  
    a2 = a**2  
    b2 = b**2  
    c2 = c**2  
  
    # length of sides be a, b, c  
    a = math.sqrt(a2);  
    b = math.sqrt(b2);  
    c = math.sqrt(c2);  
  
    # From Cosine law  
    alpha = math.acos((b2 + c2 - a2) /
                         (2 * b * c));  
    betta = math.acos((a2 + c2 - b2) / 
                         (2 * a * c));  
    gamma = math.acos((a2 + b2 - c2) / 
                         (2 * a * b));  
  
    # Converting to degree  
    alpha = alpha * 180 / math.pi;  
    betta = betta * 180 / math.pi;  
    gamma = gamma * 180 / math.pi;  
  
    return alpha, betta, gamma



def calculate_endpoint(start_point, angle, length):
    # Convert angle from degrees to radians
    angle_rad = math.radians(angle)
    
    # Calculate the change in x and y coordinates
    delta_x = length * math.cos(angle_rad)
    delta_y = length * math.sin(angle_rad)
    
    # Calculate the end point
    x_end = start_point[0] + delta_x
    y_end = start_point[1] + delta_y
    
    return x_end, y_end


def main():
    lengA = 486 
    lengB = 514 
    lengC = 990
    AngA, AngB, AngC = printAngle(lengA,lengB,lengC)
    startpoint = [1000,1000]
    endpoint = calculate_endpoint(startpoint, int(AngB), lengB)
    img = np.ones((1000, 2000, 3), dtype = np.uint8)
    img = 255* img
    print (AngA, AngB, AngC)
    print(endpoint)
    cv2.line(img,(int(startpoint[0]), int(startpoint[1])),(int(endpoint[0]),int(endpoint[1])-1000) , (0, 255, 0) , 9)
    endpoint = calculate_endpoint(startpoint, 180-AngB, lengC)
    cv2.line(img,(int(startpoint[0]), int(startpoint[1])),(int(endpoint[0]),int(endpoint[1])-1000) , (0, 255, 0) , 9)

    while True:
        cv2.imshow('frame',img)
        if cv2.waitKey(1) == ord('q'):
            break
   
    # release everything
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   main()
   
   
   