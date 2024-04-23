import math
import cv2
import numpy as np
import vector
import matplotlib.pyplot as plt


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


class Segment:
    def __init__(self, x, y, length_, angle_):
        self.a = vector.VectorObject2D(x = 0, y= 0)
        self.b = vector.VectorObject2D(x = 0, y= 0)
        self.a.x = x
        self.a.y = y
        self.angel = math.radians(angle_)
        self.length = length_
    
    def follow(self, mx,my):
        target = vector.VectorObject2D(mx, my)
        base = vector.VectorObject2D.subtract(target, self.a)
        angle = np.arctan2(base.y/base.x)
    
    def calcB(self):
        dx = self.length * math.cos(self.angel)
        dy = self.length * math.sin(self.angel)
        print(self.angel)
        print(math.cos(self.angel),math.sin(self.angel))
        self.b.x = self.a.x + dx
        self.b.y = self.a.y + dy
        
    def update(self):
        self.calcB()
    
    def show(self):
        plt.plot([self.a.x, self.b.x], [self.a.y, self.b.y])
        plt.plot([0, 0, 2000], [1000, 0, 0],)
        plt.plot([0, 0, 2000], [0, 1000, 0], 'o')
        plt.show()

def on_move(event):
    if event.inaxes:
        return event.xdata, event.ydata
        #print(f'data coords {event.xdata} {event.ydata},',
        #      f'pixel coords {event.x} {event.y}')

def main():
    lengA = 95
    lengB = 391  
    lengC = 425
    startpoint = [1000,0]

    seg = Segment(startpoint[0], startpoint[1], lengC, 45) # x,y, length, angel
    binding_id = plt.connect('motion_notify_event', on_move)
    seg.follow(mouseX, mouseY)
    seg.update()
    seg.show()


if __name__ == "__main__":
   main()
   
   
   