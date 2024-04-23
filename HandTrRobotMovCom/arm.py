import pygame
from pygame.locals import *
import math
import os
import zmq


class Segment:
    def __init__(self, x, y, length, index):
        self.a = pygame.Vector2(x, y)
        self.angle = 0
        self.length = length
        self.b = pygame.Vector2()
        self.parent = None
        self.child = None
        self.stroke_weight = self.map(index, 0, 20, 1, 10)

    def map(self, value, start1, stop1, start2, stop2):
        return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1))

    def calculate_b(self):
        dx = self.length * math.cos(self.angle)
        dy = self.length * math.sin(self.angle)
        self.b = self.a + pygame.Vector2(dx, dy)

    def set_a(self, pos):
        self.a = pos

    def attach_a(self):
        self.set_a(self.parent.b)

    def follow(self, target):
        dir = target - self.a
        if dir.length() > 0:
            self.angle = math.atan2(dir.y, dir.x)
            dir.scale_to_length(self.length)
            dir *= -1
            self.a = target + dir
    
    def print_angle(self):
        print("Angle of end segment:", math.degrees(self.angle))
    

    def update(self):
        self.calculate_b()
        #self.print_angle()
        return int(math.degrees(self.angle))

    def show(self, screen):
        pygame.draw.line(screen, (255, 255, 255), self.a, self.b, int(self.stroke_weight))


pygame.init()

width, height = 600, 400
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

seglength = [45*3, 38*3, 9*3]
start = Segment(300, 200, seglength[0], 0)
current = start

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

angels = [0,0,0] #Kopf, Körper, basis
angCounter = 1



for i in range(2):
    next_segment = Segment(current.b.x, current.b.y, seglength[i+1], i)
    current.child = next_segment
    next_segment.parent = current
    current = next_segment

end = current
base = pygame.Vector2(width / 2, height)

rotation = [0,0,0]


running = True
while running:
    os.system('cls' if os.name == 'nt' else 'clear')
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    screen.fill((51, 51, 51))

    mouse_pos = pygame.mouse.get_pos()

    end.follow(pygame.Vector2(*mouse_pos))
    angels[0] = end.update()

    next_segment = end.parent
    while next_segment is not None:
        next_segment.follow(next_segment.child.a)
        angels[angCounter] = next_segment.update()
        next_segment = next_segment.parent
        angCounter += 1
    angCounter = 1
    
    start.set_a(base)
    start.calculate_b()

    next_segment = start.child
    while next_segment is not None:
        next_segment.attach_a()
        next_segment.calculate_b()
        next_segment = next_segment.child

    end.show(screen)

    next_segment = end.parent
    while next_segment is not None:
        next_segment.show(screen)
        next_segment = next_segment.parent
    
    #Head Arm
    if (angels[0]>=90):
        angels[0] = angels[0]-360
    #Base Arm
    if (angels[2]>=90):
        angels[2] = angels[2]-360   
    #Body Arm
    angels[1] += 90
    angels[0] -= angels[1] 
    if (angels[1]>=180):
    #    angels[1] = angels[1]+90  
        angels[1] = angels[1]-360  
    #    angels[0] -= angels[1] 

    
  
  
    
    print (angels)
    socket.send_string(str(angels))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()