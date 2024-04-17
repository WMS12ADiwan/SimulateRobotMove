import pygame
from pygame.locals import *
import math
import os

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
        self.print_angle()

    def show(self, screen):
        pygame.draw.line(screen, (255, 255, 255), self.a, self.b, int(self.stroke_weight))


pygame.init()

width, height = 600, 400
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

start = Segment(300, 200, 100, 0)
current = start

for i in range(2):
    next_segment = Segment(current.b.x, current.b.y, 100, i)
    current.child = next_segment
    next_segment.parent = current
    current = next_segment

end = current
base = pygame.Vector2(width / 2, height)

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    screen.fill((51, 51, 51))

    mouse_pos = pygame.mouse.get_pos()

    end.follow(pygame.Vector2(*mouse_pos))
    end.update()

    next_segment = end.parent
    while next_segment is not None:
        next_segment.follow(next_segment.child.a)
        next_segment.update()
        next_segment = next_segment.parent

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
        
    os.system('cls' if os.name == 'nt' else 'clear')

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
