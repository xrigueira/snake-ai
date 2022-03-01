import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

# Initiate
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Define RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 1024

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        
        self.w = w
        self.h = h
        
        # initiate display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x-BLOCK_SIZE, self.head.y), Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        
        self.food = Point(x, y)
        
        if self.food in self.snake:
            
            self._place_food()
    
    def play_step(self, action):
        self.frame_iteration += 1
        # First, get user input
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                
                pygame.quit()
                quit()          
        
        # Second, move
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Third, check if there is game over
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            
            game_over = True
            reward = -10
            return game_over, reward, self.score
        
        # Fourth, place new food or just move
        if self.head == self.food:
            
            self.score += 1
            reward = 10
            self._place_food()
        
        else:
            
            self.snake.pop()
        
        # Fifth, update user interface and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # Sixth, return game over and score
        return game_over, reward, self.score
    
    def is_collision(self, pt=None):
        
        if pt is None:
            
            pt = self.head
        
        # if it hits and edge
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # if it eats itself
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _update_ui(self):
        
        self.display.fill(BLACK)
        
        for pt in self.snake:
            
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render('Score: ' + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction) # get the index of the current direction
        
        if np.array_equal(action, [1, 0, 0]):
            
            new_dir = clock_wise[idx]
        
        elif np.array_equal(action, [0, 1, 0]):
            
            next_idx = (idx + 1) % 4 # module for to return to the beginning of the list (clock_wise)
            new_dir = clock_wise[next_idx]
        
        else:
            
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
            
        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT:
            
            x += BLOCK_SIZE
        
        elif self.direction == Direction.LEFT:
            
            x -= BLOCK_SIZE
        
        elif self.direction == Direction.DOWN:
            
            y += BLOCK_SIZE
        
        elif self.direction == Direction.UP:
            
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)

                    
                    
        