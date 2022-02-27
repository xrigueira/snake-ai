import torch
import random
import numpy as np
from collections import deque # data structure
from snakeGameRL import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        
        # more params
        self.n_games = 0
        self.epsilon = 0 # controls the randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if we exceed that mem it will remove elemtns from the left
        self.model = Linear_QNet(11, 256, 3) # layers of the nn as parameters
        self.trainer = QTrainer(self.model, learningRate=LR, gamma=self.gamma)
        # model and trainer
    
    def get_state(self, game):
        
        head = game.snake[0] # head of the snake
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            
            # check if there is danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_r)),
            
            # check if there is danger to the right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_r)),
            
            # check if there is danger to the left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_r)),
            
            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # food location
            game.food.x < game.head.x, # food to the left
            game.food.x > game.head.x, # food to the right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
        ]
        
        return np.array(state, dtype=int) # binarizes
    
    def remember(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        
        if len(self.memory) > BATCH_SIZE:
            
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        
        else:
            
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        
        # random moves: tradeoff explorations / exploitation. More random moves in the beginning to explore
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            
            move = random.randint(0, 2)
            final_move[move] = 1
        
        else:
            
            state0 = torch.tensor(state, dtype=torch.float) # convert to tensor
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
            
    
def train():
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    
    while True:
        
        # get old state
        state_old = agent.get_state(game)
        
        # get move based on the current state
        final_move = agent.get_action(state_old)
        
        # perform move and get state
        done, reward, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            
            # train the long memory (experience memory), plot the results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    
    train()