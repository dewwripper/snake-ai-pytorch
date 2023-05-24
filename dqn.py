import pandas as pd
import random
import torch
from environment import Environment
from model import Linear_QNet, QTrainer


BATCH_SIZE = 1000
LR = 0.001

class DQNAgent:

    def __init__(self, environment):
        self.episode = environment.getFileMaxLine()
        self.epsilon = 0.1
        self.gamma = 0.9
        self.model = Linear_QNet(38, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    
    def get_state(self, environment): 
        state = environment.getCurrentState()[0]
        return np.array(state, dtype=int)


    def get_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0,1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()
        return action
    

def train():
    environment = Environment()
    agent = DQNAgent(environment)
    count = 0
    while True:
        state_old = agent.get_state(environment)
        action = agent.get_action(environment)
        reward = environment.getNextState(count)[1]
        state_new = agent.get_state(environment)
        agent.trainer.train_step(state_old, action)
        count = count + 1

