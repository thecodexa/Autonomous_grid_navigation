import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque



class QNetwork(nn.Module):
    def __init__(self, state_size=27, action_size=4):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)
    

class DQNAgent:
    def __init__(self):
        self.policy_network=QNetwork()
        self.target_network=QNetwork()
        self.buffer=ReplayBuffer()
        self.epsilon=1.0
        self.epsilon_decay=0.995
        self.epsilon_min=0.05
        self.gamma=0.99
        self.batch_size=64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,3)
        else:
            with torch.no_grad():
                output=self.policy_network(torch.FloatTensor(state).unsqueeze(0))
                return output.argmax(dim=1).item()
                
    def decay_epsilon(self):
        self.epsilon=max(self.epsilon*self.epsilon_decay,self.epsilon_min)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            print(f"Not enough data in buffer: {len(self.buffer)}/{self.batch_size}")
            return 
        
        states,actions,rewards,next_states,dones=self.buffer.sample(self.batch_size)

