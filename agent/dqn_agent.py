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
            torch.FloatTensor(np.array(next_states)),
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
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.00001)
        self.target_network.load_state_dict(self.policy_network.state_dict()) #target network should have the same weights as policy network at the beginning.
        
        self.target_update_freq = 100  # copy every 100 steps
        self.learn_step = 0            # counter to track steps

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

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

        q_values = self.policy_network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  #.gather() - Pick one value per row based on the action index.


        #applying the Bellman equation
        with torch.no_grad():    #We don't want to compute gradients for the target values.target network should not learn.
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]   #we only need values as max returns both values and indices.
            target_q = rewards + self.gamma * max_next_q * (1 - dones)  #(1 - dones) :- If done is 1 means the episode has ended, we don't add future rewards.


        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad() #clear previous gradients
        loss.backward()  #compute gradients
        self.optimizer.step()  #update weights

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.update_target_network()


    def save(self, path="model.pth"):
        torch.save({
            "policy_network": self.policy_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "epsilon":        self.epsilon,
            "learn_step":     self.learn_step,
                }, path)
        print(f"Model saved → {path}")

    def load(self, path="model.pth"):
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon    = checkpoint["epsilon"]
        self.learn_step = checkpoint["learn_step"]
        print(f"Model loaded ← {path}")


