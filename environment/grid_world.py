import numpy as np

class GridWorld:

    def __init__(self,grid_size=10, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.agent_pos=np.array([0,0])
        self.goal_pos=np.array([self.grid_size-1,self.grid_size-1])
        self.steps=0
        return self.get_observation()
    
    def get_observation(self):
        return np.concatenate([self.agent_pos, self.goal_pos])
    
    def step(self, action):
        if action==0:
            self.agent_pos[0]-=1
        elif action==1:
            self.agent_pos[0]+=1
        elif action==2:
            self.agent_pos[1]-=1
        elif action==3:
            self.agent_pos[1]+=1

        self.agent_pos=np.clip(self.agent_pos,0,self.grid_size-1)
        self.steps+=1

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward=10.0
            done=True
        elif self.steps>=self.max_steps:
            reward=-1.0
            done=True
        else:
            reward=-0.1
            done=False
        
        return self.get_observation(), reward, done