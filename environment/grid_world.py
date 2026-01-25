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

        self.obstacles={(1,2),(2,2),(3,2)}

        return self.get_observation()
    
    def get_observation(self):
        view_size=5
        radius=view_size//2

        local_view=[]

        ax, ay=self.agent_pos
        gx, gy=self.goal_pos

        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                x=ax+dx
                y=ay+dy

                if x<0 or x>=self.grid_size or y<0 or y>=self.grid_size:
                    local_view.append(-1)
                elif (x, y) in self.obstacles:
                    local_view.append(-1)
                elif x==gx and y==gy:
                    local_view.append(1)
                else:
                    local_view.append(0)

        goal_dx=gx-ax
        goal_dy=gy-ay

        return np.array(local_view + [goal_dx, goal_dy],dtype=np.float32)
    
    def step(self, action):

        prev_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        old_pos = self.agent_pos.copy()
        hit_obstacle = False

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
        if tuple(self.agent_pos) in self.obstacles:
            self.agent_pos = old_pos
            hit_obstacle = True 

        new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])   

        if np.array_equal(self.agent_pos, self.goal_pos):       
            reward=10.0
            done=True
        elif self.steps>=self.max_steps:
            reward=-1.0
            done=True
        else:
            reward=-0.1
            done=False
            if new_dist < prev_dist:
                reward += 0.05
            elif new_dist > prev_dist:
                reward -= 0.05

            if hit_obstacle:
                reward -= 0.2
        
        return self.get_observation(), reward, done
    
    def render(self):
        grid = [["." for _ in range(self.grid_size)]
                for _ in range(self.grid_size)]

        for (x, y) in self.obstacles:
            grid[x][y] = "X"

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        grid[ax][ay] = "A"
        grid[gx][gy] = "G"

        print("\nGridWorld:")
        for row in grid:
            print(" ".join(row))