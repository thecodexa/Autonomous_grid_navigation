import numpy as np
from collections import deque
import shelve
import random

class GridWorld:

    def __init__(self,grid_size=10, max_steps=200):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        while True:
            ax = random.randint(0, self.grid_size-1)
            ay = random.randint(0, self.grid_size-1)              
            self.agent_pos=np.array([ax,ay])
            gx = random.randint(0, self.grid_size-1)
            gy = random.randint(0, self.grid_size-1)
            while gx==ax and gy == ay:
                gx = random.randint(0, self.grid_size-1)
                gy = random.randint(0, self.grid_size-1)
            self.goal_pos=np.array([gx,gy])
            self.steps=0

            res_loc={(ax,ay),(gx,gy)}
            self.obstacles=set()
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if (i, j) != (ax, ay) and (i, j) != (gx, gy):
                        if(random.random()<0.15):
                            self.obstacles.add((i,j))
            
            self.visited = set()
            self.visited.add(tuple(self.agent_pos))


            if(self.bfs_checker(self.agent_pos,self.goal_pos)):
                return self.get_observation()
            

        
            
    
    def bfs_checker(self, start, goal):
        dq=deque()
        dq.append(start)
        visited = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                row.append(False)
            visited.append(row)
        visited[start[0]][start[1]]=True
        for o in self.obstacles:
            visited[o[0]][o[1]]=True

        
        adjacency=[]
        for r in range(self.grid_size):
            row=[]
            for c in range(self.grid_size):
                row.append([])
            adjacency.append(row)
        with shelve.open("Adjacency_matrix") as db:
            if str(self.grid_size) in db:
                adjacency=db[str(self.grid_size)]
            else:
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                            if( j != self.grid_size-1):
                                adjacency[i][j+1].append((i,j))
                                adjacency[i][j].append((i,j+1))
                                
                            if( i != self.grid_size-1):
                                adjacency[i][j].append((i+1,j))
                                adjacency[i+1][j].append((i,j))
                db[str(self.grid_size)] = adjacency
        
        
      
        while dq:                
            for a in adjacency[dq[0][0]][dq[0][1]]:
                
                if(visited[a[0]][a[1]] == False):
                    dq.append(a)
                    visited[a[0]][a[1]] = True
            dq.popleft()


        if(visited[goal[0]][goal[1]]==False):
            return False
        
        return True

                    


    
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
            reward=-0.15
            done=False
            if new_dist < prev_dist:
                reward += 0.02
            # elif new_dist > prev_dist:
            #     reward -= 0.01

            if hit_obstacle:
                reward -= 0.4
            
            if tuple(self.agent_pos) in self.visited:
                reward -= 0.08
            else:
                self.visited.add(tuple(self.agent_pos))
        
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