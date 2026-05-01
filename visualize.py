from agent.dqn_agent import DQNAgent
from environment.grid_world import GridWorld
import os
import time
import numpy as np

env=GridWorld()
agent=DQNAgent()

def visualize(episodes=1 ,delay=0.4):

    agent.load(path="model.pth")
    agent.epsilon=0.0

    for episode in range(episodes):
        state=env.reset()
        done=False
        total_reward=0

        while not done:
            # os.system("cls")
            env.render()
            action=agent.select_action(state)
            # if action==0: print("↑")
            # if action==1: print("↓")
            # if action==2: print("←")
            # if action==3: print("→")
            print(["↑","↓","←","→"][action])
            next_state,reward,done = env.step(action)
            print(reward)
            state=next_state
            total_reward+=reward
            time.sleep(delay)
        
        # os.system("cls")
        env.render()


        print(f"Episode {episode+1}/{episodes} | Step {env.steps} | Action: {['↑','↓','←','→'][action]} | Reward: {reward:+.2f}")


        print(f"Episode {episode+1} | {'Goal Reached ✓' if np.array_equal(env.agent_pos, env.goal_pos) else 'Timed Out ✗'}")
        print(f"Total Reward: {total_reward:.2f}")
        time.sleep(3)

def get_custom_map():
    raw1=input("Enter Agent Position(row,col): ")
    row1,col1=map(int,raw1.split(","))
    agent_pos=[row1,col1]
    raw2=input("Enter Goal Position(row,col): ")
    row2,col2=map(int,raw2.split(","))
    goal_pos=[row2,col2]
    
    if agent_pos == goal_pos:
        print("Agent & Goal can't be same position!")
        return
    
    obstacles=set()
    print("Enter obstacles one by one, press Enter when done")

    while True:
        inp=input("gimme input: ")
        if not inp:
            break
        else:
            row,col=map(int,inp.split(","))
            if (row,col) != tuple(agent_pos) and (row,col) != tuple(goal_pos):
                obstacles.add((row,col))

    env.agent_pos=np.array(agent_pos)
    env.goal_pos=np.array(goal_pos)
    env.obstacles=obstacles
    env.steps=0

    if env.bfs_checker(env.agent_pos, env.goal_pos):
        agent.load("model.pth")
        state = env.get_observation()   # get state without resetting
        done = False
        total_reward = 0

        while not done:
            os.system("cls")
            env.render()
            action = agent.select_action(state)
            print(["↑","↓","←","→"][action])
            next_state, reward, done = env.step(action)
            print(reward)
            state = next_state
            total_reward += reward
            time.sleep(0.1)

        os.system("cls")
        env.render()
        print(f"{'Goal Reached ✓' if np.array_equal(env.agent_pos, env.goal_pos) else 'Timed Out ✗'}")
        print(f"Total Reward: {total_reward:.2f}")
    else:
        print("No valid path! Try different obstacles.")

    


if __name__ == "__main__":
    print("1. Watch agent on random maps")
    print("2. Enter custom map")
    choice = input("Choose (1 or 2): ")
    
    if choice == "1":
        visualize(episodes=10, delay=0.2)
    elif choice == "2":
        get_custom_map()
    else:
        print("Invalid choice")

