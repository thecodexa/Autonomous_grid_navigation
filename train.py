from agent.dqn_agent import DQNAgent
from environment.grid_world import GridWorld
import os
import numpy as np

env = GridWorld()
agent = DQNAgent()

def train():
    recent_successes = []
    best_reward = -float("inf")
    if os.path.exists("model.pth"):
        agent.load("model.pth")
    else:
        print("No saved model found, starting fresh!")
    for episode in range(10001):
        state=env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.buffer.push(state,action,reward,next_state,done)
            agent.learn()
            state = next_state 
            total_reward += reward

        agent.decay_epsilon()
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("model.pth")
        success = np.array_equal(env.agent_pos, env.goal_pos)
        recent_successes.append(1 if success else 0)
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        if episode % 100 == 0:
            success_rate = sum(recent_successes) / len(recent_successes) * 100
            print(f"Episode {episode} | Epsilon: {agent.epsilon:.3f} | Reward: {total_reward:.2f} | Success Rate: {success_rate:.1f}%")
            agent.save("model.pth")

if __name__ == "__main__":
    train()