from agent.dqn_agent import DQNAgent
from environment.grid_world import GridWorld
import os
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()
plt.show(block=False)

env = GridWorld()
agent = DQNAgent()

success_history = []
success_rate_history = []
window = 100

def train():
    recent_successes = []
    best_reward = -float("inf")
    if os.path.exists("model.pth"):
        agent.load("model.pth")
    else:
        print("No saved model found, starting fresh!")
    for episode in range(11001):
        state=env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.buffer.push(state,action,reward,next_state,done)
            if len(agent.buffer) > 1000:
                agent.learn()
            state = next_state 
            total_reward += reward

        if episode > 200:
            agent.decay_epsilon()
        if total_reward > best_reward:
            best_reward = total_reward
        success = np.array_equal(env.agent_pos, env.goal_pos)
        recent_successes.append(1 if success else 0)
        success_history.append(1 if success else 0)
        start = max(0, len(success_history) - window)
        rate = sum(success_history[start:]) / len(success_history[start:])
        success_rate_history.append(rate)
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        if episode % 100 == 0:
            success_rate = sum(recent_successes) / len(recent_successes) * 100
            print(f"Episode {episode} | Epsilon: {agent.epsilon:.3f} | Reward: {total_reward:.2f} | Success Rate: {success_rate:.1f}%")
            agent.save("model.pth")
            ax.clear()
            ax.plot(success_rate_history, label="Success Rate")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Success Rate")
            ax.set_title("Live Training Performance")
            ax.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

if __name__ == "__main__":
    train()