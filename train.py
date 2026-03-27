from agent.dqn_agent import DQNAgent
from environment.grid_world import GridWorld

env = GridWorld()
agent = DQNAgent()

def train():
    for episode in range(1000):
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
        if episode % 100 == 0:
            print(f"Episode {episode} | Epsilon: {agent.epsilon:.3f} | Total Reward: {total_reward}")

if __name__ == "__main__":
    train()