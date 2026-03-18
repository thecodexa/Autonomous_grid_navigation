from agent.dqn_agent import ReplayBuffer
import numpy as np

buffer = ReplayBuffer()

state = np.zeros(27)
next_state = np.ones(27)

buffer.push(state, 1, 0.5, next_state, False)

print(len(buffer))