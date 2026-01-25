from environment.grid_world import GridWorld


env = GridWorld(grid_size=5)


obs = env.reset()
# print("Initial observation:", obs)
print("Observation length:", len(obs))
print("Observation:", obs)
env.render()


actions = [3, 3,2, 1, 1, 3, 1]

# obs = env.get_observation()
# print("New goal direction:", obs[-2:])
# env.render()
# print("Observation:", obs)
for i, action in enumerate(actions):
    obs, reward, done = env.step(action)
    env.render()
    print("Reward:",reward)
    # print(f"Step {i+1}: obs={obs}, reward={reward}, done={done}")