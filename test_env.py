from environment.grid_world import GridWorld

# create environment
env = GridWorld(grid_size=5)

# reset environment
obs = env.reset()
# print("Initial observation:", obs)
print("Observation length:", len(obs))
print("Observation:", obs)
env.render()

# take some actions
actions = [3, 3, 1, 1]  # right, right, down, down
env.step(3)  # move right
env.step(1)  # move right
env.step(1)  # move right
env.step(1)  # move right
env.step(3)  # move right
obs = env.get_observation()
print("New goal direction:", obs[-2:])
env.render()
print("Observation:", obs)
# for i, action in enumerate(actions):
#     obs, reward, done = env.step(action)
#     env.render()
    # print(f"Step {i+1}: obs={obs}, reward={reward}, done={done}")