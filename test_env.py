from environment.grid_world import GridWorld

# create environment
env = GridWorld(grid_size=5)

# reset environment
obs = env.reset()
print("Initial observation:", obs)

# take some actions
actions = [3, 3, 1, 1]  # right, right, down, down

for i, action in enumerate(actions):
    obs, reward, done = env.step(action)
    print(f"Step {i+1}: obs={obs}, reward={reward}, done={done}")