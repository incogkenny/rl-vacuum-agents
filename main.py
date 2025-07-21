import time

import numpy as np

from vacuum_env import VacuumEnv

# Initialise the environment
env = VacuumEnv()

# Reset the environment
observation = env.reset()
print(f"Initial State: {observation}")

# Run the environment for a few steps
for step in range(50):
    # Generate a random action (left and right wheel speeds)
    actions = ["forward", "backward", "turn_left", "turn_right"]
    action = np.random.choice(env.action_space)
    print(f"Step {step + 1}, Action: {actions[action]}")

    # Take a step in the environment
    observation, reward, done, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Done: {done}")

    # Render the environment
    env.render()

    time.sleep(0.1)

    # Check if the episode is done
    if done:
        print("Environment is done!")
        break

# Close the environment
env.close()