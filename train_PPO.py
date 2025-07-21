import time
import logging
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from vacuum_env import VacuumEnv
from PPOAgent import PPOAgent  # Make sure this matches your PPOAgent file

# Hyperparameters
EPISODES = 250
MAX_STEPS = 1000

# Logging setup
logging.basicConfig(
    filename="PPO_training_log.csv",
    filemode="w",
    format="%(message)s",
    level=logging.INFO
)
logging.info("episode, total_reward, dirt_collected, elapsed_time")

# Create environment and agent
env = VacuumEnv(1, detection_radius=250, max_steps=MAX_STEPS)
state_size = len(env.update_state())
action_size = len(env.action_space)
agent = PPOAgent(state_size, action_size)
total_time = 0

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    action_count = {"forward": 0, "backward": 0, "left": 0, "right": 0}
    start_time = time.time()

    while not done:
        env.render()
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        if action == 0:
            action_count["forward"] += 1
        elif action == 1:
            action_count["backward"] += 1
        elif action == 2:
            action_count["left"] += 1
        elif action == 3:
            action_count["right"] += 1

        # Estimate current state value
        with torch.no_grad():
            _, value = agent.model(torch.FloatTensor(state).unsqueeze(0).to(agent.device))
        value = value.item()

        agent.store((state, action, log_prob, reward, done, value))
        total_reward += reward
        state = next_state


    agent.update()

    elapsed_time = time.time() - start_time
    total_time += elapsed_time

    logging.info(f"{episode + 1},{total_reward:.2f},{env.count.dirtCollected},{elapsed_time:.2f}")

    # Keep history for plotting
    if episode == 0:
        reward_history = []
        dirt_history = []

    reward_history.append(total_reward)
    dirt_history.append(env.count.dirtCollected)

    # Live plot
    clear_output(wait=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(reward_history, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(dirt_history, label="Dirt Collected", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Dirt Cleaned")
    plt.title("Dirt Cleaned per Episode")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    elapsed_minutes, elapsed_seconds = divmod(int(elapsed_time), 60)
    total_minutes, total_seconds = divmod(int(total_time), 60)
    print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward:0.2f}, "
          f"Dirt Collected: {env.count.dirtCollected}, "
          f"Time Taken: {elapsed_minutes}:{elapsed_seconds:02d}, "
          f"Total Time: {total_minutes}:{total_seconds:02d}")

    print(f"Episode {episode + 1}: {action_count}")


# Save trained model
agent.save("ppo_model.pth")

# Close environment
env.close()
