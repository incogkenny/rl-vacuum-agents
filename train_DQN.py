import time
import logging
from vacuum_env import VacuumEnv
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
from IPython.display import clear_output


# Hyperparameters
EPISODES = 250
BATCH_SIZE = 32

logging.basicConfig(
    filename="DQN_training_log.csv",
    filemode="w",
    format="%(message)s",
    level=logging.INFO
)
logging.info("episode, total_reward, dirt_collected, elapsed_time, total_time")  # CSV header

# Initialise environment and agent
env = VacuumEnv(1, 300, 60,500 )
state_size = len(env.update_state())  # Number of state features
action_size = len(env.action_space)  # Number of possible actions
agent = DQNAgent(state_size, action_size)
total_time = 0
total_actions = 0

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    # if episode <300:
    #     env.bot.x  = 250
    #     env.bot.y  = 250
    total_reward = 0
    done = False
    action_count = {"forward": 0, "backward": 0, "left": 0, "right": 0}

    start_time = time.time()  # Start time for the episode

    while not done:
        # Render the environment (optional)
        env.render()

        # Agent selects an action
        action = agent.act(state)


        # Environment responds to the action
        next_state, reward, done, _ = env.step(action)
        total_actions += 1
        if action == 0:
            action_count["forward"] += 1
        elif action == 1:
            action_count["backward"] += 1
        elif action == 2:
            action_count["left"] += 1
        elif action == 3:
            action_count["right"] += 1

        # Store experience in replay memory
        agent.remember(state, action, reward, next_state, done)

        # Update state and accumulate reward
        state = next_state
        total_reward += reward

        # Train the agent with a batch of experiences
        if total_actions > BATCH_SIZE:
            agent.replay(BATCH_SIZE)

    end__time = time.time()  # End time for the episode
    elapsed_time = end__time - start_time
    total_time += elapsed_time

    # Log training metrics
    logging.info(f"{episode + 1},{total_reward}, {env.count.dirtCollected},{elapsed_time:.2f},{total_time:.2f}")

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


# Convert times to minutes:seconds format
    elapsed_minutes, elapsed_seconds = divmod(int(elapsed_time), 60)
    total_minutes, total_seconds = divmod(int(total_time), 60)

    print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward:0.2f}, "
            f"Dirt Collected: {env.count.dirtCollected}, "
          f"Time Taken: {elapsed_minutes}:{elapsed_seconds:02d}, "
          f"Total Time: {total_minutes}:{total_seconds:02d}")

    print(f"Episode {episode + 1}: {action_count}")

# Save the trained model
print(agent.epsilon)
agent.save("dqn_model.pth")

# Close the environment
env.close()