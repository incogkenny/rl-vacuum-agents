import time

from PPOAgent import PPOAgent
from DQNAgent import DQNAgent
from SpiralAgent import SpiralAgent
from greedy_agent import GreedyAgent
from vacuum_env import VacuumEnv


def test_spiral_agent():
    env = VacuumEnv()
    agent = SpiralAgent(env)
    agent.run()
    env.close()

def test_greedy_agent():
    env = VacuumEnv(3)
    agent = GreedyAgent(env)
    episodes = 10
    total_dirt = len(env.passive_objects)
    total_dirt_collected = []
    total_steps_taken = []

    for i in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, _, done, _ = env.step(action)
            env.render()
        print(f"Episode {i + 1}: Total Dirt Collected: {env.count.dirtCollected}/{total_dirt}, Steps Taken: {env.steps_taken}")
        total_dirt_collected.append(env.count.dirtCollected)
        total_steps_taken.append(env.steps_taken)
    print(f"\nAverage Dirt Collected: {sum(total_dirt_collected)/episodes}/{total_dirt}, Average Steps Taken: {sum(total_steps_taken)/episodes}")
    print(f"Best Dirt Collected: {max(total_dirt_collected)}/{total_dirt}, Least Steps Taken: {min(total_steps_taken)}")

def test_DQNAgent():
    env = VacuumEnv(3)
    state_size = len(env.update_state())  # Number of state features
    action_size = len(env.action_space)  # Number of possible actions
    agent = DQNAgent(state_size, action_size)
    agent.load("dqn_model.pth")  # Load pre-trained model
    agent.epsilon = 0  # Set epsilon to 0 for testing
    print("Model loaded successfully")
    total_dirt = len(env.passive_objects)
    total_dirt_collected = []
    total_steps_taken = []

    episodes = 20

    for i in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            env.render()
        print(f"Episode {i + 1}: Total Dirt Collected: {env.count.dirtCollected}/{total_dirt}, Steps Taken: {env.steps_taken}")
        total_dirt_collected.append(env.count.dirtCollected)
        total_steps_taken.append(env.steps_taken)
    print(f"\nAverage Dirt Collected: {sum(total_dirt_collected)/episodes}/{total_dirt}, Average Steps Taken: {sum(total_steps_taken)/episodes}")
    print(f"Best Dirt Collected: {max(total_dirt_collected)}/{total_dirt}, Least Steps Taken: {min(total_steps_taken)}")

def test_PPOAgent():
    env = VacuumEnv(3, 250, max_steps=1000)
    state_size = len(env.update_state())  # Number of state features
    action_size = len(env.action_space)  # Number of possible actions
    agent = PPOAgent(state_size, action_size)
    agent.load("ppo_model.pth")  # Load pre-trained model
    print("Model loaded successfully")
    total_dirt = len(env.passive_objects)
    total_dirt_collected = []
    total_steps_taken = []

    episodes = 20

    for i in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(state)
            state, reward, done, info = env.step(action)
            env.render()
        print(f"Episode {i + 1}: Total Dirt Collected: {env.count.dirtCollected}/{total_dirt}, Steps Taken: {env.steps_taken}")
        total_dirt_collected.append(env.count.dirtCollected)
        total_steps_taken.append(env.steps_taken)
    print(f"\nAverage Dirt Collected: {sum(total_dirt_collected)/episodes}/{total_dirt}, Average Steps Taken: {sum(total_steps_taken)/episodes}")
    print(f"Best Dirt Collected: {max(total_dirt_collected)}/{total_dirt}, Least Steps Taken: {min(total_steps_taken)}")



if __name__ == "__main__":
    test_DQNAgent()