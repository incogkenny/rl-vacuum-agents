import random

class RandomWalkAgent:
    def __init__(self, action_space):
        """
        Initializes the Random Walk Agent.
        :param action_space: List of possible actions the agent can take.
        """
        self.action_space = action_space

    def act(self, state):
        """
        Selects a random action from the action space.
        :param state: The current state of the environment (not used in random walk).
        :return: A random action.
        """
        return random.choice(self.action_space)