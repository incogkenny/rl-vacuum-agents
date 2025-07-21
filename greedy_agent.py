
import time


class GreedyAgent:
    def __init__(self, env):
        self.env = env

    def act(self, state):
        # Extract distance and angle to the nearest dirt
        _, _, _, distance, angle = state

        if distance == 1.0:  # No dirt detected
            return 3  # Turn right to search
        else:
            if abs(angle) < 0.1:  # Dirt is straight ahead
                return 0  # Move forward
            elif angle < 0:  # Dirt is to the left
                return 2  # Turn left
            else:  # Dirt is to the right
                return 3  # Turn right


