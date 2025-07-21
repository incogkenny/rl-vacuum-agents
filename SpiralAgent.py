import time

import numpy as np
import math

class SpiralAgent:
    def __init__(self, env, row_spacing=50):
        self.env = env
        self.direction = "right"
        self.row_spacing = row_spacing
        self.phase = "sweep"  # sweep → turn1 → down → turn2
        self.forward_count = 0
        self.max_forward = 60  # change for wider rooms
        self.current_heading = 0

    def decide_action(self, obs):
        x, y, heading, _, _ = obs
        self.current_heading = heading * 2 * math.pi  # Convert normalized to radians

        # === PHASE 1: Move Forward ===
        if self.phase == "sweep":
            self.forward_count += 1
            if self.forward_count >= self.max_forward:
                self.forward_count = 0
                self.phase = "turn1"
            return self._align_and_move("right" if self.direction == "right" else "left")

        # === PHASE 2: Turn Down ===
        elif self.phase == "turn1":
            aligned = self._is_facing("down")
            if aligned:
                self.phase = "down"
                return 0  # move forward
            return 2  # turn left to face down

        # === PHASE 3: Move Down One Row ===
        elif self.phase == "down":
            self.phase = "turn2"
            return 0  # move forward one row

        # === PHASE 4: Turn to Resume Sweep ===
        elif self.phase == "turn2":
            target = "left" if self.direction == "right" else "right"
            aligned = self._is_facing(target)
            if aligned:
                self.direction = target
                self.phase = "sweep"
                return 0  # start sweeping new row
            return 2  # turn toward sweep direction

        return 0

    def _is_facing(self, target):
        target_angle = self._target_angle(target)
        diff = abs((self.current_heading - target_angle + math.pi) % (2 * math.pi) - math.pi)
        return diff < 0.2  # within ~11 degrees

    def _target_angle(self, direction):
        return {
            "right": 0,
            "down": math.pi / 2,
            "left": math.pi,
            "up": 3 * math.pi / 2
        }[direction]

    def _align_and_move(self, direction):
        if not self._is_facing(direction):
            return 2  # turn left (you could add logic to pick best turn)
        return 0  # move forward

    def run(self, render=True, max_steps=1000):
        obs = self.env.reset()
        self.env.bot.x, self.env.bot.y = (30,30)  # Set bot's starting location to the top-left corner
        done = False
        total_reward = 0
        step = 0

        while not done and step < max_steps:
            action = self.decide_action(obs)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            step += 1
            if render:
                self.env.render()
                time.sleep(0.01)

        print(f"Spiral agent finished with total reward: {total_reward}")
