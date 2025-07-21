import tkinter as tk
import random
import math
import numpy as np
from vacuum_bot import Counter, Dirt, Bot

# This class is adapted from the COMP3004 Lab 4 code
# and is used to create a OpenAi Gym-like environment for the same robot in the lab
# environment. The environment is a grid of dirt and the robot can move around
# and collect dirt. The environment is visualised using tkinter.

class VacuumEnv:
    def __init__(self, env_id=3, detection_radius=250, fov_angle = 90, max_steps=1000):
        self_env_layouts = [500, 700, 1000]
        self.env_size = self_env_layouts[env_id-1]
        self.env_id = env_id
        self.fov_angle = fov_angle
        self.window = tk.Tk()
        self.canvas = self.initialise()
        self.agents, self.passive_objects, self.count = self.create_objects()
        self.bot = self.agents[0]  # Assuming a single bot
        self.action_space = [0, 1, 2, 3]  # forward, backward, left, right
        self.detection = detection_radius
        self.state_space = self.update_state()
        self.max_steps = max_steps
        self.steps_taken = 0
        self.last_action = 5
        self.last_distance = 1
        self.idle_steps = 0
        self.inactive_steps = 0
        self.recently_hit_wall = False


    def initialise(self):
        """
        Dynamically initializes the canvas based on the environment ID.
        """
        self.window.resizable(False, False)
        if self.env_id == 1:
            size = 500
        elif self.env_id == 2:
            size = 700
        elif self.env_id == 3:
            size = 1000
        else:
            raise ValueError("Invalid environment ID")

        self.canvas = tk.Canvas(self.window, width=size, height=size, bg="white")
        self.canvas.pack()
        return self.canvas

    def forward(self):
        '''
        This function moves the bot forward by a fixed amount.
        :return dirt_collected: The amount of dirt collected during movement.
        '''

        self.bot.sl = 10.0
        self.bot.sr = 10.0
        self.move(1)
        self.passive_objects, dirt_collected = self.bot.collectDirt(self.canvas, self.passive_objects, self.count)
        return dirt_collected


    def backward(self):
        '''
        This function moves the bot backward by a fixed amount.
        :return dirt_collected: The amount of dirt collected during movement.
        '''
        self.bot.sl = -10.0
        self.bot.sr = -10.0
        self.move(1)
        self.passive_objects, dirt_collected = self.bot.collectDirt(self.canvas, self.passive_objects, self.count)
        return dirt_collected


    def turn_left(self):
        '''
        This function makes the bot turn left by setting the left wheel speed to a negative value
        and the right wheel speed to a positive value.
        :return dirt_collected: The amount of dirt collected during movement.
        '''
        self.bot.sl = -2.5  # Negative speed for left wheel
        self.bot.sr = 2.5   # Positive speed for right wheel
        self.move(1)
        self.passive_objects, dirt_collected = self.bot.collectDirt(self.canvas, self.passive_objects, self.count)
        return dirt_collected


    def turn_right(self):
        '''
        This function makes the bot turn right by setting the left wheel speed to a positive value
        and the right wheel speed to a negative value.
        :return dirt_collected: The amount of dirt collected during movement.
        '''
        self.bot.sl = 2.5   # Positive speed for left wheel
        self.bot.sr = -2.5  # Negative speed for right wheel
        self.move(1)
        self.passive_objects, dirt_collected = self.bot.collectDirt(self.canvas, self.passive_objects, self.count)
        return dirt_collected


    def reset(self):
        # Reset the environment and return the initial observation/state
        self.canvas.delete("all")
        self.agents, self.passive_objects, self.count = self.create_objects()
        self.bot = self.agents[0]
        self.steps_taken = 0
        return self.update_state()  # Initial observation/state

    def step(self, action):
        # Apply the action and update the environment
        if action == 0:
            dirt_collected = self.forward()
            self.inactive_steps = 0
        elif action == 1:
            dirt_collected = self.backward()
        elif action == 2:
            dirt_collected = self.turn_left()
            if dirt_collected == 0:
                self.inactive_steps += 1
        elif action == 3:
            dirt_collected = self.turn_right()
            if dirt_collected == 0:
                self.inactive_steps += 1

        reward = 0

        if dirt_collected > 0:
            self.inactive_steps = 0

        # Calculate reward (e.g., based on dirt collected) +1 per dirt collected
        reward =  dirt_collected - 0.01 # penalty for action
        if action == 1:
            reward -= 0.1 # penalty for moving backward



        # Add penalty for hitting edges
        if self.recently_hit_wall:
            reward -= 0.5
            self.recently_hit_wall = False

        # Add penalty for inactivity
        if self.inactive_steps > 50:
            reward -= 2.0
            self.inactive_steps = 0


        # Return observation, reward, done, and info
        observation = self.update_state()
        info = {}

        # Small reward for searching
        if observation[3] == 1.0:  # No dirt detected
            if action == 0:
                reward += 0.01
            else:
                reward += 0.005

        if self.last_distance > observation[3]:
            reward += 0.02
        # increment steps taken
        self.steps_taken += 1

        if action == self.last_action:
            reward -= 0.001

        # end if all dirt is collected or max steps reached
        done = (
                self.steps_taken >= self.max_steps or
                self.count.dirtCollected >= self.count.totalDirt
        )
        self.last_action = action

        return observation, reward, done, info

    def move(self, dt):
        """
        Handles the physics of the bot's movement and dynamically adjusts the canvas size based on env_id.
        :param dt: Time step for movement.
        """
        # Dynamically set canvas size based on env_id
        if self.env_id == 1:
            canvas_width = canvas_height = 500
        elif self.env_id == 2:
            canvas_width = canvas_height = 700
        elif self.env_id == 3:
            canvas_width = canvas_height = 1000
        else:
            raise ValueError("Invalid environment ID")

        # Straight line movement
        if self.bot.sl == self.bot.sr:
            self.bot.x += self.bot.sr * math.cos(self.bot.theta)
            self.bot.y += self.bot.sr * math.sin(self.bot.theta)
        else:
            # Rotational movement
            R = (self.bot.ll / 2.0) * ((self.bot.sr + self.bot.sl) / (self.bot.sl - self.bot.sr))
            omega = (self.bot.sl - self.bot.sr) / self.bot.ll
            ICCx = self.bot.x - R * math.sin(self.bot.theta)
            ICCy = self.bot.y + R * math.cos(self.bot.theta)

            cos_omega_dt = math.cos(omega * dt)
            sin_omega_dt = math.sin(omega * dt)

            self.bot.x = cos_omega_dt * (self.bot.x - ICCx) - sin_omega_dt * (self.bot.y - ICCy) + ICCx
            self.bot.y = sin_omega_dt * (self.bot.x - ICCx) + cos_omega_dt * (self.bot.y - ICCy) + ICCy
            self.bot.theta = (self.bot.theta + omega * dt) % (2.0 * math.pi)

        # Constrain bot within canvas boundaries
        if self.bot.x - 25 < 0:  # Left edge
            self.bot.x = 25
            self.bot.sl = self.bot.sr = 0
            self.recently_hit_wall = True
        elif self.bot.x + 25 > canvas_width:  # Right edge
            self.bot.x = canvas_width - 25
            self.bot.sl = self.bot.sr = 0
            self.recently_hit_wall = True

        if self.bot.y - 25 < 0:  # Top edge
            self.bot.y = 25
            self.bot.sl = self.bot.sr = 0
            self.recently_hit_wall = True
        elif self.bot.y + 25 > canvas_height:  # Bottom edge
            self.bot.y = canvas_height - 25
            self.bot.sl = self.bot.sr = 0
            self.recently_hit_wall = True

        # Redraw the bot
        self.canvas.delete(self.bot.name)
        self.canvas.delete("detection")
        self.bot.draw(self.canvas, self.detection, math.radians(self.fov_angle))

    def create_objects(self):

        agents = []
        passive_objects = []
        count = Counter()

        if self.env_id == 1:
            # Environment 1 : Small Open room with Grid of dirt
            size = 500
            no_dirt = 25
            dirt_positions = generate_grid_dirt(no_dirt, 100, 500)
            bot_start = (250, 250) # Centre of canvas

        elif self.env_id == 2:
            # Environment 2 : Medium Room with Clustered dirt
            size = 700
            no_dirt = 20
            dirt_positions = generate_clustered_dirt(no_dirt, (size//4, size//4), 75)
            bot_start = (600, 600)

        elif self.env_id == 3:
            # Environment 3 : Large Room with Randomly Scattered dirt
            size = 1000
            no_dirt = 50
            dirt_positions = generate_random_dirt(no_dirt, 1000)
            bot_start = generate_random_position(1000)

        else:
            raise ValueError("Invalid environment ID")

        # Create dirt objects
        for i, (x,y) in enumerate(dirt_positions):
            dirt = Dirt(f"Dirt_{i}", x, y)
            passive_objects.append(dirt)
            dirt.draw(self.canvas)
        count.totalDirt = no_dirt

        # Place the bot
        bot = Bot("Bot1", passive_objects, count)
        bot.x, bot.y = bot_start
        agents.append(bot)
        bot.draw(self.canvas)


        return agents, passive_objects, count

    def update_state(self):
        # Update the state based on the bot's position and orientation with the distance and angle to the nearest dirt
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        max_distance = math.sqrt(canvas_width**2 + canvas_height**2)

        # Choose detection FOV based on fov_id
        fov = math.radians(self.fov_angle)

        distance, angle = self.bot.detect_dirt(self.detection, fov)
        distance = distance / max_distance if distance is not None else 1.0
        angle = angle  if angle is not None else 0 # Normalize to [-π, π]

        self.state_space =np.array([
            self.bot.x / canvas_width,
            self.bot.y / canvas_height,
            self.bot.get_orientation(),
            distance,
            angle
        ])
        return self.state_space

    def render(self):
        # Render the environment
        self.window.update_idletasks()
        self.window.update()

    def close(self):
        # Close the environment
        self.window.destroy()

def generate_grid_dirt(count, spacing, size):
    dirt = []
    rows = int(size / spacing)
    cols = int(size / spacing)
    for i in range(count):
        x = (i % cols) * spacing + spacing // 2
        y = (i // cols) * spacing + spacing // 2
        dirt.append((x, y))
    return dirt

def generate_clustered_dirt(count, cluster_center, radius):
    dirt = []
    for _ in range(count):
        x = random.randint(cluster_center[0] - radius, cluster_center[0] + radius)
        y = random.randint(cluster_center[1] - radius, cluster_center[1] + radius)
        dirt.append((x, y))
    return dirt

def generate_random_dirt(count, size):
    dirt = []
    for _ in range(count):
        x = random.randint(30, size-30)
        y = random.randint(30, size-30)
        dirt.append((x, y))
    return dirt

def generate_random_position(size):
    return random.randint(30, size-30), random.randint(30, size-30)