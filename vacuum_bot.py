import tkinter as tk
import random
import math
import numpy as np

from DQNAgent import DQNAgent
from rule_based import rule_based_logic


# The Brain class controls the bot's behavior and decision-making
class Brain:

    def __init__(self,botp, mode="rule-based"): # Change mode for different agents
        # Initialise the Brain with a reference to the bot
        self.bot = botp
        self.mode = mode
        self.algorithm = None # Placeholder for DQN or other algorithms
        self.turningCount = 0 # how long bot should turn
        self.movingCount = random.randrange(50,100) # how long bot should move straight
        self.currentlyTurning = False
        self.map = self.bot.map() # get map of dirt from bots sensors


    def thinkAndAct(self, x, y, sl, sr, count):
        """
        Determines the bot's next action based on the current mode.
        Returns the updated wheel speeds and optional position overrides.
        :param x: Current x position of the bot.
        :param y: Current y position of the bot.
        :param sl: Current left wheel speed
        :param sr: Current right wheel speed
        :param count: Dirt collected so far.
        :return:
        """
        if self.mode == "rule-based":
            return rule_based_logic(self, x, y, sl, sr, count)
        elif self.mode == "dqn":
            if self.algorithm is None:
                self.algorithm = DQNAgent(self.bot.map.shape[0], 3)  # Example state size and action size
            # Stub for DQN logic
            action = self.algorithm.act(self.bot.map.flatten())
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

# The Bot class represents the robot and handles its movement, sensors, and interactions
class Bot:

    def __init__(self,namep,passiveObjectsp,counterp):
        self.name = namep  # Name of the bot
        self.x = random.randint(100, 900)  # Initial x-coordinate
        self.y = random.randint(100, 900)  # Initial y-coordinate
        self.theta =  random.uniform(0.0, 2.0 * math.pi)  # Initial orientation (angle in radians)
        self.ll = 60  # Axle width (distance between wheels)
        self.sl = 0.0  # Speed of the left wheel
        self.sr = 0.0  # Speed of the right wheel
        self.passiveObjects = passiveObjectsp  # List of passive objects (e.g., dirt)
        self.counter = counterp  # Counter for collected dirt

    # Think and act method to be called by the brain
    def thinkAndAct(self, agents, passiveObjects):
        """
       Calls the bot's brain to decide the next action and updates the bot's wheel speeds or position.
       :param agents: List of all agents in the environment.
       :param passiveObjects: List of passive objects in the environment.
       """
        self.sl, self.sr, xx, yy = self.brain.thinkAndAct\
            (self.x, self.y, self.sl, self.sr, self.counter.dirtCollected)
        if xx != None:
            self.x = xx
        if yy != None:
            self.y = yy
        
    def setBrain(self,brainp):
        self.brain = brainp

    #returns the result from the ceiling-mounted dirt camera
    def map(self):
        bot_map = np.zeros((10, 10), dtype=np.int16)
        for p in self.passiveObjects:
            if isinstance(p,Dirt):
                xx = int(math.floor(p.centreX/100.0))
                yy = int(math.floor(p.centreY/100.0))
                bot_map[xx][yy] += 1
        return bot_map

    def distance_to(self, obj):
        xx,yy = obj.getLocation()
        return math.sqrt( math.pow(self.x-xx,2) + math.pow(self.y-yy,2) )

    def distanceToRightSensor(self,lx,ly):
        return math.sqrt( (lx-self.sensorPositions[0])*(lx-self.sensorPositions[0]) + \
                          (ly-self.sensorPositions[1])*(ly-self.sensorPositions[1]) )

    def distanceToLeftSensor(self,lx,ly):
        return math.sqrt( (lx-self.sensorPositions[2])*(lx-self.sensorPositions[2]) + \
                            (ly-self.sensorPositions[3])*(ly-self.sensorPositions[3]) )

    def detect_dirt(self, detection_radius=100, cone_angle=math.pi / 4):
        """
        Detects the nearest dirt within a specified radius and cone in front of the bot.
        :param detection_radius: The radius within which to detect dirt.
        :param cone_angle: The half-angle of the detection cone in radians (e.g., π/4 for 45 degrees).
        :return: A tuple (distance, angle) to the nearest dirt, or (None, None) if no dirt is found.
        """
        nearest_dirt = None
        min_distance = float('inf')
        angle_to_dirt = None

        for obj in self.passiveObjects:
            if isinstance(obj, Dirt):
                # Calculate distance to the dirt
                distance = self.distance_to(obj)
                if distance <= detection_radius:
                    # Calculate angle to the dirt
                    dx = obj.centreX - self.x
                    dy = obj.centreY - self.y
                    angle = math.atan2(dy, dx) - self.theta
                    angle = (angle + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]

                    # Check if the dirt is within the cone
                    if abs(angle) <= cone_angle:
                        if distance < min_distance:
                            min_distance = distance
                            nearest_dirt = obj
                            angle_to_dirt = angle

        return min_distance if nearest_dirt else None, angle_to_dirt

    def get_orientation(self):
        """
        Returns the current orientation of the bot as an angle in radians normalised.
        :return: The bot's orientation (theta).
        """
        return self.theta % (2 * math.pi)

    # what happens at each timestep
    def update(self,canvas,passiveObjects,dt):
        self.move(canvas,dt)

    # draws the robot at its current position
    def draw(self, canvas, detection_radius=250, fov_angle=math.radians(90)):
        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)
        sin_theta_perp = math.sin((math.pi / 2.0) - self.theta)
        cos_theta_perp = math.cos((math.pi / 2.0) - self.theta)



        # Convert to degrees
        fov_angle_deg = math.degrees(fov_angle)

        # Adjust orientation and reference direction
        start_angle = -math.degrees(self.theta) - fov_angle_deg / 2  # Rotate clockwise from bot heading

        # Bounding box
        bbox = (
            self.x - detection_radius, self.y - detection_radius,
            self.x + detection_radius, self.y + detection_radius
        )

        canvas.create_arc(
            bbox,
            start=start_angle,
            extent=fov_angle_deg,
            outline="cyan",
            width=2,
            style="arc",
            tags="detection"
        )


# --- Draw bot body ---
        points = [
            self.x + 30 * sin_theta - 30 * sin_theta_perp,
            self.y - 30 * cos_theta - 30 * cos_theta_perp,
            self.x - 30 * sin_theta - 30 * sin_theta_perp,
            self.y + 30 * cos_theta - 30 * cos_theta_perp,
            self.x - 30 * sin_theta + 30 * sin_theta_perp,
            self.y + 30 * cos_theta + 30 * cos_theta_perp,
            self.x + 30 * sin_theta + 30 * sin_theta_perp,
            self.y - 30 * cos_theta + 30 * cos_theta_perp,
            ]
        canvas.create_polygon(points, fill="blue", tags=self.name)

        # --- Calculate sensor positions ---
        self.sensorPositions = [
            self.x + 20 * sin_theta + 30 * sin_theta_perp,
            self.y - 20 * cos_theta + 30 * cos_theta_perp,
            self.x - 20 * sin_theta + 30 * sin_theta_perp,
            self.y + 20 * cos_theta + 30 * cos_theta_perp,
            ]

        # --- Draw bot center ---
        canvas.create_oval(
            self.x - 16, self.y - 16, self.x + 16, self.y + 16,
            fill="gold", tags=self.name
        )

        # --- Draw wheels ---
        wheel_positions = [
            (self.x - 30 * sin_theta, self.y + 30 * cos_theta),  # Left wheel
            (self.x + 30 * sin_theta, self.y - 30 * cos_theta),  # Right wheel
        ]
        for wx, wy in wheel_positions:
            canvas.create_oval(
                wx - 3, wy - 3, wx + 3, wy + 3,
                fill="red" if wx < self.x else "green", tags=self.name
            )

        # --- Draw sensors ---
        for i in range(0, len(self.sensorPositions), 2):
            canvas.create_oval(
                self.sensorPositions[i] - 3, self.sensorPositions[i + 1] - 3,
                self.sensorPositions[i] + 3, self.sensorPositions[i + 1] + 3,
                fill="yellow", tags=self.name
            )


    # handles the physics of the movement
    # cf. Dudek and Jenkin, Computational Principles of Mobile Robotics
    def move(self, canvas, dt):
        if self.sl == self.sr:  # Straight line movement
            self.x += self.sr * math.cos(self.theta)
            self.y += self.sr * math.sin(self.theta)
        else:
            R = (self.ll / 2.0) * ((self.sr + self.sl) / (self.sl - self.sr))
            omega = (self.sl - self.sr) / self.ll
            ICCx = self.x - R * math.sin(self.theta)
            ICCy = self.y + R * math.cos(self.theta)

            cos_omega_dt = math.cos(omega * dt)
            sin_omega_dt = math.sin(omega * dt)

            self.x = cos_omega_dt * (self.x - ICCx) - sin_omega_dt * (self.y - ICCy) + ICCx
            self.y = sin_omega_dt * (self.x - ICCx) + cos_omega_dt * (self.y - ICCy) + ICCy
            self.theta = (self.theta + omega * dt) % (2.0 * math.pi)

            # Define canvas boundaries
            canvas_width = 1000
            canvas_height = 1000

            # Keep bot within boundaries
            if self.x - 30 < 0: # left edge
                self.x = 30
                self.sl = 0
                self.sr = 0
            elif self.x + 30 > canvas_width: # right edge
                self.x = canvas_width - 30
                self.sl = 0
                self.sr = 0

            if self.y - 30 < 0: # top edge
                self.y = 30
                self.sl = 0
                self.sr = 0
            elif self.y + 30 > canvas_height: # bottom edge
                self.y = canvas_height - 30
                self.sl = 0
                self.sr = 0

        # Redraw the bot
        canvas.delete(self.name)
        canvas.delete("detection")
        self.draw(canvas)

    def collectDirt(self, canvas, passiveObjects, count):
        '''
        This function checks if the bot is close to any dirt and collects it.
        :return: The updated list of passive objects and the number of dirt collected.
        '''
        toDelete = [idx for idx, rr in enumerate(passiveObjects) if isinstance(rr, Dirt) and self.distance_to(rr) < 30]
        dirt_collected = len(toDelete)

        for idx in reversed(toDelete):
            canvas.delete(passiveObjects[idx].name)
            del passiveObjects[idx]
            count.itemCollected(canvas)
        return passiveObjects, dirt_collected
        
class Dirt:
    def __init__(self,namep,xx,yy):
        self.centreX = xx
        self.centreY = yy
        self.name = namep

    def draw(self,canvas):
        body = canvas.create_oval(self.centreX-1,self.centreY-1,\
                                  self.centreX+1,self.centreY+1,\
                                  fill="grey",tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY

class Counter:
    def __init__(self):
        self.dirtCollected = 0
        self. totalDirt = 0

    def itemCollected(self,canvas):
        self.dirtCollected += 1
        canvas.delete("dirtCount")
        canvas.create_text(50,50,anchor="w",\
                           text="Dirt collected: "+str(self.dirtCollected),\
                           tags="dirtCount")
    
def initialise(window):
    window.resizable(False,False)
    canvas = tk.Canvas(window,width=1000,height=1000)
    canvas.pack()
    return canvas

def buttonClicked(x,y,agents):
    for rr in agents:
        if isinstance(rr,Bot):
            rr.x = x
            rr.y = y

def createObjects(canvas, mode='rule-based'):
    '''

    :param canvas:
    :param mode:
    :return:
    '''
    agents = []
    passiveObjects = []

    # place line of dirt across top
    i = 0
    for xx in range(0,10):
        for _ in range(50+random.randrange(-10,10)):
            x = xx*100+random.randrange(0,100)
            y = 0+random.randrange(0,100)
            dirt = Dirt("Dirt"+str(i),x,y)
            i += 1
            passiveObjects.append(dirt)
            dirt.draw(canvas)
            
    # place line of dirt down side
    for yy in range(1,10):
        for _ in range(100+random.randrange(-10,10)):
            x = 9*100+random.randrange(0,100)
            y = yy*100+random.randrange(0,100)
            dirt = Dirt("Dirt"+str(i),x,y)
            i += 1
            passiveObjects.append(dirt)
            dirt.draw(canvas)

    # place less dirt everywhere else
    for xx in range(0,9):
        for yy in range(1,10):
            for _ in range(10+random.randrange(-3,3)):
                x = xx*100+random.randrange(0,100)
                y = yy*100+random.randrange(0,100)
                dirt = Dirt("Dirt"+str(i),x,y)
                i += 1
                passiveObjects.append(dirt)
                dirt.draw(canvas)

    count = Counter()

    # place Bot
    bot = Bot("Bot1",passiveObjects,count)
    brain = Brain(bot, mode)
    bot.setBrain(brain)
    agents.append(bot)
    bot.draw(canvas)

    canvas.bind( "<Button-1>", lambda event: buttonClicked(event.x,event.y,agents) )


    return agents, passiveObjects, count

def moveIt(canvas,agents,passiveObjects,count):
    '''
    This function recursively calls itself every 50ms to update the canvas and move the agents.
    '''
    for rr in agents:
        rr.thinkAndAct(agents,passiveObjects)
        rr.update(canvas,passiveObjects,1.0)
        passiveObjects, dirt_collected = rr.collectDirt(canvas,passiveObjects,count)
    canvas.after(50,moveIt,canvas,agents,passiveObjects,count)


def main():
    window = tk.Tk()
    canvas = initialise(window)
    agents, passiveObjects, count = createObjects(canvas)
    moveIt(canvas,agents,passiveObjects,count)
    window.mainloop()


if __name__ == '__main__':
    main()
