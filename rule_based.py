import random


def rule_based_logic(brain, x, y, sl, sr, count):
    if brain.currentlyTurning:
        speed_left, speed_right = -2.0, 2.0
        brain.turningCount -= 1
        if brain.turningCount <= 0:
            brain.movingCount = random.randrange(50, 100)
            brain.currentlyTurning = False
    else:
        speed_left, speed_right = 5.0, 5.0
        brain.movingCount -= 1
        if brain.movingCount <= 0:
            brain.turningCount = random.randrange(20, 40)
            brain.currentlyTurning = True

    if x + speed_left < 0 or x + speed_left > 1000:
        speed_left, speed_right = -speed_left, -speed_right
    if y + speed_right < 0 or y + speed_right > 1000:
        speed_left, speed_right = -speed_left, -speed_right

    return speed_left, speed_right, None, None