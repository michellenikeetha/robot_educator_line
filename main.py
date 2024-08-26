# !/usr/bin/env pybricks-micropython
import random
from pybricks.ev3devices import Motor, ColorSensor
from pybricks.parameters import Port, Color
from pybricks.tools import wait
from pybricks.robotics import DriveBase


# Q-table initialized to zeros
q_table = {
    'Left': [0, 0, 0],   # Actions: [Move Left, Move Right, Move Forward]
    'Center': [0, 0, 0],
    'Right': [0, 0, 0],
    'Off Track': [0, 0, 0]
}

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate


# Initialize the motors.
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

# Initialize the color sensor.
color_sensor = ColorSensor(Port.S3)

# Initialize the drive base.
robot = DriveBase(left_motor, right_motor, wheel_diameter=55.5, axle_track=104)


BLACK = 9
WHITE = 85
threshold = (BLACK + WHITE) / 2

# Set the drive speed at 100 millimeters per second.
DRIVE_SPEED = 100

# Set the gain of the proportional line controller. This means that for every
# percentage point of light deviating from the threshold, we set the turn
# rate of the drivebase to 1.2 degrees per second.

# For example, if the light value deviates from the threshold by 10, the robot
# steers at 10*1.2 = 12 degrees per second.
PROPORTIONAL_GAIN = 1.8

# Robot control functions (simplified)
def move_left():
    # Code to move the robot left
    # Right motor moves forward, left motor moves backward (or slower)
    right_motor.run(200)  # Adjust speed as needed
    left_motor.run(-200)  # Adjust speed as needed
    wait(100)  # Adjust timing to control turn sharpness
    right_motor.stop()
    left_motor.stop()

def move_right():
    # Code to move the robot right
    # Left motor moves forward, right motor moves backward (or slower)
    left_motor.run(200)  # Adjust speed as needed
    right_motor.run(-200)  # Adjust speed as needed
    wait(100)  # Adjust timing to control turn sharpness
    left_motor.stop()
    right_motor.stop()

def move_forward():
    # Code to move the robot forward
    # Both motors move forward at the same speed
    left_motor.run(200)  # Adjust speed as needed
    right_motor.run(200)  # Adjust speed as needed
    wait(100)  # Adjust timing to control the duration of the movement
    left_motor.stop()
    right_motor.stop()


actions = [move_left, move_right, move_forward]

# Reward function
def get_reward(state):
    if state == 'Center':
        return 10
    elif state == 'Left' or state == 'Right':
        return 5
    else:
        return -10

# Main loop
for episode in range(20):  # Example number of episodes
    state = 'Center'  # Start state, typically from sensor reading
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random action
            action_index = random.randint(0, 2)
        else:
            # Exploitation: Choose the best action based on Q-values
            action_index = q_table[state].index(max(q_table[state]))

        # Take the action
        actions[action_index]()

        # Get new state
        if color_sensor.color() == Color.WHITE:
            new_state = 'Center'  # Adjust as per your sensor's output
        elif color_sensor.color() == Color.BLACK:
            new_state = 'Off Track'
        else:
            new_state = 'Off Track'

        # Get new state and reward
        # new_state = 'Center'  # Update based on sensor readings
        reward = get_reward(new_state)

        # Q-learning update
        q_table[state][action_index] += alpha * (
            reward + gamma * max(q_table[new_state]) - q_table[state][action_index]
        )

        # Update state
        state = new_state

        # Check if done (e.g., robot is off track or reached the goal)
        if state == 'Off Track':
            done = True