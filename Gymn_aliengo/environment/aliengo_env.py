
#BASICS
import numpy as np

#GYMNASIUM
import gymnasium as gym
from gymnasium import spaces

#SIMULATION
import pybullet as p
import pybullet_data

URDF_PATH  = "/home/rluser/RL_Dog/assets/URDF/aliengo.urdf"
MOTORS_NUM = 4*3
SPACE_DIM  = 32 

class AlienGOEnv(gym.Env):
    def __init__(self):
        super(AlienGOEnv, self).__init__()
        
        # Define action space 
        # Assuming AlienGO has 12 joints, each with a continuous action space [-1, 1]    TO CHECK !!!
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(MOTORS_NUM,), dtype=np.float32)     # CHECK if np.inf is correct !!
        
        # Define the observation space: positions, velocities, etc.
        # This is an example, modify based on your robot's state space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(SPACE_DIM,), dtype=np.float32)
        
        # Initialize simulation
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(URDF_PATH, [0, 0, 0.5], useFixedBase=False)
    
    def reset(self):
        # Reset the robot and environment to initial state
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(URDF_PATH, [0, 0, 0.5], useFixedBase=False)
        
        # Get initial observation
        observation = self._get_observation()
        return observation
    
    def step(self, action):
        # Apply action to the robot
        for i in range(12):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, action[i])
        
        # Step simulation
        p.stepSimulation()
        
        # Get observation, reward, done, and info
        observation = self._get_observation()
        reward = self._compute_reward(observation, action)
        done = self._is_done(observation)
        info = {}
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        pass  # Optional: implement rendering
    
    def close(self):
        p.disconnect()
    
    def _get_observation(self):
        # Example observation: joint positions and velocities
        observation = []
        for i in range(MOTORS_NUM):
            joint_info = p.getJointState(self.robot, i)
            observation.extend(joint_info[0:2])  # Position and velocity
        return np.array(observation, dtype=np.float32)
    
    def _compute_reward(self, observation, action):
        # Example reward function: stay upright and move forward
        # Customize this based on your task
        base_position = p.getBasePositionAndOrientation(self.robot)[0]
        reward = base_position[0]  # Reward for moving forward
        return reward
    
    def _is_done(self, observation):
        # Example done condition: robot falls over
        base_position = p.getBasePositionAndOrientation(self.robot)[0]
        if base_position[2] < 0.2:  # If height is too low, robot has fallen
            return True
        return False

# Register the environment
gym.register(
    id='AlienGO-v0',
    entry_point='your_module_name:AlienGOEnv',  # Replace with your module name
)

# Use the environment
env = gym.make('AlienGO-v0')
