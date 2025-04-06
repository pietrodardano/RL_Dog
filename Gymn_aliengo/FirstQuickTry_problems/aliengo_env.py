import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np

urdf_path  = '/home/rluser/RL_Dog/src/assets/URDF/aliengo.urdf'
myGPU = "cuda:0"

class AlienGoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AlienGoEnv, self).__init__()

        # Connect to PyBullet
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load the AlienGo URDF
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0.5], useFixedBase=False)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)                   #12 = 4legs * 3motors 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

    def step(self, action):
        # Apply the actions to the robot's motors
        for i in range(12):  # Assuming 12 motors
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=action[i])
        p.stepSimulation()

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        return obs, reward, done, {}

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0.5], useFixedBase=False)
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)

    def _get_observation(self):
        # Collect observations like joint positions, velocities, etc.
        joint_states = p.getJointStates(self.robot_id, range(12))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        return np.array(joint_positions + joint_velocities)

    def _compute_reward(self):
        # Define your reward function
        return 0

    def _check_done(self):
        # Define your termination condition
        return False
