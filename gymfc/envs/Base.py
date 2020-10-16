import math
import numpy as np
from .gazebo_env import GazeboEnv
import logging
logger = logging.getLogger("gymfc")


class Base(GazeboEnv):
    def __init__(self, world="attitude-iris.world", 
                 omega_bounds = [-math.pi, math.pi], 
                 motor_count = 4, 
                 memory_size=1,): 
        
        self.omega_bounds = omega_bounds
        self.max_sim_time = max_sim_time
        self.memory_size = memory_size
        self.motor_count = motor_count
        self.observation_history = []

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        # Step the sim
        self.obs = self.step_sim(action)
        self.error = self.omega_target - self.obs.angular_velocity_rpy
        self.observation_history.append(np.concatenate([self.error]))
        state = self.state()
        done = self.sim_time >= self.max_sim_time
        reward = self.compute_reward()
        info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}

        return state, reward, done, info