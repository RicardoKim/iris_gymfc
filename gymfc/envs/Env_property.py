from .gazebo_env import GazeboEnv
import numpy as np

class Env_property(GazeboEnv):
    def __init(self):
        self.omega_target = self.make_target()
    def reward(self):
        """ Compute the reward """
        return -np.clip(np.sum(np.abs(self.error))/(self.omega_bounds[1]*3), 0, 1)


    def make_target(self):
        """ Sample a random angular velocity """
        return  self.np_random.uniform(self.omega_bounds[0], self.omega_bounds[1], size=3)
    