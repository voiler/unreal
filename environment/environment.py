# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Environment(object):
    # cached action size
    action_size = -1

    @staticmethod
    def create_environment(env_name):
        from . import gym_environment
        return gym_environment.GymEnvironment(env_name)

    @staticmethod
    def get_action_size(env_name):
        if Environment.action_size >= 0:
            return Environment.action_size
        from . import gym_environment
        Environment.action_size = \
            gym_environment.GymEnvironment.get_action_size(env_name)
        return Environment.action_size

    def __init__(self):
        pass

    def process(self, action):
        pass

    def reset(self):
        pass

    def stop(self):
        pass

    def _subsample(self, a, average_width):
        s = a.shape
        sh = s[0] // average_width, average_width, s[1] // average_width, average_width
        return a.reshape(sh).mean(-1).mean(1)

    def _calc_pixel_change(self, state, last_state):
        d = np.absolute(state[:, 2:-2, 2:-2] - last_state[:, 2:-2, 2:-2])
        # (80,80,3)
        m = np.mean(d, 0)
        c = self._subsample(m, 4)
        return c
