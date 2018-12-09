from typing import Dict, List, Tuple
from enum import Enum
from random import random
import numpy as np
import tensorflow as tf


def normalize(num_list: np.array) -> np.array:
    s = sum(num_list)
    return num_list / s


def NStepTransition(s, a, r, ss, gamma):
    total_reward = 0
    pre = 1
    for reward in r:
        total_reward += pre * reward
        pre *= gamma
    return np.hstack(s, a, total_reward, ss)

def take_vector_elems(vectors, indices):
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))

class _PrioritizationPhase(Enum):
    WARM_UP = 1
    SEQUENTIAL = 2
    RANDOMIZATION = 3
    FOCUSED_TRAINING = 4


class PrioritizedStateList:
    def __init__(self, game, states):
        self.game = game
        self.states = states
        self.ep_rewards = {i: [] for i in states}
        self.phase = _PrioritizationPhase.WARM_UP
        self

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the highest priority state. Priority is based one what phase the training is in
        WARM_UP: Get addequate data for all states
        """
        pass

    def record_reward(self, state, reward):
        self.ep_reward[state].append(reward)
