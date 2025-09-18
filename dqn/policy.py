import numpy as np
import tensorflow as tf


class EpsilonScheduler:
    def __init__(self, eps_start, eps_end, decay_steps):
        self.s, self.e, self.N = float(eps_start), float(eps_end), int(decay_steps)
    def at(self, t):
        frac = min(1.0, t/max(1,self.N))    
        return self.s + frac * (self.e - self.s)
    

def select_action(q_net, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, 4)   # random action
        q = q_net(state[None, :])  # add batch dim
        return int(tf.argmax(q, axis=1).numpy()[0])  # greedy action
