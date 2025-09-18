import tensorflow as tf
from tensorflow.keras import layers, Model

class QNetwork(Model):
    def __init__(self, obs_sim = 8, n_actions = 4):
          
          super().__init__()
          self.d1 = layers.Dense(256, activation = 'relu', input_shape = (obs_sim,))
          self.d2 = layers.Dense(256, activation = 'relu')
          self.out = layers.Dense(n_actions, activation = None)

    def call(self, x):
          x = tf.convert_to_tensor(x, dtype = tf.float32)
          z = self.d1(x)
          z = self.d2(z)
          return self.out(z)
def hard_update(target: tf.keras.Model, online: tf.keras.Model):
          target.set_weights(online.get_weights())

def soft_update(target: tf.keras.Model, online: tf.keras.Model, tau: float = 0.005):
          tw = target.get_weights()
          ow = online.get_weights()
          target.set_weights([(1 - tau) * t + tau * o for t, o in zip(tw, ow)])