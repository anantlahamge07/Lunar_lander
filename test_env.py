from dqn.networks import QNetwork, hard_update
from dqn.train_step import dqn_update
import tensorflow as tf, numpy as np

# build nets
online = QNetwork(); target = QNetwork()
_ = online(tf.random.normal((1,8))); _ = target(tf.random.normal((1,8)))
hard_update(target, online)

opt = tf.keras.optimizers.Adam(1e-3)

# fake minibatch
B = 32
s  = np.random.randn(B,8).astype('float32')
a  = np.random.randint(0,4,size=B).astype('int32')
r  = np.random.randn(B).astype('float32')
s2 = np.random.randn(B,8).astype('float32')
d  = (np.random.rand(B) < 0.1).astype('float32')

# Normal DQN
loss1 = dqn_update(online, target, opt, (s,a,r,s2,d), gamma=0.99, double=False)
print("Normal DQN loss:", float(loss1))

