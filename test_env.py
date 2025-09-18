from dqn.policy import EpsilonScheduler, select_action
from dqn.networks import QNetwork
import numpy as np

sched = EpsilonScheduler(1.0, 0.05, 100_000)
print(sched.at(0), sched.at(50_000), sched.at(200_000))  # 1.0 ~0.525 0.05

net = QNetwork(); _ = net(np.random.randn(1,8).astype('float32'))  # build

state = np.zeros(8, dtype='float32')
for eps in [1.0, 0.5, 0.0]:
    actions = [select_action(net, state, epsilon=eps) for _ in range(5)]
    print(f"ε={eps}: {actions}")