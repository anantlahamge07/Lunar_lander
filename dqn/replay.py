import numpy as np

class ReplayBuffer:
    """Fixed-size circular buffer for (s, a, r, s', done)."""
    def __init__(self, obs_dim: int, size: int):
        self.size = int(size)
        self.obs  = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.act  = np.zeros((self.size,), dtype=np.int32)
        self.rew  = np.zeros((self.size,), dtype=np.float32)
        self.done = np.zeros((self.size,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.size if self.full else self.ptr

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.obs[i]  = s
        self.act[i]  = a
        self.rew[i]  = r
        self.next[i] = s2
        self.done[i] = float(d)
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0: self.full = True

    def sample(self, batch_size: int):
        n = len(self)
        if n == 0:
            raise ValueError("ReplayBuffer is empty.")
        idx = np.random.randint(0, n, size=batch_size)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next[idx], self.done[idx])