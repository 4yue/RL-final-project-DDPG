import numpy as np


class Memory(object):
    def __init__(self, size, dims):
        """
        :param size:  the capacity of memory
        :param dims:  the dimension of a tuple
        """
        self.size = size
        self.dims = dims
        self.data = np.zeros((size, dims))
        self.index = 0

    def store(self, state, action, reward, next_state):
        """
        store the information (state, action, reward, next_state) into memory
        """
        i = self.index % self.size
        self.data[i, :] = np.hstack((state, action, reward, next_state))
        self.index += 1

    def sample(self, n):
        """
        sample from memory, return a tuple
        """
        assert self.index >= self.size, 'Memory has not been fulfilled'
        indices = np.random.choice(self.size, size=n)
        return self.data[indices, :]
