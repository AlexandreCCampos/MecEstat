import numpy as np
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(0)

# Integration parameters
T = 1
N = 500
dt = T / N

# Times to sample at
t = np.linspace(0, T, N)[:, None]

# Sample dW's and compute cumulative sum
dW = dt ** 0.5 * np.random.normal(size=(N - 1, 1))
W = np.concatenate([[0], np.cumsum(dW)])

# # Sample dW's and compute cumulative sum
# dW = dt ** 0.5 * np.random.normal(size=N - 1)
# W = np.cumsum(dW)

plt.plot(W)
plt.show()