import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(seed=0)

x_0 = 1.0
dt = 0.01
n = 10000

t = np.arange(stop=n) * dt
x = np.zeros(
    shape=n
)
x[0] = x_0



def F(x, a=0.0):
    return x * (a - x**2)


# def G(x, epsilon=1.0):
#     return epsilon


valores_eta = [- dt**(-0.5), dt**(-0.5)]
eta_t = rng.choice(
    a=valores_eta,
    size=n
)

a = 1.0
epsilon = 0.25
for i in range(n - 1):
    t[i + 1] = t[i] + dt
    x[i + 1] = x[i] + dt * (F(x[i], a) + epsilon * eta_t[i])


fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(-2, 2)

ax.plot(t, x)

fig.savefig('teste.png')
