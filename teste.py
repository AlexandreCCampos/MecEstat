import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(seed=1)

x_0 = 1.0
dt = 0.01
n = 10000
epsilon = 0.25

t = np.arange(stop=n) * dt
x = np.zeros(
    shape=n
)
x[0] = x_0



def F(x, t, a=1.0, A=0.12, Omega=1e-3):
    return x * (a - x**2) + A * np.cos(Omega * t)


# def G(x, epsilon=1.0):
#     return epsilon


# valores_eta = [- dt**(-0.5), dt**(-0.5)]
# eta_t = rng.choice(
#     a=valores_eta,
#     size=n
# )
eta_t = np.random.normal(0, 1./np.sqrt(epsilon), 1000)


a = 1.0
epsilon = 0.25
A = 0.12
for i in range(n - 1):
    forca = F(x[i], t[i])
    t[i + 1] = t[i] + dt
    x[i + 1] = x[i] + dt * forca + epsilon * eta_t[i]


fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(-2, 2)

ax.plot(t, x)

fig.savefig('teste_sorteio_gaussiano.png')

# blah