import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm #gaussian plot

rng = np.random.default_rng(seed=1)

x_0 = 1.0
dt = 0.01
n = 10000
epsilon = 0.25
a = 1.0
A = 0.12

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
eta_t = rng.normal(0, 1./np.sqrt(epsilon), n)

for i in range(n - 1):
    forca = F(x[i], t[i])
    t[i + 1] = t[i] + dt
    x[i + 1] = x[i] + dt * forca + epsilon * eta_t[i]


fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(-5, 5)

ax.plot(t, x)

fig.savefig('teste_sorteio_gaussiano.png')


#plotting gaussian to show the rnd gen is in fact gaussian
# Plotting the histogram.
fig2, ax2 = plt.subplots()
ax2.hist(eta_t, bins=25, density=True, alpha=0.6, color='b')

mu, std = norm.fit(eta_t) 
xmin, xmax = ax2.get_xlim()
# x = np.linspace(xmin, xmax, 1)
x = np.arange(xmin, xmax, 0.1)
# p = norm.pdf(x, 0,  1./np.sqrt(epsilon))
print(1./np.sqrt(epsilon))
p = norm.pdf(x, mu, std)

ax2.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
ax2.set_title(title)
fig2.savefig('gaussiana.png')

# ------
