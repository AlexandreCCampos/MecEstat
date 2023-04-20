import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import norm  # gaussian plot

rng = np.random.default_rng(seed=100)

# Parâmetros
a = 1.0
A = 0.12
Omega = 1e-3
T = 2 * np.pi / Omega
epsilon = 0.25

n = 1001

# t está em unidades de T (período)
t, dt = np.linspace(
    start=0.0,
    stop=10,
    num=n,
    retstep=True,
)

x_0 = 1.0
x = np.zeros(
    shape=n
)
x[0] = x_0

# eta = rng.choice(
#     a=np.array([- 1, 1]) / np.sqrt(dt),
#     size=n
# )
# eta = rng.normal(0, 1./np.sqrt(dt), n)

dW = rng.normal(
    loc=0.0,
    scale=1.0 / np.sqrt(dt),
    size=n
)

# Prescrição de Itô
# x(t + dt) = x(t) + dt * ( F(x(t)) + G(x(t)) * eta(t) )
# dx = x(a-x²)dt + epsilon dW
for i in range(n - 1):
    F_i = x[i] * (a - x[i]**2) + A * np.cos(2 * np.pi * t[i])
    x[i + 1] = x[i] + dt * F_i + epsilon * dW[i]


print()
print('Parâmetros')
print('a =       ', a)
print('A =       ', A)
print('Omega =   ', Omega)
print('T =       ', T)
print('epsilon = ', epsilon)
print()
print('Condições iniciais')
print('x_0 =', x_0)
print('n =  ', n)
print('dt = ', dt)
print()

fig, ax = plt.subplots()

ax.set_xlabel('t / T')
ax.set_xlim(0, 10)

ax.set_ylabel('x(t)')
ax.set_ylim(-2, 2)

ax.plot(t, x)
fig.savefig('teste.png')


#####################################################################

# fig.savefig('teste_sorteio_gaussiano.png')


# # plotting gaussian to show the rnd gen is in fact gaussian
# # Plotting the histogram.
# fig2, ax2 = plt.subplots()
# ax2.hist(eta_t, bins=25, density=True, alpha=0.6, color='b')

# mu, std = norm.fit(eta_t)
# xmin, xmax = ax2.get_xlim()
# # x = np.linspace(xmin, xmax, 1)
# x = np.arange(xmin, xmax, 0.1)
# # p = norm.pdf(x, 0,  1./np.sqrt(epsilon))
# print(1./np.sqrt(epsilon))
# p = norm.pdf(x, mu, std)

# ax2.plot(x, p, 'k', linewidth=2)
# title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
# ax2.set_title(title)
# fig2.savefig('gaussiana.png')

# # ------
