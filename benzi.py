import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

rng = np.random.default_rng(seed=0)

# Parâmetros
a = 1.0
A = 0.12
Omega = 1e-3
T = 2 * np.pi / Omega
epsilon = 0.25

# t NÃO está em unidades de T (período)
dt = 0.01  # dt tem que ser pequeno para dx também seja pequeno e não divirja
t_f = 10 * T
n = int(t_f // dt)
t = dt * np.arange(n)

x_0 = 1.0
x = np.zeros(
    shape=n
)
x[0] = x_0

print()
print('Parâmetros')
print('a =       ', a)
print('A =       ', A)
print('Omega =   ', Omega)
print('T =       ', T)
print('epsilon = ', epsilon)
print()
print('x_0 =     ', x_0)
print('n =       ', n)
print('dt =      ', dt)
print()

dW = rng.normal(
    loc=0.0,
    scale=np.sqrt(dt),
    size=n,
)  # Ver o arquivo wiener.py

# dx = x(a-x²)dt + epsilon dW
for i in trange(n - 1, desc='Reprodução Referência 1'):
    F_i = x[i] * (a - x[i]**2) + A * np.cos(Omega * t[i])
    x[i + 1] = x[i] + dt * F_i + epsilon * dW[i]


fig, ax = plt.subplots()

ax.set_xlabel('t / T')
ax.set_xlim(0, t_f / T)
# ax.set_xlim(0, 100 * dt)

ax.set_ylabel('x(t)')
ax.set_ylim(-2, 2)

ax.plot(t / T, x)
fig.savefig('benzi.pdf')
