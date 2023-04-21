import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

rng = np.random.default_rng(seed=0)


def dU_dT(T, U0, T1, T2):
    return 32 * U0 * (T - T1) * (T - T2) * (2 * T - T1 - T2) / (T1 - T2)**4


# Parâmetros
U0 = 1.0
T1 = 280.0
T2 = 290.0
g = 0.25
periodo = 1e5
omega = 2 * np.pi / periodo
epsilon = 0.0005

dt = 0.01
t_f = 300.0 * 1e3
n = int(t_f // dt)
t = dt * np.arange(n)

T_0 = T2
T = np.zeros(
    shape=n
)
T[0] = T2

# eta = rng.choice(
#     a=np.array([- 1, 1]) / np.sqrt(dt),
#     size=n
# )
# eta = rng.normal(
#     loc=0.0,
#     # scale=1.0 / np.sqrt(dt),
#     scale=1.0,
#     size=n
# )

dW = rng.normal(
    loc=0.0,
    scale=np.sqrt(dt),
    size=n,
)  # Ver o arquivo wiener.py

# Prescrição de Itô
# T(t + dt) = T(t) + dt * ( -U'(T) + g * eta(t) )
# dT = dt * ( -U'(T)) + g * dW
for i in trange(n - 1, desc='Trabalho'):
    F_i = - dU_dT(T[i], U0, T1, T2) - epsilon * np.cos(omega * t[i])
    T[i + 1] = T[i] + dt * F_i + g * dW[i]


fig, ax = plt.subplots()

ax.set_xlabel('t')
ax.set_xlim(0, t_f)

ax.set_ylabel('T')
ax.set_ylim(T1 - 10, T2 + 10)

ax.plot(t, T)
# fig.savefig('t.pdf')
plt.show()
