import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(seed=2)


def dU_dT(T, U0, T1, T2):
    return 32 * U0 * (T - T1) * (T - T2) * (2 * T - T1 - T2) / (T1 - T2)**4


# Parâmetros
U0 = 1.0
T1 = 0.0
T2 = 10.0
g = 5.0
periodo = 1e0
omega = 2 * np.pi / periodo
epsilon = 100.0

n = 100001

t, dt = np.linspace(
    start=0.0,
    stop=1e1,
    num=n,
    retstep=True,
)

T_0 = T2
T = np.zeros(
    shape=n
)
T[0] = 2.0

# eta = rng.choice(
#     a=np.array([- 1, 1]) / np.sqrt(dt),
#     size=n
# )
eta = rng.normal(
    loc=0.0,
    scale=1.0 / np.sqrt(dt),
    size=n
)

# Prescrição de Itô
# T(t + dt) = T(t) + dt * ( -U'(T) + g * eta(t) )
for i in range(n - 1):
    F_i = - dU_dT(T[i], U0, T1, T2) - epsilon * np.cos(omega * t[i])
    T[i + 1] = T[i] + dt * (F_i + g * eta[i])


fig, ax = plt.subplots()

ax.set_xlabel('t')
# ax.set_xlim(100, 500)
ax.set_xlim(t.min(), t.max())

ax.set_ylabel('T(t)')
ax.set_ylim(-5, 15)

ax.plot(t, T)
fig.savefig('t.pdf')
# plt.show()
