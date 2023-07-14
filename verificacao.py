import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
import pickle

rng = np.random.default_rng(seed=0)


def dU_dT(T, U0, T1, T2):
    return 32 * U0 * (T - T1) * (T - T2) * (2 * T - T1 - T2) / (T1 - T2) ** 4


def d2U_dT2(T, U0, T1, T2):
    return (
        32
        * (6 * T**2 - 6 * T * T1 + T1**2 - 6 * T * T2 + 4 * T1 * T2 + T2**2)
        * U0
        / (T1 - T2) ** 4
    )


# Parâmetros
periodo = 1e5
omega = 2 * np.pi / periodo
T1 = 280.0
T2 = 290.0
U0 = 213

dt = 0.01
t_f = 300.0 * 1e3
n = int(t_f // dt)
t = dt * np.arange(n)

# array_epsilon = np.linspace(20-10, 20+10, num=5)
# array_g = np.linspace(5.6-2, 5.6+2, num=5)

array_epsilon = np.array([0.1, 1, 10, 100, 1000])
array_g = np.array([0.1, 1, 10, 100, 1000])

T = np.zeros(shape=n)
T[0] = T1

dW = rng.normal(
        loc=0.0,
        scale=np.sqrt(dt),
        size=n,
    )  # Ver o arquivo wiener.py

fig, axs = plt.subplots(
    nrows=array_epsilon.size,
    ncols=array_g.size
)
# fig.set_size_inches(10, 10)

for j in range(array_epsilon.size):
    axs[j, 0].set_ylabel("$\epsilon$ = " + str(array_epsilon[j]))

for k in range(array_g.size):
    axs[-1, k].set_xlabel("$g$ = " + str(round(array_g[k], 1)))

for j, k in np.ndindex(array_epsilon.size, array_g.size):
    # axs[j, k].grid(visible=True)

    # axs[j, k].set_xlabel("Anos ($\\times 10^3$)")
    axs[j, k].set_xlim(0, t_f / 1e3)

    # axs[j, k].set_ylabel("Temperatura (K)")
    axs[j, k].set_ylim(T1 - 10, T2 + 10)

    axs[j, k].set_yticklabels([])
    axs[j, k].set_xticklabels([])
    axs[j, k].set_xticks([])
    axs[j, k].set_yticks([])

fig.savefig(
    fname="verificacao.pdf",
    dpi=300,
)

for j, k in np.ndindex(array_epsilon.size, array_g.size):
    epsilon = array_epsilon[j]
    g = array_g[k]

    # Prescrição de Itô
    # T(t + dt) = T(t) + dt * ( -U'(T) + g * eta(t) )
    # dT = dt * ( -U'(T)) + g * dW
    for i in trange(n - 1, desc="Subplot " + str(j) + ' ' + str(k)):
        F_i = -dU_dT(T[i], U0, T1, T2) - epsilon * np.cos(omega * t[i])
        T[i + 1] = T[i] + dt * F_i + g * dW[i]

    passo = 100000
    axs[j, k].plot(
        (t / 1e3)[::passo],
        T[::passo],
    )

    fig.savefig(
        fname="verificacao.pdf",
        dpi=300,
    )

    # plt.show()

# pickle.dump((fig, axs), open('verificacao.pickle', 'wb'))