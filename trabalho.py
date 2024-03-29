import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

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
C = 2 * np.pi * (T1 - T2) ** 2 / (np.sqrt(16 * 32) * U0)
g = np.sqrt(2 * U0 / np.log(periodo / C))
epsilon = 20

print("C = ", C)
print("g = ", g)

dt = 0.01
t_f = 300.0 * 1e3
n = int(t_f // dt)
t = dt * np.arange(n)

T = np.zeros(shape=n)
T[0] = T1

dW = rng.normal(
    loc=0.0,
    scale=np.sqrt(dt),
    size=n,
)  # Ver o arquivo wiener.py

if __name__== '__main__':
    # Prescrição de Itô
    # T(t + dt) = T(t) + dt * ( -U'(T) + g * eta(t) )
    # dT = dt * ( -U'(T)) + g * dW
    for i in trange(n - 1, desc="Trabalho"):
        F_i = -dU_dT(T[i], U0, T1, T2) - epsilon * np.cos(omega * t[i])
        T[i + 1] = T[i] + dt * F_i + g * dW[i]


    ############## Gráfico

    fig, ax = plt.subplots()

    # fig.set_size_inches(10, 10)

    ax.grid(visible=True)

    ax.set_xlabel("Anos ($\\times 10^3$)")
    ax.set_xlim(0, t_f / 1e3)

    ax.set_ylabel("Temperatura (K)")
    ax.set_ylim(T1 - 10, T2 + 10)

    # ax.plot(
    #     t / 1e3,
    #     T,
    #     linewidth=0.1,
    # )

    passo = 100000
    ax.plot(
        (t / 1e3)[::passo],
        T[::passo],
    )

    # ax.plot(
    #     (t / 1e3),
    #     T,
    # )

    fig.savefig(
        fname="trabalho.pdf",
        dpi=300,
    )

    # plt.show()
