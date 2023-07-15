import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
import pickle
from multiprocessing import Pool

from trabalho import (
    rng,
    dU_dT,
    d2U_dT2,
    periodo,
    omega,
    T1,
    T2,
    U0,
    g,
    epsilon,
    dt,
    t_f,
    n,
    t,
    dW,
)


# array_epsilon = np.linspace(20-10, 20+10, num=5)
# array_g = np.linspace(5.6-2, 5.6+2, num=5)

array_U0 = np.array([0.1, 1, 10])
array_g = np.array([0.1, 10, 20])

# array_U0 = np.array([1, U0])
# array_g = np.array([1, g])

fig, axs = plt.subplots(nrows=array_U0.size, ncols=array_g.size)
# fig.set_size_inches(10, 10)

for j in range(array_U0.size):
    axs[j, 0].set_ylabel("$U_0$ = " + str(array_U0[j]))

for k in range(array_g.size):
    axs[-1, k].set_xlabel("$g$ = " + str(round(array_g[k], 1)))

for j, k in np.ndindex(array_U0.size, array_g.size):
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


def f(args):
    j = args[0]
    k = args[1]
    T = np.zeros(shape=n)
    T[0] = T1

    U0 = array_U0[j]
    g = array_g[k]

    # Prescrição de Itô
    # T(t + dt) = T(t) + dt * ( -U'(T) + g * eta(t) )
    # dT = dt * ( -U'(T)) + g * dW
    for i in trange(n - 1, desc="Subplot " + str(j) + " " + str(k)):
        F_i = -dU_dT(T[i], U0, T1, T2) - epsilon * np.cos(omega * t[i])
        T[i + 1] = T[i] + dt * F_i + g * dW[i]
    
    return T

if __name__ == '__main__':
    with Pool(4) as p:
        u = p.map(
             func=f,
             iterable=np.ndindex(array_U0.size, array_g.size)
         )
        lista_u = list(u)


    for m in range(array_U0.size * array_g.size):
        j, k = list(np.ndindex(array_U0.size, array_g.size))[m]
        T_m = lista_u[m]

        passo = 100000
        axs[j, k].plot(
            (t / 1e3)[::passo],
            T_m[::passo],
        )

    fig.savefig(
        fname="verificacao.pdf",
        dpi=300,
    )

    # plt.show()


# for j, k in np.ndindex(array_U0.size, array_g.size):
#     U0 = array_U0[j]
#     g = array_g[k]

#     # Prescrição de Itô
#     # T(t + dt) = T(t) + dt * ( -U'(T) + g * eta(t) )
#     # dT = dt * ( -U'(T)) + g * dW
#     for i in trange(n - 1, desc="Subplot " + str(j) + " " + str(k)):
#         F_i = -dU_dT(T[i], U0, T1, T2) - epsilon * np.cos(omega * t[i])
#         T[i + 1] = T[i] + dt * F_i + g * dW[i]

#     passo = 100000
#     axs[j, k].plot(
#         (t / 1e3)[::passo],
#         T[::passo],
#     )

#     fig.savefig(
#         fname="verificacao.pdf",
#         dpi=300,
#     )

#     # plt.show()

# pickle.dump((fig, axs), open("verificacao.pickle", "wb"))
