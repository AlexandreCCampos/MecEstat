import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
import pickle
import multiprocessing
from functools import partial

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

# array_U0 = np.array([0.1, 1, 10, 100, 1000])
# array_g = np.array([0.1, 10, 20, 30, 40])

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
    # axs[j, k].set_xlim(0, t_f / 1e3)

    # axs[j, k].set_ylabel("Temperatura (K)")
    # axs[j, k].set_ylim(T1 - 10, T2 + 10)

    axs[j, k].set_yticklabels([])
    axs[j, k].set_xticklabels([])
    axs[j, k].set_xticks([])
    axs[j, k].set_yticks([])

fig.savefig(
    fname="verificacao.pdf",
    dpi=300,
)


def f(args, axs2):
    j = args[0]
    k = args[1]
    U0 = array_U0[j]
    g = array_g[k]
    T = np.zeros(shape=n)
    T[0] = T1
    # Prescrição de Itô
    # T(t + dt) = T(t) + dt * ( -U'(T) + g * eta(t) )
    # dT = dt * ( -U'(T)) + g * dW

    x = np.linspace(0, 10, 100)
    y = np.sin(x)*10

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.plot(x, y)

    # fig2.savefig("test_exclude_later.png")
    axs2[j, k].plot(x, y)

    # for i in trange(n - 1, desc="Subplot " + str(j) + " " + str(k)):
    #     F_i = -dU_dT(T[i], U0, T1, T2) - epsilon * np.cos(omega * t[i])
    #     T[i + 1] = T[i] + dt * F_i + g * dW[i]

    # passo = 100000
    # axs[j, k].plot(
    #     (t / 1e3)[::passo],
    #     T[::passo],
    # )
    # return (j, k, fig2)

# # Create a manager to handle shared objects
# manager = multiprocessing.Manager()

# # Create a shared list using the manager
# shared_list = manager.list()


# Define the number of processes to use
num_processes = multiprocessing.cpu_count()

# Create a pool of processes
pool = multiprocessing.Pool(processes=num_processes)

# Create a partial function with shared_list as a fixed argument
# process_item_partial = partial(f, axs2=axs)

# pool.map(process_item_partial, np.ndindex(array_U0.size, array_g.size),axs)
# graphlist = list(pool.map(f, np.ndindex(array_U0.size, array_g.size)))

list_axs = []
for i in np.ndindex(array_U0.size, array_g.size):
    list_axs.append(axs[i[0], i[1]])

pool.map(f, np.ndindex(array_U0.size, array_g.size), list(list_axs))


# for i in graphlist:

#     print(i[0])
#     print(i[1])
#     print(i[2])
#     fig2, ax2 = plt.subplots()
#     ax2.add_subplot(i[2])

#     fig2.savefig("test_exclude_later.png")
#     axs[i[0], i[1]].add_subplot(i[2])

fig.savefig(
    fname="verificacao.pdf",
    dpi=300,
)

# Close the pool
pool.close()
pool.join()

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
#         T[::passo],qui
#     )

#     fig.savefig(
#         fname="verificacao.pdf",
#         dpi=300,
#     )

#     # plt.show()

# pickle.dump((fig, axs), open("verificacao.pickle", "wb"))
