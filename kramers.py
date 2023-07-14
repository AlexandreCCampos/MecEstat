import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

rng = np.random.default_rng(seed=0)


def U(T, U0, T1, T2):
    return 16 * U0 * (T - T1)**2 * (T - T2)**2 / (T1 - T2)**4


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
T1 = 280.0
T2 = 290.0
g = 1.0
U0 = g**2 / (2 * 0.12)
periodo = 1e5
omega = 2 * np.pi / periodo
epsilon = 0.001 * 340
dt = 0.01
t_f = 300.0 * 1e3
n = int(t_f // dt)
t = dt * np.arange(n)

C = periodo / np.exp(2 * U0 / g**2)

def const_kramers(U0, T1=T1, T2=T2):
    Tmed = (T1 + T2) / 2
    const_mais = 2*np.pi / np.sqrt( - d2U_dT2(T=Tmed, U0=U0, T1=T1, T2=T2) * d2U_dT2(T=T2, U0=U0, T1=T1, T2=T2) )
    const_menos = 2*np.pi / np.sqrt( - d2U_dT2(T=Tmed, U0=U0, T1=T1, T2=T2) * d2U_dT2(T=T1, U0=U0, T1=T1, T2=T2) )
    return const_mais, const_menos

print(const_kramers(U0))