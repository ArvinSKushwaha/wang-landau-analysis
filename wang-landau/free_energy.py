from functools import partial
from glob import glob
from typing import Any, Callable

import jax
import matplotlib as mpl
import polars as pl

from jax import numpy as np
from jaxtyping import Array, Float
from matplotlib import pyplot as plt

mpl.use('module://matplotlib-backend-sixel')

jax.config.update('jax_enable_x64', True)

beta = np.linspace(0.0, 1.0, 1000, dtype=np.float64)
cv_t = np.linspace(2.0, 2.7, 1000, dtype=np.float64)

L = [8, 16, 32]
U = {}
F = {}
C_V = {}


def d_dT(func_of_beta: Callable[[Float[Array, ''], Any, Any], Any]):
    return lambda beta, *args: -(beta**2) * jax.grad(func_of_beta)(beta, *args)


@jax.jit
def partition_function_(beta: Float[Array, ''], log_dos: Float[Array, 'n'], energies: Float[Array, 'n']):
    return np.sum(np.exp(log_dos - energies * beta))


@jax.jit
def internal_energy_(beta: Float[Array, ''], log_dos: Float[Array, 'n'], energies: Float[Array, 'n']):
    return (
        -jax.grad(lambda beta: np.log(partition_function_(beta, log_dos, energies - energies.min())))(beta)
        + energies.min()
    )


@jax.jit
def free_energy_beta_(beta: Float[Array, ''], log_dos: Float[Array, 'n'], energies: Float[Array, 'n']):
    return -np.log(partition_function_(beta, log_dos, energies - energies.min())) + energies.min() * beta


@jax.jit
def heat_capacity_(beta: Float[Array, ''], log_dos: Float[Array, 'n'], energies: Float[Array, 'n']):
    return d_dT(internal_energy_)(beta, log_dos - log_dos.min(), energies)


for x in L:
    files = [
        i
        for _, i in sorted(
            [(float(f[14 + len(str(x)) : -4]), f) for f in glob(f'./ising-{x}/data-*.csv')], reverse=True
        )
    ]
    f = files[-1]

    df = pl.read_csv(f)
    df = df.filter(pl.col('entropy') > 0)

    energies = np.asarray(df['energies'], dtype=np.float64)
    log_dos = np.asarray(df['entropy'], dtype=np.float64)

    log_dos = log_dos - log_dos.mean()

    internal_energy = partial(internal_energy_, log_dos=log_dos, energies=energies)
    free_energy_beta = partial(free_energy_beta_, log_dos=log_dos, energies=energies)
    heat_capacity = partial(heat_capacity_, log_dos=log_dos, energies=energies)

    U[x] = jax.lax.map(internal_energy, beta, batch_size=64) / x**2
    F[x] = jax.lax.map(free_energy_beta, beta, batch_size=64) / x**2
    C_V[x] = jax.lax.map(heat_capacity, 1 / cv_t, batch_size=64) / x**2

for x in L:
    plt.plot(beta, U[x], label=f'$\\langle E \\rangle_{{{x}}}$')

plt.legend()
plt.xlabel('$\\beta$')
plt.ylabel('$U / N^2$')
plt.title('Internal Energy for Periodic Ising Model')
plt.grid(True)
plt.savefig('../report/U_Ising.png')
plt.show()

for x in L:
    plt.plot(beta, F[x], label=f'$\\beta F_{{{x}}}$')

plt.legend()
plt.xlabel('$\\beta$')
plt.ylabel('$\\beta F / N^2$')
plt.title('Free Energy for Periodic Ising Model')
plt.grid(True)
plt.savefig('../report/F_Ising.png')
plt.show()

for x in L:
    plt.plot(1 / cv_t, C_V[x], label=f'$(C_V)_{{{x}}}$')

plt.axvline(0.5 * np.log(1 + 2**0.5), linestyle=':', label='$\\beta_{ONS}$')  # type: ignore

plt.legend()
plt.xlabel('$\\beta$')
plt.ylabel('$C_V / N^2$')
plt.title('Heat Capacity for Periodic Ising Model')
plt.grid(True)
plt.savefig('../report/C_V_Ising.png')
plt.show()
