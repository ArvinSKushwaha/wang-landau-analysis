from dataclasses import dataclass, field

import jax
import numpy
import polars as pl

from jax import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Scalar, UInt
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use

mpl_use('module://matplotlib-backend-sixel')


@dataclass
class IsingSystem:
    n: int

    j: float
    h: float

    def random_state(self, key: PRNGKeyArray) -> 'IsingState':
        spins = jax.random.bernoulli(key, shape=(self.n, self.n))

        return IsingState(
            spins=spins,
            energy=self.compute_energy(IsingState(spins)),
        )

    def compute_energy(self, state: 'IsingState') -> Float[Scalar, '']:
        z = state.spins * 2.0 - 1.0
        return (
            np.sum(self.h * z)
            + np.sum(self.j * z * np.roll(z, (1, 0), (0, 1)))
            + np.sum(self.j * z * np.roll(z, (0, 1), (0, 1)))
        )


@dataclass
class IsingState:
    spins: Bool[Array, '{n} {n}']
    energy: Float[Scalar, ''] = field(default_factory=lambda: np.array(0.0))

    def flip(self, system: IsingSystem, key: PRNGKeyArray) -> 'IsingState':
        x, y = jax.random.randint(key, shape=(2,), minval=0, maxval=system.n)
        spins = self.spins.at[x, y].set(~self.spins[x, y])
        cluster = np.roll(spins, (-x + 1, -y + 1), (0, 1))[:3, :3] * 2.0 - 1.0
        self_energy_change = 2.0 * (spins[x, y] * 2.0 - 1.0) * system.h
        coupling_energy_change = (
            2.0 * (cluster[0, 1] + cluster[1, 0] + cluster[2, 1] + cluster[1, 2]) * system.j * cluster[1, 1]
        )

        return IsingState(spins, self.energy + self_energy_change + coupling_energy_change)


jax.tree_util.register_dataclass(
    IsingSystem,
    data_fields=[],
    meta_fields=['n', 'j', 'h'],
)

jax.tree_util.register_dataclass(
    IsingState,
    data_fields=['spins', 'energy'],
    meta_fields=[],
)


@dataclass
class WangLandauIsing:
    min_energy: float
    max_energy: float
    bins: int

    system: IsingSystem
    f: float
    energies: Float[Array, 'n']
    histogram: UInt[Array, 'n']
    entropy: Float[Array, 'n']

    @classmethod
    def init_state(cls, system: IsingSystem, min_energy: float, max_energy: float, bins: int) -> 'WangLandauIsing':
        system = system
        f = 1.0
        energies = np.linspace(
            min_energy,
            max_energy,
            bins,
            endpoint=False,
        )
        histogram = np.zeros(bins, dtype=np.uint32)
        entropy = np.zeros(bins)

        return cls(min_energy, max_energy, bins, system, f, energies, histogram, entropy)

    def flatness(self) -> Float[Scalar, '']:
        mask = self.histogram > 0

        return jax.lax.cond(
            mask.sum() > 2,
            lambda: np.nan_to_num(
                (
                    np.max(np.float32(self.histogram), where=mask, initial=0.0)
                    - np.min(np.float32(self.histogram), where=mask, initial=np.inf)
                )
                / np.nan_to_num(np.mean(self.histogram, where=mask), nan=1.0),
                nan=np.inf,
                posinf=np.inf,
                neginf=np.inf,
            ),
            lambda: np.inf,
        )

    def energy_idx(self, energy: Float[Scalar, '']) -> int:
        float_idx = self.bins * (energy - self.min_energy) / (self.max_energy - self.min_energy)

        return np.uint32(np.floor(float_idx.clip(0.0, self.bins - 1)))

    def increment_energy(self, energy: Float[Scalar, ''], f: float):
        idx = self.energy_idx(energy)
        self.histogram = self.histogram.at[idx].add(1)
        self.entropy = self.entropy.at[idx].add(f)

    def run(self, key: PRNGKeyArray):
        init_key, key = jax.random.split(key)

        ising_state = self.system.random_state(init_key)

        def make_valid_state(a: tuple[IsingState, PRNGKeyArray]) -> tuple[IsingState, PRNGKeyArray]:
            ising_state, key = a

            jax.debug.print('{}', ising_state.energy)

            distance_from_energy_range = lambda energy: abs(
                abs(energy - (self.max_energy + self.min_energy) / 2) - (self.max_energy - self.min_energy) / 2
            )

            ising_state_distance = distance_from_energy_range(ising_state.energy)

            flip_key, key = jax.random.split(key)
            proposal_key, key = jax.random.split(key)

            new_ising_state = ising_state.flip(self.system, flip_key)
            new_ising_state_distance = distance_from_energy_range(new_ising_state.energy)

            r = jax.random.uniform(proposal_key)

            next_ising_state = jax.lax.cond(
                r < np.exp(ising_state_distance - new_ising_state_distance),
                lambda states: states[0],
                lambda states: states[1],
                (new_ising_state, ising_state),
            )

            return next_ising_state, key

        ising_state, key = jax.lax.while_loop(
            lambda a: (self.max_energy < a[0].energy) | (a[0].energy < self.min_energy),
            make_valid_state,
            (ising_state, key),
        )

        def wang_landau_step(
            a: tuple['WangLandauIsing', IsingState, PRNGKeyArray, int],
        ) -> tuple['WangLandauIsing', IsingState, PRNGKeyArray, int]:
            self_state, ising_state, key, i = a
            flip_key, key = jax.random.split(key)
            new_ising_state = ising_state.flip(self_state.system, flip_key)

            ising_state_entropy = self_state.entropy[self_state.energy_idx(ising_state.energy)]
            new_ising_state_entropy = self_state.entropy[self_state.energy_idx(new_ising_state.energy)]

            proposal_key, key = jax.random.split(key)
            r = jax.random.uniform(proposal_key)

            next_ising_state = jax.lax.cond(
                (r < np.exp(ising_state_entropy - new_ising_state_entropy)),
                lambda states: states[0],
                lambda states: states[1],
                (new_ising_state, ising_state),
            )

            def update_self_state(
                a: tuple['WangLandauIsing', IsingState, PRNGKeyArray, int],
            ) -> tuple['WangLandauIsing', IsingState, PRNGKeyArray, int]:
                self_state, next_ising_state, key, i = a
                self_state.increment_energy(next_ising_state.energy, self_state.f)

                self_state.f, self_state.histogram = jax.lax.cond(
                    self_state.flatness() <= 0.05,
                    lambda a: (a[0] / 2.0, a[1].at[:].set(0)),
                    lambda a: a,
                    (self_state.f, self_state.histogram),
                )

                jax.lax.cond(
                    i % 10 == 0,
                    lambda self_state: jax.debug.print(
                        '{} {:.5e} {:.5e} {} {}',
                        self_state.f,
                        np.nan_to_num(self_state.histogram.mean(where=self_state.histogram > 0), nan=1.0),
                        self_state.flatness(),
                        np.max(np.float32(self_state.histogram), where=self_state.histogram > 0, initial=0.0),
                        np.min(np.float32(self_state.histogram), where=self_state.histogram > 0, initial=np.inf),
                    ),
                    lambda self_state: None,
                    self_state,
                )

                jax.lax.cond(
                    i % 100 == 0,
                    lambda state: jax.debug.callback(WangLandauIsing.save_dataframe, state),
                    lambda state: None,
                    self_state,
                )
                return self_state, next_ising_state, key, i + 1

            return jax.lax.cond(
                (next_ising_state.energy >= self.min_energy) & (next_ising_state.energy <= self.max_energy),
                update_self_state,
                lambda s: s,
                (self_state, next_ising_state, key, i),
            )

        jax.debug.print('Starting Wang-Landau')
        state, ising_state, key, _ = jax.lax.while_loop(
            lambda state: state[0].f > 1e-8, wang_landau_step, (self, ising_state, key, 0)
        )

        self.f = state.f
        self.entropy = state.entropy
        self.histogram = state.histogram

    def save_dataframe(self):
        plt.plot(self.energies[self.histogram > 0], self.entropy[self.histogram > 0])
        plt.show()

        df = pl.DataFrame(
            {
                'energies': numpy.asarray(self.energies[self.histogram > 0]),
                'histogram': numpy.asarray(self.histogram[self.histogram > 0]),
                'entropy': numpy.asarray(self.entropy[self.histogram > 0]),
            }
        )

        df.write_csv('data.csv')


jax.tree_util.register_dataclass(
    WangLandauIsing,
    data_fields=['system', 'f', 'energies', 'histogram', 'entropy'],
    meta_fields=['min_energy', 'max_energy', 'bins'],
)


key = jax.random.key(420645820)
system = IsingSystem(10, -1.0, 0.0)

wang_landau = WangLandauIsing.init_state(system, 50.0, 200.0, 800)
wang_landau.run(key)

wang_landau.save_dataframe()
