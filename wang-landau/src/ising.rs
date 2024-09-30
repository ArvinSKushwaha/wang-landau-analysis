use std::ops::Not;

use oorandom::Rand64;

#[derive(Debug, Copy, Clone)]
pub struct IsingSystem<const N: usize> {
    pub j: f64,
    pub h: f64,
}

#[derive(Debug, Clone)]
pub struct IsingState<'a, const N: usize> {
    system: &'a IsingSystem<N>,
    spins: [[Spin; N]; N],
}

impl<'a, const N: usize> IsingState<'a, N> {
    pub fn new(system: &'a IsingSystem<N>, spins: [[Spin; N]; N]) -> Self {
        Self { system, spins }
    }

    pub fn random(system: &'a IsingSystem<N>, seed: u128) -> Self {
        let mut rng = Rand64::new(seed);
        let mut rand = rng.rand_u64();

        let spins = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if (i * N + j) % 32 == 0 {
                    rand = rng.rand_u64();
                }

                let spin = rand & (1 << ((i * N + j) % 32)) != 0;

                Spin(spin)
            })
        });

        Self::new(system, spins)
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Spin(pub bool);

impl Spin {
    fn value(self) -> f64 {
        if self.0 {
            -1.
        } else {
            1.
        }
    }
}

impl Not for Spin {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl std::fmt::Debug for Spin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 {
            write!(f, "🠇")
        } else {
            write!(f, "🠅")
        }
    }
}

impl<'a, const N: usize> IsingState<'a, N> {
    pub fn energy(&self) -> f64 {
        let mut energy = 0.;
        for i in 0..N {
            for j in 0..N {
                let iminus = (i + N - 1) % N;
                let jminus = (j + N - 1) % N;

                energy += self.spins[i][j].value() * self.system.h;
                energy += self.spins[i][j].value()
                    * (self.spins[iminus][j].value() + self.spins[i][jminus].value())
                    * self.system.j;
            }
        }

        energy
    }

    pub fn flip_energy_change(&self, i: usize, j: usize) -> f64 {
        let (iminus, iplus) = ((i + N - 1) % N, (i + 1) % N);
        let (jminus, jplus) = ((j + N - 1) % N, (j + 1) % N);

        let mut delta_energy = -2. * self.spins[i][j].value() * self.system.h;
        delta_energy -= 2.
            * self.spins[i][j].value()
            * (self.spins[iminus][j].value()
                + self.spins[iplus][j].value()
                + self.spins[i][jminus].value()
                + self.spins[i][jplus].value())
            * self.system.j;

        delta_energy
    }

    pub fn flip(&self, i: usize, j: usize) -> Self {
        let mut spins = self.spins;
        spins[i][j] = !spins[i][j];

        Self {
            system: self.system,
            spins,
        }
    }
}

pub struct Walker<State> {
    state: State,
    energy: f64,
    proposal: Option<(usize, usize)>,
}

impl<'a, const N: usize> Walker<IsingState<'a, N>> {
    pub fn new(state: IsingState<'a, N>) -> Self {
        Self {
            energy: state.energy(),
            state,
            proposal: None,
        }
    }

    #[inline]
    pub fn energy(&self) -> f64 {
        self.energy
    }

    pub fn generate_proposal(&mut self, value: u64) {
        let value = value as usize;

        if self.proposal.is_none() {
            self.proposal = Some((value % (N * N) / N, value % N))
        } else {
            panic!("Cannot generate proposal when one already set.");
        }
    }

    fn proposed_state(&self) -> Option<IsingState<'a, N>> {
        if let Some((i, j)) = self.proposal {
            Some(self.state.flip(i, j))
        } else {
            None
        }
    }

    pub fn proposed_energy(&self) -> Option<f64> {
        if let Some((i, j)) = self.proposal {
            Some(self.energy + self.state.flip_energy_change(i, j))
        } else {
            None
        }
    }

    pub fn accept_proposal(&mut self) {
        if self.proposal.is_some() {
            self.energy = self.proposed_energy().unwrap();
            self.state = self.proposed_state().unwrap();
            self.proposal = None;
        } else {
            panic!("Cannot accept proposal that does not exist.");
        }
    }

    pub fn reject_proposal(&mut self) {
        if self.proposal.is_some() {
            self.proposal = None;
        } else {
            panic!("Cannot accept proposal that does not exist.");
        }
    }
}
