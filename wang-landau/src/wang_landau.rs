use std::{fs::File, path::Path};

use color_eyre::Result;
use oorandom::Rand64;

use polars::prelude::*;

#[derive(Debug, Clone)]
pub struct WangLandau {
    min_energy: f64,
    max_energy: f64,
    bins: usize,
    entropy: Vec<f64>,
    histogram: Vec<u64>,

    f: f64,
    prev_energy: Option<f64>,

    rng: Rand64,
    i: usize,
}

impl WangLandau {
    pub fn new(min_energy: f64, max_energy: f64, bins: usize, seed: u128) -> WangLandau {
        WangLandau {
            min_energy,
            max_energy,
            bins,
            f: 1.,
            entropy: vec![0.; bins],
            histogram: vec![0; bins],
            prev_energy: None,
            rng: Rand64::new(seed),
            i: 0,
        }
    }

    pub fn f(&self) -> f64 {
        self.f
    }

    pub fn i(&self) -> usize {
        self.i
    }

    fn get_energy_idx(&self, energy: f64) -> usize {
        let portion = (energy - self.min_energy) / (self.max_energy - self.min_energy);
        ((portion * self.bins as f64).floor().max(0.) as usize).min(self.bins - 1)
    }

    pub fn accept_state_transition(&mut self, energy: f64) -> bool {
        self.i += 1;

        if energy < self.min_energy || energy > self.max_energy {
            false
        } else if let Some(prev_energy) = self.prev_energy {
            let r = self.rng.rand_float();
            let prev_idx = self.get_energy_idx(prev_energy);
            let idx = self.get_energy_idx(energy);

            let transition_probability = (self.entropy[prev_idx] - self.entropy[idx]).exp();
            let (transition, energy, idx) = if r < transition_probability {
                (true, energy, idx)
            } else {
                (false, prev_energy, prev_idx)
            };

            self.histogram[idx] += 1;
            self.entropy[idx] += self.f;
            self.prev_energy.replace(energy);

            transition
        } else {
            self.prev_energy.replace(energy);

            true
        }
    }

    pub fn flatness(&self) -> f64 {
        let size = self.histogram.iter().copied().filter(|i| *i > 0).count();

        if size > 2 {
            let max = self
                .histogram
                .iter()
                .copied()
                .filter(|i| *i > 0)
                .max()
                .unwrap() as f64;

            let min = self
                .histogram
                .iter()
                .copied()
                .filter(|i| *i > 0)
                .min()
                .unwrap() as f64;

            let mean = self
                .histogram
                .iter()
                .copied()
                .filter(|i| *i > 0)
                .sum::<u64>() as f64
                / (size as f64);

            1. - (max - min) / mean
        } else {
            0.
        }
    }

    pub fn check_flat(&mut self) -> bool {
        let flatness = self.flatness();

        if flatness > 0.95 {
            self.f /= 2.;
            self.histogram.fill(0);

            println!("Flat! {}: {:.3e}", self.i, self.f);
            self.i = 0;

            true
        } else {
            false
        }
    }

    pub fn to_csv(&self, filename: impl AsRef<Path>) -> Result<()> {
        let energies: Series = (0..self.bins)
            .map(|i| {
                (i as f64 / (self.bins + 1) as f64) * (self.max_energy - self.min_energy)
                    + self.min_energy
            })
            .collect();
        let energies = energies.with_name("energies".into());

        let entropies: Series = self.entropy.iter().collect();
        let entropies = entropies.with_name("entropy".into());

        let mut dataframe = DataFrame::new(vec![energies, entropies])?;

        let f = File::options()
            .truncate(true)
            .create(true)
            .read(true)
            .write(true)
            .open(filename)?;
        let csv_writer = polars::io::csv::write::CsvWriter::new(f);
        csv_writer
            .include_header(true)
            .with_separator(b',')
            .finish(&mut dataframe)?;

        Ok(())
    }
}
