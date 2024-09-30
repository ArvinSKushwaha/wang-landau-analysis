use ising::{IsingState, IsingSystem, Walker};
use oorandom::Rand64;
use wang_landau::WangLandau;

mod ising;
mod wang_landau;

fn main() -> color_eyre::Result<()> {
    const N: usize = 64;
    let seed = 12354789423215;
    let mut rng = Rand64::new(seed);

    let system = IsingSystem::<N> { j: -1., h: 0. };
    let state = IsingState::random(
        &system,
        ((rng.rand_u64() as u128) << 64) | (rng.rand_u64() as u128),
    );

    let mut walker = Walker::new(state);
    let mut wang_landau = WangLandau::new(
        -2. * (N as f64).powi(2),
        2. * (N as f64).powi(2),
        N * N * 4,
        ((rng.rand_u64() as u128) << 64) | (rng.rand_u64() as u128),
    );

    while wang_landau.f() > 1e-8 {
        walker.generate_proposal(rng.rand_u64());
        if wang_landau.accept_state_transition(walker.proposed_energy().unwrap()) {
            walker.accept_proposal();
        } else {
            walker.reject_proposal();
        }

        if wang_landau.check_flat() || wang_landau.i() % 1_000_000 == 0 {
            wang_landau.to_csv(format!("./ising-{}/data-{}.csv", N, wang_landau.f()))?;
        }
    }

    wang_landau.to_csv(format!("./ising-{}/data.csv", N))?;
    Ok(())
}
