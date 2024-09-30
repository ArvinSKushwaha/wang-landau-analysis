#import "@local/catppuccin:0.1.0": get-palette, themes

#let course = "PY 722"
#let document_type = "Homework 1"
#let student = "Arvin Kushwaha"
#let professor = "Dr. Lubos Mitas"

#let mocha-palette = get-palette(themes.mocha)

#set document(title: [#course #document_type #student], author: (student))
#set page(
  "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
  header: [
    #set text(10pt)
    #smallcaps[#course #document_type]
    #h(1fr)
    _#(student)_
  ],
)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#show raw: set text(
    font: "FiraCode Nerd Font Mono",
    ligatures: true,
    historical-ligatures: true,
    discretionary-ligatures: true
)
#show par: set block(spacing: 0.55em)
#show heading: set block(above: 1.4em, below: 1em)

#align(center)[
  #v(5em)

  #text(size: 16pt)[#course #document_type] \
  #v(0.55em)
  #text(size: 14pt)[#student] \
  #v(0.55em)
  #text(size: 14pt)[#professor] \
  #v(0.55em)
  #text(size: 14pt)[#datetime(
      year: 2024,
      month: 9,
      day: 24,
    ).display("[month repr:long] [day padding:none], [year]")]

  #v(2em)
]

= Introduction

The Wang-Landau algorithm @wanglandau2001 is one of a collection of algorithms
categorized as "flat-histogram algorithms." In contrast to the very commonly
used Metropolis-Hastings Monte Carlo state sampling method that attempts to
recreate the state distribution of a system at some given temperature $T$, we
have an iteratively refined estimate of the *density of states* (DOS), which is
independent of temperature. This provides the notable advantage of being able
to capture critical phenomena and complex behavior without having to perform
simulations across multiple $T$ parameters. Suppose we have a system $S$, with
configuration space $Omega$ and energies $E: Omega -> [E_("min"), E_("max")]$.
We choose $E_("min"), E_1, E_2, dots, E_(n - 1), E_("max")$ such that
$E_("min") = E_0 < E_1 < E_2 < dots.c < E_(n - 1) < E_n = E_("max")$. This
gives us $n$ bins of energies: $[E_0, E_1], [E_1, E_2], dots, [E_(n - 2), E_(n
- 1)], [E_(n - 1), E_n]$, where we denote $B_i = [E_(i - 1), E_i]$. The
algorithm generally proceeds as follows: <wang-landau>

#block(
    width: 100%,
    fill: mocha-palette.colors.rosewater.rgb,
    inset: 0.65em,
    radius: 0.5em,
)[
    #line(length: 100%)
    #v(-0.75em)
    #h(0.15em) Wang-Landau (Vanilla)
    #v(-0.75em)
    #line(length: 100%)
    #v(-0.4em)
    
    #enum[
        Initialize the energy histogram, $forall i, H_i = 0$, and the entropy
        (i.e., $log("density of states")$) $forall i, S_i = 0$, where $H_i$
        records the number of times a visit to a state with energy in $B_i$ has
        occurred and $S_i$ is the current estimate of the entropy for the
        energy range $B_i$.

        #v(0.35em)

        Initialize a starting state to walk from: $x_0 in Omega$ and let our
        iteration $k = 0$. Initialize $f = 0$. Additionally, choose some
        tolerance $epsilon$. A common choice for this is $epsilon = 1 times
        10^(-8)$.

        #v(0.55em)
    ][
        Generate a state transition $x_k -> x'_k$ (with some specific
        properties). If $E(x_k) in B_a$ and $E(x'_k) in B_b$, then let the
        acceptance probability $P = min(1, exp(S_a - S_b))$. Let $r$ be a
        uniform random number in $[0, 1)$, if $r < P$, then let $x_(k + 1) =
        x'_k$, otherwise, let $x_(k + 1) = x_k$.

        #v(0.55em)
    ][
        If $E(x_(k + 1)) in B_i$, $H_i <- H_i + 1$ and $S_i <- S_i + f$.

        #v(0.55em)
    ][
        If the histogram $H$ is "flat," then we set $f <- f\/2$ and reset the
        histogram to zero: $forall i, H_i = 0$. If $f < epsilon$, then
        terminate the algorithm, the $S_i$ at the current state represents the
        $log("density of states")$.

        #v(0.35em)
        
        For computing "flatness", the test
        $ (max(H_i) - min(H_i)) / (1/n sum_i H_i) < p $
        is often used, where a common value for $p$ is 0.05. Adjusting $p$ lets
        you trade convergence time with accuracy.

        #v(0.55em)
    ][
        Let $k <- k + 1$, return to step 2.
    ]
]

There is some significant freedom in how this algorithm can be implemented to
reduce the computational expense of walking in the state space. An arbitrary proposal algorithm can be used. Consider a proposal probability function $p(x_1 -> x_2)$. Then, we can adjust the Wang-Landau algorithm as follows, in step 2, let the acceptance probability instead be:
$ P = min(1, exp(S_a - S_b) frac(p(x_2 -> x_1), p(x_1 -> x_2))). $
In the pleasant case of a symmetric proposal, like flipping a random spin for
an Ising model, this additional factor can be neglected, as $p(x_1 -> x_2) =
p(x_2 -> x_1)$, but in general this is not the case. Notice however, that this freedom of choice gives us the power to apply more "exotic" proposal distributions.

= Implementation and Application

With the results of the Wang-Landau algorithm, the density of states: $g(E) = exp(S(E))$, the construction of the partition function is rather straightforward and leads directly to the free energy, internal energy, and heat capacity:

$
Z & = sum_(s in Omega) e^(-beta E(s)) prop sum_(E) e^(-beta E) g(E) \
F & = - 1/beta ln Z = - 1/beta ln(sum_E e^(-beta E) g(E)) \
U & = -(diff ln Z) / (diff beta) = sum_E E e^(-beta E) g(E) \
C_V & = (diff U) / (diff T) = - beta^2 (diff U) / (diff beta) = beta^2 (diff^2
    ln Z) / (diff beta^2) = beta^2 sum_E E^2 e^(-beta E) g(E)
$

Because there are numerous derivatives here, we can take advantage of the JAX
library @jax2018github, which provide excellent compositional transformations
and JIT-ing utilities. In the following code, `log_dos` is the $S$
(microcanonical entropy) and `energies` are the energies of the bins. The
following are short samples of code, with some modifications for readability.
The full code can be found on GitHub here: #highlight(
    link("https://github.com/ArvinSKushwaha/wang-landau-analysis"),
    fill: mocha-palette.colors.rosewater.rgb.mix(white)).

```python
@jax.jit
def partition_function(beta: float, log_dos: Array, energies: Array):
    return np.sum(np.exp(log_dos - energies * beta))
```

The `partition_function` method is just a direct implementation of the equation
for $Z$ above.

```python
@jax.jit
def internal_energy(beta: float, log_dos: Array, energies: Array):
    min_energy = energies.min()
    return -jax.grad(
        lambda beta: np.log(partition_function(
            beta, 
            log_dos, 
            energies - min_energy
        ))
    )(beta) + min_energy

```

Here, we used the `jax.grad` function that takes a function and returns another
function that is the derivative of the function. In other words,
$#raw("jax.grad") : f |-> f'$. Notice the modification of `energies` to
`energies - energies.min()`. This is for the purposes of numerical stability,
to prevent $infinity$ from arising.

```python
@jax.jit
def free_energy_beta(beta: float, log_dos: Array, energies: Array):
    min_energy
    return -np.log(partition_function(
        beta,
        log_dos,
        energies - min_energy
    )) + min_energy * beta
```

Unfortunately, since $F$ diverges to $-infinity$ as $beta -> 0$ (at least for
models like the 2-D Ising Model), it's much nicer to plot $beta F$, which is
what the `free_energy_beta` method computes. Again, this is a direct
implementation of the equation above.

```python
def d_dT(func_of_beta: Callable[[float, Any, Any], Any]):
    return lambda beta, *args: -(beta**2) * jax.grad(func_of_beta)(beta, *args)

@jax.jit
def heat_capacity(beta: float, log_dos: Array, energies: Array):
    return d_dT(internal_energy_)(beta, log_dos - log_dos.min(), energies)
```

Here, the `d_dT` function takes the place of `jax.grad`, and effectively implements the following identity:
$ (diff f) / (diff T) = -beta^2 (diff f) / (diff beta). $
We implement it because heat capacity ($C_V$) is most cleanly represented as
$(diff U) / (diff T)$.
Here, we substitute `log_dos` with `log_dos - log_dos.min()` for numerical
stability.

= Analysis on the Periodic Ising Model

We analyze the periodic 2-dimensional Ising model with even side length ($N$)
given by the following Hamiltonian and configuration space:
$
Omega = {-1, +1}^(N^2) => |Omega| = 2^(N^2) \
cal(H)(s) = -J sum_(i = 1)^(N) sum_(j = 1)^(N) s_(i, j) (s_(i + 1, j) + s_(i, j +
1)), "where" s in Omega, s_(i, j) = s_(i + N, j) = s_(i, j + N) in {-1, +1}
$

For the state-space walker in the Wang-Landau algorithm, we use the naive step
of flipping a spin at a random site on the lattice. This trivially satisfies
detailed balance.

#v(0.65em)

In @dos_ising, we can confirm that the primary expected characteristics of the
periodic Ising model density of states are demonstrated, for example the
symmetry and approximately paraboloid shape of the curve. The correctness of
the converged DOS can be further confirmed by the characteristics of the
derived quantities in @u_ising, @f_ising, and @c_v_ising. We find that @u_ising
exactly matches the expected internal energy presented in @malsagov2017 and
@f_ising does also, albeit with a vertical offset that is likely a consequence
of the choice of normalization for the partition function. Finally, and most
relevantly is the estimation of the critical temperature from the heat capacity
($C_V$) of the system. In @c_v_ising, we see that as the size of the lattice
increases, the peak of the heat capacity, which is the estimator for the
critical temperature becomes a better approximation to the expected critical
temperature from the Onsager solution: $beta_"ONS"$. Thus, the Wang-Landau
algorithm has done a fairly effective job of computing the thermodynamic
properties of the 2D periodic Ising model, without having had to simulate the
model at every temperature of relevance.

#figure(
    image("DOS_Ising.png", width: 80%), 
    caption: [The rescaled density of states for a 2-dimensional Ising model
    with periodic boundary conditions and varying side lengths computed with
    the #link(<wang-landau>, 
        highlight(
            fill: mocha-palette.colors.rosewater.rgb.mix(white)
        )[Wang-Landau algorithm]
    ).]
    
) <dos_ising>

#figure(
    image("U_Ising.png", width: 80%), 
    caption: [The rescaled internal energy for a 2-dimensional Ising model
    with periodic boundary conditions and varying side lengths computed with
    the #link(<wang-landau>, 
        highlight(
            fill: mocha-palette.colors.rosewater.rgb.mix(white)
        )[Wang-Landau algorithm]
    ).]
) <u_ising>

#figure(
    image("F_Ising.png", width: 80%), 
    caption: [The rescaled free energy for a 2-dimensional Ising model
    with periodic boundary conditions and varying side lengths computed with
    the #link(<wang-landau>, 
        highlight(
            fill: mocha-palette.colors.rosewater.rgb.mix(white)
        )[Wang-Landau algorithm]
    ).]
) <f_ising>

#figure(
    image("C_V_Ising.png", width: 80%), 
    caption: [The rescaled heat capacity for a 2-dimensional Ising model
    with periodic boundary conditions and varying side lengths computed with
    the #link(<wang-landau>, 
        highlight(
            fill: mocha-palette.colors.rosewater.rgb.mix(white)
        )[Wang-Landau algorithm]
    ). It can be seen that the peak of the heat capacity curve is converging
    towards $beta_"ONS"$, the expected critical temperature.]
) <c_v_ising>

#bibliography("bib.yml")
