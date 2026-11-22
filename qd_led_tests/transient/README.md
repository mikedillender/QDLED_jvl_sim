# Stepped-transient reconstruction of Jung et al. (2021)

`qdtransient/` (repo root) is a standalone replacement for the Sesame-based reconstruction.
It solves Poisson + continuity for (φ, ln n, ln p) on a 1-D finite-volume mesh with a fully
implicit (backward-Euler) Newton solver, and applies the bias as a staircase: every voltage step
starts from the state left by the previous one. With the published parameters and no fitting
parameters it reproduces Figs. 2, 3, 4, 5 and SI Figs. 1–5 of

> S.-M. Jung et al., *npj Comput. Mater.* **7**, 122 (2021)

No leakage current, no ΔV<sub>bi</sub> shift and no α rescaling are applied (all three are needed
in the Sesame build).

What this implies for the published papers is written up in
[`docs/jmk_model_issues.md`](../../docs/jmk_model_issues.md).

## Quick start

```
pip install numpy scipy matplotlib          # + pymupdf for compare_with_paper.py
python jung2021_runs.py                     # 54 sweeps, ~3 min on 20 cores -> data/
python jung2021_figures.py                  # stepped-transient figures -> figures/
python jung2021_figures.py steady           # history-connected steady states
python procedure_comparison.py              # dwell / gating study -> figures/procedure_comparison.png
python compare_with_paper.py <Jung.pdf> <Jung_SI.pdf>   # paper panels stacked over ours
python side_by_side.py [<Jung.pdf> <Jung_SI.pdf>]       # panel-by-panel, published | ours
python paper_checks.py                      # mesh convergence + square-pulse trapped charge
```

```python
from qdtransient import presets, TransientSolver
s = TransientSolver(presets.jung2021(M=2, tau=1.5e-6, C=1e-30))
obs, profiles, X = s.sweep(np.arange(0, 10.01, 0.1), dwell=1e-5, profiles_at=[4, 10])
```

## The model, and why each choice is the one Jung et al. used

| Choice | Evidence in the paper |
|---|---|
| Transport-layer drift uses the electrostatic field only (no band offsets inside HIL/HTL/ETL), so carrier densities are continuous across PEDOT:PSS/TFB | SI Fig. 3: holes fall smoothly from 2.8e19 to ~3e17 cm⁻³ over 5 nm instead of dropping by exp(−0.43 eV/kT). SI Fig. 1: the HTL potential sits ~0.1 V *above* the HIL, which is the p⁺/p diffusion potential kT·ln(2.8e19/3e17), not a heterojunction. |
| Decoupled Dirichlet contacts: φ(anode) = V − 0.64 V with p = N<sub>A</sub>(PEDOT), φ(cathode) = 0; minority/cathode densities from Eq. (10) | SI Fig. 1 (anode at V − 0.64), SI Fig. 3 (p ≈ 2–3e19 at z = 0, n ≈ 10 cm⁻³ at z = 0 as Eq. (10) gives). In (φ, n, p) variables this is just a boundary condition and converges at every bias, V = 0 included. |
| QD nodes coupled to HTL/ETL **only** by one-way capture, Eq. (1) with F<sub>p</sub> = −∂φ/∂z − ΔE<sub>V</sub>/(q r) ≥ 0; hopping Eq. (5) with ν from SI Eqs. (1)–(2); no emission, no drift-diffusion link | Main-text Eq. (1); Jo et al. (2024) state the off-decay is "determined solely by the carrier recombination process, neglecting the carrier leakages". |
| Standard box-method control volumes on the node set {…, 39, 40, 42.5, 47.5, 50, 51, …} nm, i.e. 1.5 r for the outer QDs and 2 r for inner ones; the HTL/ETL surface boxes extend r/2 into the EML | Only these volumes make Fig. 2d self-consistent: ΣU·w with w = 3.75 nm per QD gives J(10 V) = 0.83 A cm⁻² and L(10 V) ≈ 1.27e5 cd m⁻², exactly the quoted values; SI Fig. 4 peak ρ = 42 C cm⁻³ needs the 1.75 nm HTL surface box. |
| QD charge included in Poisson | Jo et al. (2024) Fig. 6 (charged QDs bend the potential). |
| 0.1 V staircase, 10 µs per step (≈ 1e4 V s⁻¹) | See "Steady state or stepped transient?" below. |

## Agreement (stepped, 10 µs / 0.1 V)

| Quantity | Paper | qdtransient |
|---|---|---|
| Fig. 2d, 4 V: U<sub>RAD</sub>, U<sub>SRH</sub>, U<sub>AUG</sub> (cm⁻³ s⁻¹) | 7.4e22, 1.1e23, 9.9e22 | 7.2e22, 1.08e23, 9.6e22 |
| Fig. 2d, 10 V vs 4 V ratios (rad, SRH, Auger) | 15×, 4×, 55× | 15×, 4.2×, 54× |
| Fig. 3, M = 2: J(10 V), L(10 V) | 0.83 A cm⁻², 127,226 cd m⁻² | 0.81, 122,400 (FWHM 30 nm assumed) |
| Fig. 3, M = 2: V<sub>th</sub>, EQE<sub>max</sub>, EQE(10 V) | 2.2 V, 5.2 % @ 4.1 V, 3.2 % | 2.2–2.3 V, 5.24 % @ 4.1 V, 3.24 % |
| Fig. 3b, τ = 1.5 µs, EQE at 3/4/5/6/10 V | ≈ 4.7/9.3/12/13.7/16.8 % | 4.5/9.2/12.0/13.7/16.8 % |
| Fig. 4f (M = 3, balanced): EQE peak | 5.2 % @ 6.7 V | 5.24 % @ 6.7 V |
| Fig. 4g (1.5:0.5): majority at 3/4/5/10 V | 2e16/1e18/2.3e18/5.5e18 | 2.3e16/1.0e18/2.5e18/5.7e18 |
| Fig. 4i: EQE peak, EQE(10 V) | 5.2 % @ 4.0 V, 1.8 % | 5.24 % @ 3.9 V, 1.80 % |
| Fig. 4j/l (δ = +0.4, gated): majority 5/10 V, minority 5/10 V, EQE 5/10 V | 4e18/2.9e18, 5e15/1.4e17, 2.5/3.25 % | 3.7e18/2.6e18, 5.0e15/1.4e17, 2.6/3.3 % |
| Fig. 5 red: L(6 V), EQE<sub>max</sub>, EQE(6 V) | 25,867, 10.5 % @ 3.6 V, 9.1 % | 26,068, 10.46 % @ 3.6 V, 9.10 % |
| Fig. 5 green: L(7.6 V), EQE<sub>max</sub>, EQE(7.6 V) | 72,905, 7.7 % @ 3.8 V, 5.6 % | 69,703, 7.75 % @ 3.8 V, 5.63 % |
| Fig. 5 blue: L(8 V), EQE<sub>max</sub>, EQE(8 V) | 1,633, 1.5 % @ 5.6 V, 1.2 % | 1,710, 1.56 % @ 5.6 V, 1.26 % |

Paper values in Figs. 3b and 4 are read off the plots by eye (±10–20 %); the rest are quoted in the text.

## What caused the discrepancies in the Sesame build

1. **Abrupt EQE turn-on (thesis Fig. B.6).** Below the capture threshold, the paper's model has no
   equation for the QD occupation: capture is one-way and there is no emission, so the dots keep
   whatever the voltage history put in them. A direct Newton steady-state solve is therefore
   singular there. `sesame/observables.py` regularises it by keeping the Scharfetter–Gummel current
   on every QD link, divided by 1e10 (`jp[qd_links] /= 1e10`). With the divisor raised to 1e20, the
   Sesame solve no longer converges even at 0.1 V. The residual link makes the problem solvable, but
   it does so by putting the dots in quasi-equilibrium with the transport layers. Running
   `qd_led_tests/emulation/recreate_fig3.build(M=2, tau=10e-6)` (on the `replicating_JMK` branch):

   | V | J (A cm⁻²) | p<sub>QD</sub> | n<sub>QD</sub> | EQE |
   |---|---|---|---|---|
   | 2.0 | 9e-15 | 7.8e14 | 7e5 | 0 |
   | 2.5 | 2e-12 | 4.3e17 | 4e7 | 0 |
   | 2.9 | 2e-10 | 1.9e18 | 1e9 | 18.3 % |
   | 3.0 | 7e-5 | 2.3e18 | 4e14 | 18.5 % |

   Holes flood the dots before either carrier can be captured (capture starts at ≈ 2.9 V with Sesame's
   V<sub>bi</sub>). The first captured electrons then meet p ≈ 2e18, so EQE = LEE·γp/(γp + 1/τ) from the
   first step. That equilibration through a 1e-16 cm² V⁻¹ s⁻¹ link takes ≫ 1 s, so it never happens in
   a time-stepped solution: there the storage term fixes the occupancy from history and the dots
   start empty. The constant leakage current used to smooth the EQE is therefore not needed.
   (`TransientSolver(..., sesame_leak=1e-10)` adds the same residual link, for experimenting with
   infinite-time steady states.)
2. **α × 0.4.** In the Sesame mesh the HTL/ETL surface node sits 0.25 nm from the QD layer on a
   0.25 nm grid, and `Builder.set_qd_poisson_widths` clips its Poisson box to 0.375 nm. p<sub>H</sub> is
   then the peak of a resolved accumulation layer. In Jung's mesh (1 nm transport-layer spacing, node
   exactly at the interface, unclipped box of 1 nm/2 + r/2 = 1.75 nm), the same sheet charge is spread
   over 1.75 nm and p<sub>H</sub> ≈ 2e20 cm⁻³ (SI Fig. 3). Since injection ∝ p<sub>H</sub>, the published α only
   works with the published mesh. With it, α = 3e-9 (and the Table 1 values) are used as given.
3. **V<sub>bi</sub>.** A decoupled contact is unproblematic when the unknowns are (φ, n, p); no 0.88 V
   reference shift is needed.
4. **HIL/HTL barrier.** Confirmed as an omission in the original drift term (see the table above).
   The "honest" Sesame build that aligned the HIL valence band was equivalent.

## Steady state or stepped transient?

For balanced injection the two agree above ~4 V. For imbalanced injection they do not, because the
majority carrier has to pile up in the dots until its own space charge balances the two capture
currents, which takes tens of µs near threshold. `figures/procedure_comparison.png`:

| Fig. 4i (α 1.5:0.5) | EQE peak position |
|---|---|
| Paper | 4.0 V |
| History-connected steady state | 3.3 V |
| 100 µs / 0.1 V | 3.4 V |
| **10 µs / 0.1 V** | **3.9 V** (majority and minority densities also match at every voltage) |
| 1 µs / 0.1 V | 5.0 V |

So the published curves are snapshots of a voltage ramp at roughly 1e4 V s⁻¹, not steady states.

## Where the published results are not self-consistent: Fig. 4 j–l

With ΔE<sub>V</sub> ≠ ΔE<sub>C</sub>, the carrier with the smaller offset reaches its capture threshold first
(≈ 1.6 V for δ = +0.4 eV). It charges the dots until it screens its own interface field, which
raises the field on the other side; the second carrier then starts at ≈ 2.8 V (steady state) or
≈ 3.6 V (10 µs staircase), and the EQE shows at most a 3.5 % spike. The paper instead has nothing in
the dots until 4.1 V, which is the max(ΔE<sub>C</sub>, ΔE<sub>V</sub>) threshold of its Eq. (2), followed by a
5.2 % spike.

Everything in Fig. 4j–l (majority 9e17 → 4e18 → 2.9e18, minority 2e15 → 1.4e17, the
SRH dip, EQE 5.2 → 2.5 → 3.25 %, and the 1.4 % curves for the doubly unbalanced cases) is reproduced
if capture is allowed only while **both** interface fields exceed their offsets
(`TransientSolver(..., gate_injection=True)`). That gate has no physical basis; it is the minimal
assumption that reproduces the published panels, and it is what a linear-potential reading of
Eq. (2) implies. It only matters when ΔE<sub>V</sub> ≠ ΔE<sub>C</sub>.

## Numerics

* Unknowns (φ, ln n, ln p) per node; densities span 1e-40 … 3e20 cm⁻³.
* Scharfetter–Gummel fluxes in the transport layers; capture/hopping on EML links.
* Banded Jacobian from 9-colour finite differences, row-equilibrated `solve_banded`; per-component
  Newton clipping; adaptive backward-Euler step (doubling on easy steps, quartering on failure).
* `max(F, 0)^1.5` is replaced by a softplus of width 1 kV/cm (≈ 1e-3 of operating fields): a dot
  layer that is charging up sits exactly at F ≈ 0, where the bare expression has an infinite
  curvature that stalls Newton.
* A full 0–10 V sweep (101 points, 10 µs each) takes ~20 s.

## Files

| File | |
|---|---|
| `../../qdtransient/device.py` | layers, QD layer, mesh/box volumes, contacts |
| `../../qdtransient/solver.py` | residual, Jacobian, Newton, time stepping, sweeps, pulses, observables |
| `../../qdtransient/presets.py` | Jung 2021 (Supp. Tables 1–2, Table 1) and Jo 2024 parameter sets |
| `jung2021_runs.py` | every configuration in Figs. 2–5; writes `data/*.npz` |
| `jung2021_figures.py` | paper-layout figures from the saved runs |
| `procedure_comparison.py` | steady vs. dwell time vs. gating for Fig. 4 g–l |
| `compare_with_paper.py` | stacks the published panels (CC BY 4.0) over ours |
| `side_by_side.py` | every panel of Figs. 2–5 and SI Figs. 1–5, published next to ours with matched axes |
| `paper_checks.py` | mesh-convergence table and the square-pulse trapped-charge figure used in the write-up |
