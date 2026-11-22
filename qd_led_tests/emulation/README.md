# JMK-emulation variants (Sesame)

Scripts that reproduce the Jung/Kim (2021, npj Comput. Mater.) QD-LED results using the
Sesame-based model, and document the modelling choices required to do so.

> **Status.** These Sesame scripts are kept as the record of the steady-state reconstruction
> (thesis Ch. 4 / App. B). The stepped-transient solver in `qdtransient/`
> ([`../transient/README.md`](../transient/README.md)) reproduces the published figures without
> the V<sub>bi</sub> shift, the leakage current or the α rescaling used here, and explains why they were
> needed. Notes marked *Superseded* below have been revised accordingly.

## Files
- `../jmk_emulation.py`       — v2: merged, *degenerate* transport region (field-free). Reproduces
                                Fig 5 currents/EQE but via unphysical degenerate HTL doping.
- `../jmk_emulation_honest.py`— v4 (preferred): **reported doping** (HTL 1e17, HIL 2.8e19), with the
                                abrupt PEDOT/HTL valence step **removed** (HIL valence aligned to HTL).
                                Injected holes flood the quasi-neutral HTL (matches their Supp. Fig. 3)
                                instead of being suppressed by exp(-dEv/kT).
- `recreate_fig3.py`          — Full recreation of Jung 2021 Fig 3 (tau / Auger-C / M sweeps) on the
                                offset-removed green base, LEE=20%. Includes `solve_iv()` with an
                                **alpha-continuation** prelude that rescues stiff stacks (e.g. M=4)
                                which otherwise diverge if hit with full QD injection cold at V=0.
- `recreate_fig3_decoupled.py`, `fig4_*.py`, `plot_*.py` — see below.

## Key findings
1. **HIL/HTL (PEDOT/TFB) offset** — an abrupt 0.43 eV valence step imposes p_HTL = p_HIL*exp(-dEv/kT)
   ~ 1e12, deeply depleting the thin 1e17 HTL and creating a series bottleneck (20x low current,
   higher turn-on). Removing the step (as in the honest build) restores their behaviour.
   *Confirmed:* JMK's drift term uses the electrostatic field only, so no offset between transport
   layers ever enters. Their SI Fig. 1 shows the resulting ~0.1 V p⁺/p diffusion potential at the
   interface.
2. **ITO/PEDOT (anode) reference** — Ohmic contact on the degenerate PEDOT pins Vbi to PEDOT's deep
   Fermi level instead of the ITO work function (4.70 eV), raising Vbi by ~0.9 V and the turn-on by
   ~1 V. JMK reference Vbi to ITO and ignore the ITO/PEDOT barrier.
3. **Abrupt EQE onset** — *Superseded.* Earlier attributed to a structural difference between a
   density node and a rate-equation QD occupation. The actual cause is the steady-state solve. Below
   threshold, one-way capture leaves the QD occupation undetermined. Sesame stays solvable only through
   the residual SG link on the QD links (`jp[qd_links] /= 1e10`; with 1e20 the Newton solve fails at
   0.1 V). That link fills the dots with holes from the flooded HTL before any capture (p_QD ≈ 2e18 at
   2.9 V with J ≈ 1e-10 A/cm², `recreate_fig3.build(M=2, tau=10e-6)`). The first captured electrons
   then give EQE = LEE·γp/(γp + 1/τ) at once. A time-stepped solution starts with empty dots and
   reproduces the gradual rise.

Boltzmann statistics itself is *shared* with JMK (confirmed from their Eq. 4 Einstein relation) and
is not a source of disagreement.

## Decoupled emulation & the turn-on voltage
`recreate_fig3_decoupled.py` closes the last (~1 V) turn-on gap. The offset is entirely the
**built-in-potential reference**: a self-consistent Ohmic contact anchors V_bi to the degenerate
PEDOT Fermi level (1.52 V), while JMK anchor it to the ITO/Al electrode work functions
(4.70-4.06 = 0.64 V). Since V_bi is a rigid additive term in Vth, the fix here is to solve
self-consistently and correct the reference by dVbi ≈ 0.88 V.

*Superseded in part:* a genuinely decoupled contact ('cam': WF potential + ohmic injection) diverges
at V = 0 in Sesame. That is a limitation of Sesame's quasi-Fermi-level formulation and equilibrium
start, not of drift-diffusion. With (φ, n, p) unknowns, φ(anode) = V − 0.64 V with p = N_A(PEDOT)
is an ordinary Dirichlet condition. `qdtransient` uses it at every bias, V = 0 included.

## Figure 4 (charge-balance study)
`fig4_build.py` (green base, separate alpha_p/alpha_n and a qd_shift that sets dEv-dEc=2*qd_shift),
`fig4_run_conditions.py` (runs balanced / e-rich / h-rich / dEv±0.4; the hole-rich case needs
`fig4_hrich_continuation.py`'s alpha-continuation solver), and `plot_fig4.py` (assembles the 3x3
density / recomb-rate / EQE panels).

*Superseded:* the finding that even the balanced case is hole-rich from ~3-7 V comes from the
same residual-link flooding as item 3. In `qdtransient`, Fig. 4d-i are reproduced with a 0.1 V
staircase at ~10 µs per step. The steady state peaks the hole-rich EQE at 3.3 V rather than the
published 4.0 V, so the published curves are not steady states. Fig. 4j-l cannot be obtained from the
stated equations: the smaller-offset carrier charges the dots from ~1.6 V and moves turn-on below
4.1 V. They are reproduced only if capture is gated on *both* interface fields exceeding their offsets
(see `../transient/README.md`).

## Shunt leakage & the EQE turn-on
*Superseded.* A constant J_leak (~8e-3 A/cm²) in the EQE denominator, EQE = LEE*U_rad/(J + J_leak),
smooths the abrupt Sesame turn-on into the published S-shape. It works, but it compensates for the
residual-link artifact of item 3 rather than for anything in JMK's model. The transient solver
reproduces Fig. 3 b/d/f with no leakage term.

## Summary of the Sesame emulation (what it takes to reproduce Jung et al. 2021 in Sesame)
1. Offset removed at PEDOT/HTL  -> holes flood the quasi-neutral HTL (correct current, Supp Fig 3).
   This is how JMK's drift term is actually written.
2. V_bi referenced to ITO/Al    -> correct ~2 V turn-on. A Sesame workaround for a decoupled
   Dirichlet contact.
3. Constant shunt leakage        -> smooth EQE turn-on. A workaround for the steady-state solve.
4. alpha × 0.3-0.4 (0.9e-9 in recreate_fig3.py, alpha_scale=0.4 in jmk_emulation_honest.py)
                                 -> compensates the 0.25 nm interface mesh / clipped Poisson box,
   which inflates the HTL surface density p_H that injection is proportional to. JMK's 1 nm mesh
   uses the published alpha.
