# JMK-emulation variants

Scripts that reproduce the Jung/Kim (2021, npj Comput. Mater.) QD-LED results using this
Sesame-based model, and document the modelling choices required to do so.

## Files
- `../jmk_emulation.py`       — v2: merged, *degenerate* transport region (field-free). Reproduces
                                Fig 5 currents/EQE but via unphysical degenerate HTL doping.
- `../jmk_emulation_honest.py`— v4 (preferred): **reported doping** (HTL 1e17, HIL 2.8e19) with the
                                abrupt PEDOT/HTL valence step **removed** (HIL valence aligned to HTL).
                                Injected holes flood the quasi-neutral HTL (matches their Supp. Fig. 3)
                                instead of being suppressed by exp(-dEv/kT). This is the physically
                                honest way to get their field-free/quasi-neutral HTL.
- `recreate_fig3.py`          — Full recreation of Jung 2021 Fig 3 (tau / Auger-C / M sweeps) on the
                                offset-removed green base, LEE=20%. Includes `solve_iv()` with an
                                **alpha-continuation** prelude that rescues stiff stacks (e.g. M=4)
                                which otherwise diverge if hit with full QD injection cold at V=0.

## Key findings (why this emulation is needed)
The discrepancies between this model and JMK trace to how the abrupt organic band steps are
handled under Boltzmann statistics:
1. **HIL/HTL (PEDOT/TFB) offset** — an abrupt 0.43 eV valence step imposes p_HTL = p_HIL*exp(-dEv/kT)
   ~ 1e12, deeply depleting the thin 1e17 HTL and creating a series bottleneck (20x low current,
   higher turn-on). Removing/smearing that step (as in the honest build) restores their behaviour.
2. **ITO/PEDOT (anode) reference** — Ohmic contact on the degenerate PEDOT pins Vbi to PEDOT's deep
   Fermi level (~5.6 eV) instead of the ITO work function (4.70 eV), raising Vbi by ~0.9 V and the
   turn-on by ~1 V. JMK reference Vbi to ITO and ignore the ITO/PEDOT barrier.
3. **Abrupt EQE onset** — Sesame represents the QD as a *density* node that equilibrates with the
   flooded HTL, so it is Auger-limited from turn-on. JMK use a coupled *rate-equation* QD occupation,
   which fills gradually and gives the smooth EQE rise. This is a structural framework difference.
Boltzmann statistics itself is *shared* with JMK (confirmed from their Eq. 4 Einstein relation) and
is not a source of disagreement.

## Decoupled emulation & the turn-on voltage
`recreate_fig3_decoupled.py` closes the last (~1 V) turn-on gap. The offset is entirely the
**built-in-potential reference**: a self-consistent Ohmic contact anchors V_bi to the degenerate
PEDOT Fermi level (1.52 V), while JMK anchor it to the ITO/Al electrode work functions
(4.70-4.06 = 0.64 V). A genuinely decoupled contact ('cam': WF potential + ohmic injection) was
implemented but **diverges at V=0** — forcing reservoir-level holes at the shallow ITO potential
needs an unbounded quasi-Fermi split, unsolvable in self-consistent drift-diffusion (it only works
in JMK's rate/reservoir formulation). Since V_bi is a rigid additive term in Vth, the honest fix is
to solve self-consistently and correct the reference by dVbi≈0.88 V. This is the same structural
lesson as the HTL profile and the EQE shape: the paper decouples things a DD model cannot.

## Figure 4 (charge-balance study)
`fig4_build.py` (green base, separate alpha_p/alpha_n and a qd_shift that sets dEv-dEc=2*qd_shift),
`fig4_run_conditions.py` (runs balanced / e-rich / h-rich / dEv±0.4; the hole-rich case needs
`fig4_hrich_continuation.py`'s alpha-continuation solver), and `plot_fig4.py` (assembles the 3x3
density / recomb-rate / EQE panels). Key result: even the *balanced* case (alpha_p=alpha_n, dEv=dEc)
is hole-rich from ~3-7 V because p_QD equilibration-floods from the degenerate HTL while n_QD is
supply-limited from the lightly-doped ETL; the charge balance evolves (n/p ~0.02 -> ~1) across the
operating window and shapes EQE well beyond turn-on. Symmetric injection (symmetric.py) avoids this
and gives cleaner roll-off, especially at high m (lower per-dot density -> less Auger).

## Shunt leakage & the EQE turn-on
The abrupt EQE jump is largely an artifact of the *ideal* simulation omitting real shunt/edge
leakage. Adding a constant J_leak (~8e-3 A/cm2) in the EQE denominator, EQE = LEE*U_rad/(J + J_leak),
suppresses EQE at low current and lets it recover once real injection dominates -> the smooth
S-shaped turn-on. With this, `recreate_fig3_decoupled.py` reproduces JMK Fig 3 b/d/f (smooth rise,
Auger roll-off, LEE saturation). Note this smooths the *rise*; the residual dip-and-recover in
strongly charge-imbalanced conditions (Fig 4 g-l) is a separate, DD-intrinsic effect.

## Summary of the full emulation (what it takes to reproduce Jung et al. 2021)
1. Offset removed at PEDOT/HTL  -> holes flood the quasi-neutral HTL (correct current, Supp Fig 3).
2. V_bi referenced to ITO/Al    -> correct ~2 V turn-on (the decoupled contact is unsolvable in DD).
3. Constant shunt leakage        -> smooth EQE turn-on (Fig 3 b/d/f).
Each is a place where JMK's rate/reservoir model decouples something a self-consistent
drift-diffusion model keeps coupled; two are removable band-step corrections, one (turn-on) is a
reference correction, and the EQE turn-on needs the leakage the ideal sim omits.
