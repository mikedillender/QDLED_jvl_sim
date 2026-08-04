# QDLED_jvl_sim

This repository contains an independent implementation of the QD-LED model described in:

> S.-M. Jung et al., “Modelling charge transport and electro-optical characteristics of quantum dot light-emitting diodes,” *npj Computational Materials* **7**, 2021.

The implementation is built on the [Sesame](https://github.com/usnistgov/sesame) drift-diffusion framework and was developed to examine the model’s predictions for the current–voltage, carrier-density, and external-quantum-efficiency characteristics of multilayer QD-LEDs.

## Motivation

The original publication notes that the associated code is available from the authors upon reasonable request. I was looking forward to receiving it, but unfortunately, “IP concerns” precluded its release. This repository therefore represents an independent reconstruction from the equations, parameters, and figures reported in the publication.

The reconstruction proved instructive.

## Reproduction of the published model

The model treats the organic transport layers using conventional drift-diffusion equations for doped inorganic semiconductors, while representing the quantum-dot layer as a sequence of discrete sites coupled through capture, emission, hopping, and recombination rates.

With several interpretive choices, the implementation can reproduce many of the principal trends reported in the original work. In particular, comparable current-density and EQE characteristics can be obtained when:

- the reported HIL/HTL injection barrier is not explicitly applied;
- the built-in voltage is referenced to the electrode work functions rather than to the Fermi level implied by the stated semiconductor contact conditions; and
- a small nonradiative leakage current is included when calculating the low-current EQE.

These adjustments are discussed more fully in my master’s thesis, to be published in August, and are represented in the secondary branch of this repository.

## Points requiring interpretation

Several aspects of the published formulation remain unclear.

### HIL/HTL interface

Using the stated energy levels and conventional heterojunction boundary conditions produces substantial depletion of the HTL and a corresponding voltage drop across it. This behavior is not apparent in the published carrier-density or electrostatic profiles.

Agreement is substantially improved when the HIL/HTL offset is omitted, effectively allowing the hole density to remain continuous across the interface. Whether this was an intended simplification or merely an especially cooperative interface is not specified.

### Contact potentials

A self-consistent ohmic contact to the degenerately doped PEDOT:PSS gives a built-in voltage larger than that implied by the electrode work functions. The resulting turn-on voltage is correspondingly higher than the published result.

Referencing the built-in potential directly to the ITO and Al work functions restores agreement. This is physically plausible for a highly conductive PEDOT:PSS contact, although it is not the result obtained from a literal application of the reported semiconductor parameters.

### EQE turn-on

The steady-state Newton solver predicts a relatively abrupt increase in EQE once both carrier species begin entering the quantum dots. The more gradual rise shown in the original paper can be reproduced empirically by including a leakage-current contribution in the denominator of the EQE.

A simplified transient rate model also produces carrier-density trends closer to the published curves, suggesting that voltage history, initialization, or the procedure used to approach steady state may influence the reported solution. In the absence of the original implementation, the precise origin is difficult to determine uniquely.

### Transport and injection assumptions

The capture coefficient contains a field-dependent factor proportional to \(F^{3/2}\), described as a polynomial approximation to Poole–Frenkel enhancement of the mobility. In practice, because the relevant field includes the band offset and the expression is not defined for negative fields, this term also conveniently imposes an effective injection threshold, neglecting any thermally assisted injection. 

Why exactly this is described as a "Poole-Frankel" enhancement of mobility, when the actual inter-dot hopping rate and transport-layer mobilities are field-independent, remains unclear.

More generally, the model treats the transport layers as constant-mobility, heavily doped inorganic semiconductors. This provides a tractable numerical system, although its applicability to disordered organic HTLs should be interpreted with appropriate enthusiasm.

## Limitations of this implementation

This reconstruction also has limitations of its own.

Most importantly, Sesame uses Boltzmann carrier statistics. That approximation is questionable in the high-accumulation regime near the QD interfaces. The close similarity between the electrostatic profiles generated here and those in the published work suggests that the original calculation may employ a comparable approximation, but this cannot be established without access to the source code.

The implementation should therefore be understood as a reconstruction of the reported model, not as a definitive physical description of QD-LED operation.

It is provided publicly so that the assumptions can be inspected, modified, and tested without requiring any further reasonable requests.

## Further discussion

A detailed description of the implementation, the required interpretive choices, and the comparison with the published results is provided in Chapter 4 and Appendix A of:

> M. Dillender, *Light and Charge Dynamics in Quantum Dot LEDs*, MIT master’s thesis, 2026.
