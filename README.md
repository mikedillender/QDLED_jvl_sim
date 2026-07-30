About a year and a half ago, I emailed the authors of a paper titled ["Modelling charge transport and electro-optical characteristics of quantum dot light-emitting diodes"](https://www.nature.com/articles/s41524-021-00591-9) asking if I could get a copy of their code to model QD-LEDs. I believed this was reasonable, considering that the paper states, "The codes developed in this study are available from the authors upon reasonable request." However, they told me it would not be possible due to "IP concerns."

So, I spent about a month remaking their codebase using the [Sesame framework](https://github.com/usnistgov/sesame). There are a lot of problems with the model used here, but I am pretty sure all of them are also present in the two papers published by the Jong Min Kim group. The only problem I know this code has that I'm unsure the original has is that, by using Sesame, I am using the Boltzmann approximation, which does not apply in the high-accumulation regime these devices operate in. However, considering my electrostatics look very similar to those in the JMK papers, I think they might have done this too.  

My bigger problems are that the equation they use for tunneling is not rigorously justified (it seems to be somewhat arbitrarily chosen to make it easily computable in a solver) and that the transport layers are treated as heavily doped inorganic semiconductors, which is almost certainly untrue for the HTL. Moreover, the temperature and layer thickness scalings are both incorrectly predicted by this model.

There are probably a lot of problems with my implementation, but I do not believe the model is worth spending much more time on, so I thought I should just release it publicly.

## Relationship to Sesame

This repository contains a substantially modified version of the Sesame drift-diffusion–Poisson semiconductor-device simulator developed by Benoit Gaury, Paul Haney, and contributors. 

The original Sesame project is available at: https://github.com/usnistgov/sesame

The Sesame solver was modified here to support one-dimensional QD-LED simulations, including discretized quantum-dot sites, modified interfacial transport and recombination, and QD-specific electrostatics. 

The original Sesame code is Copyright 2017 University of Maryland and is redistributed under the BSD 3-Clause license. See `LICENSE.rst`.
