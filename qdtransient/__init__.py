"""qdtransient: stepped-transient drift-diffusion + QD rate-equation solver.

A standalone replacement for the Sesame-based steady-state reconstruction of
Jung et al. (2021) / Jo et al. (2024).  See README_transient.md.
"""
from .device import Device, Layer, QDLayer
from .solver import TransientSolver, ConvergenceError, luminance
from . import presets

__all__ = ['Device', 'Layer', 'QDLayer', 'TransientSolver', 'ConvergenceError',
           'luminance', 'presets']
