"""Recreate Jung et al. (2021) Figure 3 with the offset-removed ("honest") emulation.

Base: green QD-LED (M=2, d=5 nm, Eg=2.28 eV), reported doping (HTL 1e17, HIL 2.8e19),
with the abrupt PEDOT/HTL valence step REMOVED (HIL valence aligned to HTL) so injected
holes flood the quasi-neutral HTL instead of being suppressed by exp(-dEv/kT).  LEE=20%.

Sweeps: (a,b) SRH lifetime tau, C~0 ; (c,d) Auger C, tau=1.5us ; (e,f) # QD layers M.

The M=4 stack is too stiff to solve cold at V=0, so `solve_iv` uses an alpha-continuation
prelude (ramp the QD injection prefactor from ~0 to target at V=0) before stepping voltage.
"""
import os
import numpy as np
import sesame
from sesame.solvers import Solver
from sesame.analyzer import Analyzer


def build(M=2, tau=1.5e-6, Cval=10e-31, alpha=0.9e-9, remove_offset=True):
    aff, Eg = 3.66, 2.28          # green QD
    t_hil = t_htl = 20e-7; t_etl = 40e-7; t_bqd = t_hil + t_htl
    r_qd = 2.5e-7; r_qds = 0.5e-7
    t_qdl = 2 * M * r_qd; t_aqd = t_bqd + t_qdl; t_total = t_aqd + t_etl
    dd, dd2 = 4e-7, 1.2e-7
    qc = t_bqd + (2 * np.arange(M) + 1) * r_qd
    x = np.concatenate((
        np.linspace(0, dd, 22, endpoint=False), np.linspace(dd, t_hil - dd2, 28, endpoint=False),
        np.linspace(t_hil - dd2, t_hil + dd2, 18, endpoint=False), np.linspace(t_hil + dd2, t_bqd - dd, 28, endpoint=False),
        np.linspace(t_bqd - dd, t_bqd, 16, endpoint=False), qc, np.linspace(t_aqd, t_aqd + dd, 16, endpoint=False),
        np.linspace(t_aqd + dd, t_total - dd, 28, endpoint=False), np.linspace(t_total - dd, t_total, 22)))
    s = sesame.Builder(x)
    hil_Eg = 2.0 if remove_offset else 1.57   # 2.0 -> EV_HIL aligned to HTL (offset removed)
    hil = {'Nc': 2.5e19, 'Nv': 2.5e19, 'Eg': hil_Eg, 'epsilon': 3.0, 'Et': 0, 'mu_e': 0.322e-3, 'mu_h': 0.322e-3, 'tau_e': 1.2e-6, 'tau_h': 1.2e-6, 'affinity': 3.6}
    htl = {'Nc': 2.5e19, 'Nv': 2.5e19, 'Eg': 3.0, 'epsilon': 3.5, 'Et': 0, 'mu_e': 2.0e-3, 'mu_h': 2.0e-3, 'tau_e': 1.2e-6, 'tau_h': 1.2e-6, 'affinity': 2.6}
    qdc = {'Nc': 2.5e19 * 0.13 ** 1.5, 'Nv': 2.5e19 * 0.45 ** 1.5, 'Eg': Eg, 'epsilon': 9.4, 'Et': 0, 'mu_e': 2e-6, 'mu_h': 1e-6, 'tau_e': tau, 'tau_h': tau, 'affinity': aff, 'Cn': Cval, 'Cp': Cval, 'B': 0.58e-12}
    etl = {'Nc': 2.5e19 * 0.24 ** 1.5, 'Nv': 2.5e19 * 0.59 ** 1.5, 'Eg': 3.4, 'epsilon': 8.5, 'Et': 0, 'mu_e': 2.0e-3, 'mu_h': 2.0e-3, 'tau_e': 1.2e-6, 'tau_h': 1.2e-6, 'affinity': 4.0}
    hr = lambda x: x <= t_hil; tr = lambda x: np.logical_and(x > t_hil, x <= t_bqd)
    qr = lambda x: np.logical_and(t_bqd < x, x < t_aqd); er = lambda x: x >= t_aqd
    s.add_material(etl, er); s.add_material(htl, tr); s.add_material(hil, hr); s.add_material(qdc, qr)
    dEc = etl['affinity'] - qdc['affinity']; dEv = qdc['affinity'] + qdc['Eg'] - (htl['affinity'] + htl['Eg'])
    s.add_qd(r_qds * 1e7, qd_mns=0.19, qd_mps=0.60, dEc=dEc, dEv=dEv, location=qr, r_qd=r_qd)
    s.qd_alpha_n = s.qd_alpha_p = alpha
    s.eqe_outcoupling = 0.20
    s.add_donor(1e17, er); s.add_acceptor(1e17, tr); s.add_acceptor(2.81e19, hr)
    s.contact_type('Ohmic', 'Ohmic'); s.contact_S(1e7, 1e7, 1e7, 1e7)
    return s


def solve_iv(s, V, alpha_target=None):
    """Voltage sweep with an alpha-continuation prelude at V=0 (rescues stiff multi-layer stacks)."""
    if alpha_target is None:
        alpha_target = s.qd_alpha_p
    nx = s.nx; rc = nx - 1; q = 1 if s.rho[nx - 1] < 0 else -1
    solver = Solver(use_mumps=False)
    solver.solve(s, compute='Poisson', tol=1e-8, maxiter=2000, verbose=False, htp=1)
    eq = solver.equilibrium.copy()
    # alpha continuation at V=0
    res = {'efn': np.zeros(nx), 'efp': np.zeros(nx), 'v': eq.copy()}
    for a in np.geomspace(1e-12, alpha_target, 8):
        s.qd_alpha_n = s.qd_alpha_p = a; solver.equilibrium = eq.copy()
        g = {k: res[k].copy() for k in res}; g['v'][rc] = eq[rc]
        out = solver.solve(s, guess=g, compute='all', tol=1e-6, maxiter=300, verbose=False, htp=1)
        if out is not None:
            res = out
    s.qd_alpha_n = s.qd_alpha_p = alpha_target
    J = np.full(len(V), np.nan); EQE = np.full(len(V), np.nan); Jem = np.full(len(V), np.nan)
    prev = None; last = res
    for i, vapp in enumerate(V):
        vd = vapp / s.scaling.energy
        g = {k: last[k].copy() for k in last}
        if prev is not None and vapp > 1:
            for k in g:
                g[k] = last[k] + (last[k] - prev[k])   # linear extrapolation guess
        g['v'][rc] = eq[rc] + q * vd; solver.equilibrium = eq.copy()
        out = solver.solve(s, guess=g, compute='all', tol=1e-6, maxiter=300, verbose=False, htp=1)
        if out is None:
            continue
        try:
            az = Analyzer(s, out)
            J[i] = az.full_current() * s.scaling.current
            Jem[i] = az.full_emissive_current() * s.scaling.current
            EQE[i] = az.full_emission()
        except Exception:
            pass
        prev = last; last = out
    return J, Jem, EQE


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    V = np.linspace(0, 10, 101)
    os.makedirs('fig3_data', exist_ok=True)
    for tau in [0.5e-6, 1.5e-6, 5e-6, 10e-6]:
        J, Jem, E = solve_iv(build(M=2, tau=tau, Cval=1e-33)); np.savez('fig3_data/tau_%g.npz' % (tau * 1e6), V=V, j=J, jem=Jem, eqe=E)
    for C in [1e-33, 1e-31, 2e-31, 5e-31, 10e-31, 20e-31]:
        J, Jem, E = solve_iv(build(M=2, Cval=C)); np.savez('fig3_data/C_%g.npz' % (C * 1e31), V=V, j=J, jem=Jem, eqe=E)
    for M in [2, 3, 4]:
        J, Jem, E = solve_iv(build(M=M, Cval=10e-31)); np.savez('fig3_data/M_%d.npz' % M, V=V, j=J, jem=Jem, eqe=E)
    print('done — data in fig3_data/')
