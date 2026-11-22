"""Two supporting checks for the write-up on Jung et al. (2021) / Jo et al. (2024).

1. Mesh convergence of the Jung Fig. 3 (M = 2) device: the published injection
   mobility only reproduces the published current on the 1 nm transport-layer mesh,
   because the capture current is proportional to the HTL/ETL surface-node density.
2. Square pulse (Jo et al. 2024 style) with the Jung 2021 green parameters: after
   switch-off, a charge-imbalanced device keeps its excess majority carriers in the
   dots indefinitely, because the model has no emission path.

    python paper_checks.py   -> prints a table, writes figures/pulse_trapped_charge.png
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from qdtransient import presets, TransientSolver  # noqa: E402

V = np.round(np.arange(0, 10.0001, 0.1), 4)


def mesh_run(h_nm):
    dev = presets.jung2021(M=2, tau=1.5e-6, C=1e-30, tl_spacing=h_nm * 1e-7, hil_spacing=min(0.5, h_nm) * 1e-7)
    s = TransientSolver(dev)
    obs, _, _ = s.sweep(V, dwell=1e-5)
    J = np.array([o['J'] for o in obs]); L = np.array([o['L'] for o in obs]); E = np.array([o['EQE'] for o in obs])
    pH = np.array([o['pH'] for o in obs])
    k = int(np.argmax(E)); i10 = -1
    vth = V[np.argmax(L > 1.0)]
    return h_nm, J[i10], L[i10], 100 * E[k], V[k], 100 * E[i10], vth, pH[i10]


def pulse_run(case):
    kw = {'balanced': {}, 'electron-rich (dEv - dEc = +0.2 eV)': dict(dEv=0.44, dEc=0.24)}[case]
    s = TransientSolver(presets.jung2021(M=2, **kw))
    X = s.initial_state()
    tr, _ = s.pulse(X, 5.0, 5e-4, 2e-2, n_out=60)
    t = np.array([o['t'] for o in tr]); p = np.array([o['p_qd_mean'] for o in tr])
    n = np.array([o['n_qd_mean'] for o in tr]); R = np.array([o['R_rad'] for o in tr])
    return case, t, p, n, R


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=5) as ex:
        mesh = list(ex.map(mesh_run, [1.0, 0.5, 0.25]))
        pulses = list(ex.map(pulse_run, ['balanced', 'electron-rich (dEv - dEc = +0.2 eV)']))
    print('mesh_nm  J(10V)  L(10V)  EQEmax  @V  EQE(10V)  Vth  pH(10V)')
    for r in mesh:
        print(f'{r[0]:6.2f}  {r[1]:6.3f}  {r[2]:8.0f}  {r[3]:5.2f}  {r[4]:4.1f}  {r[5]:5.2f}  {r[6]:4.1f}  {r[7]:.2e}')

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for (case, t, p, n, R), ls in zip(pulses, ['--', '-']):
        tm = t * 1e3
        ax[0].semilogy(tm, np.where(p > 1e10, p, np.nan), ls, color='red', label=f'p, {case}')
        ax[0].semilogy(tm, np.where(n > 1e10, n, np.nan), ls, color='blue', label=f'n, {case}')
        ax[1].plot(tm, R / max(R.max(), 1), ls, color='black', label=case)
    for a in ax:
        a.axvspan(0, 0.5, color='#f2d7a6', alpha=0.4, lw=0)
        a.set_xscale('symlog', linthresh=0.01)
        a.set_xlabel('time [ms]  (5 V on for 0.5 ms, shaded; then 0 V)')
    ax[0].set(ylim=(1e14, 1e19), ylabel='QD density [cm$^{-3}$]', title='QD carriers')
    ax[1].set(ylim=(0, 1.05), ylabel='emission (normalised)', title='Emission')
    ax[0].legend(frameon=False, fontsize=7, loc='lower left')
    ax[1].legend(frameon=False, fontsize=7)
    fig.suptitle('Square pulse with the Jung 2021 green parameters (M = 2): no emission path, so excess carriers stay in the dots')
    out = os.path.join(HERE, 'figures', 'pulse_trapped_charge.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)
    for case, t, p, n, R in pulses:
        i_on = np.argmin(abs(t - 5e-4)); i_end = -1
        print(f'{case}: at end of pulse p={p[i_on]:.2e} n={n[i_on]:.2e}; 19.5 ms after switch-off p={p[i_end]:.2e} n={n[i_end]:.2e}; '
              f'emission peak/end-of-pulse = {R.max()/R[i_on]:.3f}')
