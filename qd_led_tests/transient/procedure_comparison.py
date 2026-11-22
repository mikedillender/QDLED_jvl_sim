"""How the *solution procedure* changes Jung et al. (2021) Fig. 4 g-l.

Runs the charge-imbalanced M = 3 devices with
  * the history-connected steady state at every voltage,
  * 0.1 V staircases with 1, 10 and 100 us per step,
  * with and without the "both fields above threshold" injection gate,
and overlays values read by eye from the published Fig. 4 (approximate, +-20 %).

    python procedure_comparison.py      -> figures/procedure_comparison.png
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
ALPHA = dict(M=3, tau=1.5e-6, C=1e-30, alpha_p=4.5e-9, alpha_n=1.5e-9)   # hole rich, 1.5:0.5
DELTA = dict(M=3, tau=1.5e-6, C=1e-30, dEv=0.54, dEc=0.14)               # electron rich, +0.4 eV

# read by eye from Jung 2021 Fig. 4 (V, value)
PAPER = {
    'g_maj': [(3, 2e16), (4, 1.0e18), (5, 2.3e18), (6, 3.2e18), (7, 3.9e18), (8, 4.5e18), (9, 5.0e18), (10, 5.5e18)],
    'g_min': [(4, 1.1e16), (5, 1.1e16), (6, 1.3e16), (7, 1.6e16), (8, 2.0e16), (9, 2.4e16), (10, 2.9e16)],
    'i_eqe': [(3, 0.3), (4, 5.2), (5, 3.5), (6, 2.7), (7, 2.4), (8, 2.1), (9, 1.95), (10, 1.8)],
    'j_maj': [(4.1, 9e17), (5, 4e18), (6, 3.7e18), (7, 3.4e18), (8, 3.2e18), (9, 3.0e18), (10, 2.9e18)],
    'j_min': [(4.1, 2e15), (5, 5e15), (6, 1.4e16), (7, 2.9e16), (8, 5.4e16), (9, 9e16), (10, 1.4e17)],
    'l_eqe': [(4.1, 5.2), (5, 2.5), (6, 2.7), (7, 2.9), (8, 3.05), (9, 3.15), (10, 3.25)],
}

JOBS = [('alpha', ALPHA, False, None), ('alpha', ALPHA, False, 1e-6), ('alpha', ALPHA, False, 1e-5),
        ('alpha', ALPHA, False, 1e-4),
        ('delta', DELTA, False, None), ('delta', DELTA, False, 1e-5),
        ('delta', DELTA, True, None), ('delta', DELTA, True, 1e-5)]


def run(job):
    name, kw, gate, dwell = job
    s = TransientSolver(presets.jung2021(**kw), gate_injection=gate)
    obs, _, _ = s.sweep(V, dwell=dwell)
    p = np.array([o['p_qd_mean'] for o in obs]); n = np.array([o['n_qd_mean'] for o in obs])
    e = np.array([o['EQE'] for o in obs])
    return job, np.maximum(p, n), np.minimum(p, n), e


def label(gate, dwell):
    s = 'steady state' if dwell is None else f'{dwell * 1e6:g} us / 0.1 V'
    return s + (', gated' if gate else '')


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=len(JOBS)) as ex:
        res = list(ex.map(run, JOBS))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    cols = {None: 'tab:grey', 1e-6: 'tab:orange', 1e-5: 'tab:blue', 1e-4: 'tab:green'}
    for (name, _, gate, dwell), maj, mnr, e in res:
        ls = '--' if gate else '-'
        c = cols[dwell] if not (name == 'delta' and gate) else ('tab:purple' if dwell else 'tab:pink')
        r = 0 if name == 'alpha' else 1
        ax[r, 0].semilogy(V, np.where(maj > 1e10, maj, np.nan), ls, color=c, label=label(gate, dwell))
        ax[r, 0].semilogy(V, np.where(mnr > 1e10, mnr, np.nan), ls, color=c, lw=0.9)
        ax[r, 1].plot(V, 100 * e, ls, color=c, label=label(gate, dwell))
    for r, (km, kn, ke) in enumerate((('g_maj', 'g_min', 'i_eqe'), ('j_maj', 'j_min', 'l_eqe'))):
        for k in (km, kn):
            x, y = zip(*PAPER[k])
            ax[r, 0].semilogy(x, y, 'ko', mfc='none', ms=6, label='Jung Fig. 4 (read by eye)' if k == km else None)
        x, y = zip(*PAPER[ke])
        ax[r, 1].plot(x, y, 'ko', mfc='none', ms=6, label='Jung Fig. 4 (read by eye)')
    ax[0, 0].set(title=r'Fig. 4g: $\alpha_p:\alpha_n$ = 1.5:0.5, majority & minority', ylim=(1e15, 1e19), xlim=(2, 10),
                 xlabel='Voltage [V]', ylabel='QD density [cm$^{-3}$]')
    ax[0, 1].set(title='Fig. 4i: EQE', ylim=(0, 6), xlim=(2, 10), xlabel='Voltage [V]', ylabel='EQE [%]')
    ax[1, 0].set(title=r'Fig. 4j: $\delta$ = +0.4 eV, majority & minority', ylim=(1e15, 1e19), xlim=(2, 10),
                 xlabel='Voltage [V]', ylabel='QD density [cm$^{-3}$]')
    ax[1, 1].set(title=r'Fig. 4l: $\delta$ = +0.4 eV, EQE', ylim=(0, 6), xlim=(2, 10), xlabel='Voltage [V]', ylabel='EQE [%]')
    for a in ax.flat:
        a.legend(frameon=False, fontsize=7)
    fig.suptitle('Solution procedure vs. the published Fig. 4 (M = 3, tau = 1.5 us, C = 1e-30)')
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    out = os.path.join(HERE, 'figures', 'procedure_comparison.png')
    fig.savefig(out, dpi=160)
    print('wrote', out)
