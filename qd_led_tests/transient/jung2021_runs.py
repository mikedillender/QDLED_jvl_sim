"""Run every simulation needed to reproduce Jung et al. (2021) Figs. 2-5 and SI Figs. 1-5
with the stepped-transient solver in ``qdtransient``.

Usage
-----
    python jung2021_runs.py                 # all runs, both sweep modes
    python jung2021_runs.py fig3 fig4       # subsets
    python jung2021_runs.py --mode stepped  # only the stepped-transient sweeps

Two sweep modes are run for every configuration:

* ``stepped`` -- staircase 0 -> 10 V in 0.1 V steps, DWELL seconds per step,
  starting from the V = 0 state and carrying the state from step to step.
  DWELL = 10 us (a 1e4 V/s ramp) reproduces Jung Fig. 4 g-i best.
* ``steady``  -- same staircase, but every step is integrated until the state
  is stationary (the history-connected steady state).

Results are written to data/<name>_<mode>.npz.
"""
import os
import sys
import time
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

from qdtransient import presets, TransientSolver  # noqa: E402

DATA = os.path.join(HERE, 'data')
DWELL = 1e-5                      # s per 0.1 V step
VOLTS = np.round(np.arange(0.0, 10.0001, 0.1), 4)
PROFILE_V = [0, 2, 4, 6, 8, 10]

# name -> (group, preset kwargs, solver kwargs)
RUNS = {}
for tau in [0.5, 1.5, 5.0, 10.0]:
    RUNS[f'fig3_tau_{tau:g}'] = ('fig3', dict(M=2, tau=tau * 1e-6, C=0.0), {})
for C in [0, 1, 2, 5, 10, 20]:
    RUNS[f'fig3_C_{C:g}'] = ('fig3', dict(M=2, tau=1.5e-6, C=C * 1e-31), {})
for M in [2, 3, 4]:
    RUNS[f'fig3_M_{M}'] = ('fig3', dict(M=M, tau=1.5e-6, C=10e-31), {})

base4 = dict(M=3, tau=1.5e-6, C=10e-31)
RUNS['fig4_balanced'] = ('fig4', dict(base4), {})
RUNS['fig4_alpha_0.5_1.5'] = ('fig4', dict(base4, alpha_p=1.5e-9, alpha_n=4.5e-9), {})
RUNS['fig4_alpha_1.5_0.5'] = ('fig4', dict(base4, alpha_p=4.5e-9, alpha_n=1.5e-9), {})
for gate in (False, True):
    g = '_gated' if gate else ''
    RUNS[f'fig4_delta_+0.4{g}'] = ('fig4', dict(base4, dEv=0.54, dEc=0.14), dict(gate_injection=gate))
    RUNS[f'fig4_delta_-0.4{g}'] = ('fig4', dict(base4, dEv=0.14, dEc=0.54), dict(gate_injection=gate))
    RUNS[f'fig4_delta_+0.4_alpha_0.5_1.5{g}'] = ('fig4', dict(base4, dEv=0.54, dEc=0.14, alpha_p=1.5e-9, alpha_n=4.5e-9), dict(gate_injection=gate))
    RUNS[f'fig4_delta_-0.4_alpha_1.5_0.5{g}'] = ('fig4', dict(base4, dEv=0.14, dEc=0.54, alpha_p=4.5e-9, alpha_n=1.5e-9), dict(gate_injection=gate))

for color in ['red', 'green', 'blue']:
    RUNS[f'fig5_{color}'] = ('fig5', dict(color=color), {})


def make_device(kw):
    kw = dict(kw)
    if 'color' in kw:
        return presets.jung2021_table1(kw.pop('color'), **kw)
    return presets.jung2021(**kw)


def run_one(args):
    name, mode = args
    group, dkw, skw = RUNS[name]
    out = os.path.join(DATA, f'{name}_{mode}.npz')
    t0 = time.time()
    s = TransientSolver(make_device(dkw), **skw)
    dwell = DWELL if mode == 'stepped' else None
    obs, prof, _ = s.sweep(VOLTS, dwell=dwell, profiles_at=PROFILE_V)
    keys = ['V', 'J', 'L', 'EQE', 'EQE_current', 'R_rad', 'R_srh', 'R_aug', 'p_qd_mean', 'n_qd_mean',
            'pH', 'nE', 'F_p', 'F_n', 't']
    arr = {k: np.array([o[k] for o in obs]) for k in keys}
    arr['p_qd'] = np.array([o['p_qd'] for o in obs])
    arr['n_qd'] = np.array([o['n_qd'] for o in obs])
    arr['U_rad_qd'] = np.array([o['U_rad_qd'] for o in obs])
    arr['U_srh_qd'] = np.array([o['U_srh_qd'] for o in obs])
    arr['U_aug_qd'] = np.array([o['U_aug_qd'] for o in obs])
    arr['profiles'] = np.frombuffer(pickle.dumps(prof), dtype=np.uint8)
    arr['z_qd'] = s.m.z[s.m.iQ]
    np.savez_compressed(out, **arr)
    return name, mode, time.time() - t0


def load(name, mode='stepped'):
    d = dict(np.load(os.path.join(DATA, f'{name}_{mode}.npz')))
    d['profiles'] = pickle.loads(d['profiles'].tobytes())
    return d


if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    args = sys.argv[1:]
    modes = ['stepped', 'steady']
    if '--mode' in args:
        i = args.index('--mode')
        modes = [args[i + 1]]
        args = args[:i] + args[i + 2:]
    groups = set(args) or {'fig3', 'fig4', 'fig5'}
    jobs = [(n, m) for n, (g, _, _) in RUNS.items() if g in groups for m in modes]
    print(f'{len(jobs)} runs')
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for name, mode, dt in ex.map(run_one, jobs):
            print(f'{name:40s} {mode:8s} {dt:6.1f}s', flush=True)
