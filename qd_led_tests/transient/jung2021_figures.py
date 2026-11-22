"""Plot the stepped-transient reproductions of Jung et al. (2021) in the paper's layout.

    python jung2021_figures.py            # stepped-transient sweeps (default)
    python jung2021_figures.py steady     # history-connected steady states

Figures are written to figures/.  Run jung2021_runs.py first.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jung2021_runs import load  # noqa: E402

FIG = os.path.join(HERE, 'figures')
MODE = sys.argv[1] if len(sys.argv) > 1 else 'stepped'
MARK = ['s', 'o', '^', 'v', 'D', '<']
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 1.0, 'lines.linewidth': 1.4,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})

# flat-band edges (eV) for the Fig. 2b / SI Fig. 2 band diagrams
EDGES = {'HIL': (-3.6, -5.17), 'HTL': (-2.6, -5.6), 'ETL': (-4.0, -7.4)}


def marks(ax, V, y, m, color, every=1.0, **kw):
    sel = np.isclose(np.mod(V + 1e-9, every), 0, atol=1e-6)
    ax.plot(V[sel], y[sel], m, mfc='white', mec=color, ms=4.5, ls='none', **kw)


def jl_panel(ax, runs, labels, title, legend_title):
    axL = ax.twinx()
    for k, (r, lab) in enumerate(zip(runs, labels)):
        d = load(r, MODE)
        V = d['V']
        ax.plot(V, d['J'], '-', color='red')
        marks(ax, V, d['J'], MARK[k], 'red')
        L = np.where(d['L'] > 0, d['L'], np.nan)
        axL.semilogy(V, L, '-', color='black')
        marks(axL, V, L, MARK[k], 'black')
        ax.plot([], [], '-' + MARK[k], color='black', mfc='white', ms=4.5, label=lab)
    ax.set(xlim=(0, 10), ylim=(0, 1.0), xlabel='Voltage [V]', title=title)
    ax.set_ylabel('$J$ [A cm$^{-2}$]', color='red')
    ax.tick_params(axis='y', colors='red')
    axL.set(ylim=(1, 1e6), ylabel='$L$ [cd m$^{-2}$]')
    ax.legend(title=legend_title, frameon=False, fontsize=7, title_fontsize=7, loc='upper left')


def eqe_panel(ax, runs, labels, title, legend_title, ymax):
    for k, (r, lab) in enumerate(zip(runs, labels)):
        d = load(r, MODE)
        ax.plot(d['V'], 100 * d['EQE'], '-k')
        marks(ax, d['V'], 100 * d['EQE'], MARK[k], 'black')
        ax.plot([], [], '-' + MARK[k], color='black', mfc='white', ms=4.5, label=lab)
    ax.set(xlim=(0, 10), ylim=(0, ymax), xlabel='Voltage [V]', ylabel='EQE [%]', title=title)
    ax.legend(title=legend_title, frameon=False, fontsize=7, title_fontsize=7)


def fig3():
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    taus = ['0.5', '1.5', '5', '10']
    Cs = ['0', '1', '2', '5', '10', '20']
    Ms = ['2', '3', '4']
    jl_panel(ax[0, 0], [f'fig3_tau_{t}' for t in taus], ['0.5', '1.5', '5.0', '10.0'], '(a)  C = 0', r'$\tau$ [$\mu$s]')
    eqe_panel(ax[1, 0], [f'fig3_tau_{t}' for t in taus], ['0.5', '1.5', '5.0', '10.0'], '(b)  C = 0', r'$\tau$ [$\mu$s]', 20)
    jl_panel(ax[0, 1], [f'fig3_C_{c}' for c in Cs], ['0.0', '1.0', '2.0', '5.0', '10.0', '20.0'], r'(c)  $\tau$ = 1.5 $\mu$s', r'C [10$^{-31}$cm$^6$s$^{-1}$]')
    eqe_panel(ax[1, 1], [f'fig3_C_{c}' for c in Cs], ['0.0', '1.0', '2.0', '5.0', '10.0', '20.0'], r'(d)  $\tau$ = 1.5 $\mu$s', r'C [10$^{-31}$]', 20)
    jl_panel(ax[0, 2], [f'fig3_M_{m}' for m in Ms], Ms, r'(e)  C = 10$^{-30}$, $\tau$ = 1.5 $\mu$s', 'M')
    eqe_panel(ax[1, 2], [f'fig3_M_{m}' for m in Ms], Ms, r'(f)  C = 10$^{-30}$, $\tau$ = 1.5 $\mu$s', 'M', 7)
    fig.suptitle(f'Jung 2021 Fig. 3 reproduced with qdtransient ({MODE})')
    fig.savefig(os.path.join(FIG, f'fig3_{MODE}.png'), dpi=160)
    plt.close(fig)


def fig4():
    fig, ax = plt.subplots(3, 3, figsize=(12.5, 10.5), constrained_layout=True)

    def dens(a, run, style, mk=None):
        d = load(run, MODE)
        V = d['V']
        for y, c in ((d['p_qd_mean'], 'red'), (d['n_qd_mean'], 'blue')):
            a.semilogy(V, np.where(y > 1e10, y, np.nan), style, color=c)
            if mk:
                marks(a, V, y, mk, c)
        a.set(xlim=(0, 10), ylim=(1e15, 1e19), xlabel='Voltage [V]', ylabel='$p, n$ [cm$^{-3}$]')

    def rates(a, run, style, mk=None):
        d = load(run, MODE)
        V = d['V']
        for y, c in ((d['R_rad'], 'red'), (d['R_srh'], 'black'), (d['R_aug'], 'blue')):
            a.semilogy(V, np.where(y > 1e10, y, np.nan), style, color=c)
            if mk:
                marks(a, V, y, mk, c)
        a.set(xlim=(0, 10), ylim=(1e14, 1e19), xlabel='Voltage [V]', ylabel='$R$ [cm$^{-2}$ s$^{-1}$]')

    def eqe(a, run, style, mk=None, color='black', label=None):
        d = load(run, MODE)
        a.plot(d['V'], 100 * d['EQE'], style, color=color, label=label)
        if mk:
            marks(a, d['V'], 100 * d['EQE'], mk, color)
        a.set(xlim=(0, 10), ylim=(0, 6), xlabel='Voltage [V]', ylabel='EQE [%]')

    dens(ax[0, 0], 'fig4_balanced', '-'); ax[0, 0].set_title(r'(d) $\alpha_p:\alpha_n$=1:1, $\delta$=0')
    rates(ax[0, 1], 'fig4_balanced', '-'); ax[0, 1].set_title('(e) radiative / SRH / Auger')
    eqe(ax[0, 2], 'fig4_balanced', '-'); ax[0, 2].set_title('(f) EQE')

    for run, mk in (('fig4_alpha_0.5_1.5', '^'), ('fig4_alpha_1.5_0.5', 'v')):
        dens(ax[1, 0], run, '-', mk); rates(ax[1, 1], run, '-', mk)
        eqe(ax[1, 2], run, '-', mk, label=run.replace('fig4_alpha_', r'$\alpha_p:\alpha_n$=').replace('_', ':'))
    dens(ax[1, 0], 'fig4_balanced', ':'); rates(ax[1, 1], 'fig4_balanced', ':')
    eqe(ax[1, 2], 'fig4_balanced', ':', label='1:1')
    ax[1, 0].set_title(r'(g) $\alpha_p:\alpha_n$ = 0.5:1.5 ($\triangle$), 1.5:0.5 ($\nabla$)')
    ax[1, 1].set_title('(h)'); ax[1, 2].set_title('(i)'); ax[1, 2].legend(frameon=False, fontsize=7)

    for run, mk in (('fig4_delta_+0.4_gated', '^'), ('fig4_delta_-0.4_gated', 'v')):
        dens(ax[2, 0], run, '-', mk); rates(ax[2, 1], run, '-', mk)
    dens(ax[2, 0], 'fig4_balanced', ':'); rates(ax[2, 1], 'fig4_balanced', ':')
    eqe(ax[2, 2], 'fig4_balanced', ':', label=r'$\delta$=0')
    eqe(ax[2, 2], 'fig4_delta_+0.4_gated', '-', '^', label=r'$\delta$=+0.4, 1:1')
    eqe(ax[2, 2], 'fig4_delta_-0.4_gated', '-', 'v', label=r'$\delta$=-0.4, 1:1')
    eqe(ax[2, 2], 'fig4_delta_+0.4_alpha_0.5_1.5_gated', '-', color='blue', label=r'$\delta$=+0.4, 0.5:1.5')
    eqe(ax[2, 2], 'fig4_delta_-0.4_alpha_1.5_0.5_gated', '-', color='red', label=r'$\delta$=-0.4, 1.5:0.5')
    for run, c in (('fig4_delta_+0.4', 'grey'),):
        eqe(ax[2, 2], run, '--', color=c, label=r'$\delta$=+0.4, NOT gated')
    ax[2, 0].set_title(r'(j) $\delta=\Delta E_{V0}-\Delta E_{C0}$ = +0.4 ($\triangle$), -0.4 ($\nabla$)  [gated]')
    ax[2, 1].set_title('(k) [gated]'); ax[2, 2].set_title('(l)'); ax[2, 2].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f'Jung 2021 Fig. 4 reproduced with qdtransient ({MODE}; M = 3)')
    fig.savefig(os.path.join(FIG, f'fig4_{MODE}.png'), dpi=160)
    plt.close(fig)


def fig5():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axL = ax[0].twinx()
    cols = {'red': 'tab:red', 'green': 'tab:green', 'blue': 'tab:blue'}
    for c, col in cols.items():
        d = load(f'fig5_{c}', MODE)
        V = d['V']
        vmax = {'red': 6.0, 'green': 7.6, 'blue': 8.0}[c]
        s = V <= vmax + 1e-9
        ax[0].plot(V[s], d['J'][s], ':', color=col)
        axL.semilogy(V[s], np.where(d['L'][s] > 0, d['L'][s], np.nan), '-', color=col)
        ax[1].plot(V[s], 100 * d['EQE'][s], '-', color=col, label=c)
        i = np.argmax(d['EQE'][s])
        print(f'fig5 {c}: L({vmax})={d["L"][s][-1]:.0f} cd/m2  EQEmax={100*d["EQE"][s][i]:.2f}% @ {V[s][i]:.1f} V  '
              f'EQE({vmax})={100*d["EQE"][s][-1]:.2f}%  J({vmax})={d["J"][s][-1]:.3f}')
    ax[0].set(xlim=(0, 8), ylim=(0, 0.6), xlabel='Voltage [V]', ylabel='$J$ [A cm$^{-2}$] (dotted)')
    axL.set(ylim=(1, 1e6), ylabel='$L$ [cd m$^{-2}$] (solid)')
    ax[1].set(xlim=(0, 8), ylim=(0, 15), xlabel='Voltage [V]', ylabel='EQE [%]')
    ax[1].legend(frameon=False)
    fig.suptitle(f'Jung 2021 Fig. 5c,d (Table 1 parameters) with qdtransient ({MODE})')
    fig.savefig(os.path.join(FIG, f'fig5_{MODE}.png'), dpi=160)
    plt.close(fig)


def band_edges(pr, zq):
    z, phi = pr['z'] * 1e7, pr['phi']
    ec = np.empty_like(z); ev = np.empty_like(z)
    for i, zz in enumerate(z):
        if np.any(np.isclose(zz, zq * 1e7)):
            ec[i], ev[i] = -3.66, -5.94
        elif zz <= 20 + 1e-6:
            ec[i], ev[i] = EDGES['HIL']
        elif zz <= 40 + 1e-6:
            ec[i], ev[i] = EDGES['HTL']
        else:
            ec[i], ev[i] = EDGES['ETL']
    return ec - phi, ev - phi


def fig2_si():
    d = load('fig3_M_2', MODE)
    prof, zq = d['profiles'], d['z_qd']
    Vs = sorted(prof)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    aF = ax[0, 0].twinx()
    for k, V in enumerate(Vs):
        pr = prof[V]
        z = pr['z'] * 1e7
        ax[0, 0].plot(z, pr['phi'], '-' + MARK[k], color='black', mfc='white', ms=3.5, label=f'{V:g} V')
        aF.step(pr['zl'] * 1e7, pr['E'] / 1e6, where='mid', color='red')
        ec, ev = band_edges(pr, zq)
        if V <= 6:
            ax[0, 1].plot(z, ec, '-' + MARK[k], color='blue', mfc='white', ms=3.5)
            ax[0, 1].plot(z, ev, '-' + MARK[k], color='red', mfc='white', ms=3.5)
        ax[1, 0].semilogy(z, pr['p'], '-' + MARK[k], color='red', mfc='white', ms=3.5)
        ax[1, 0].semilogy(z, pr['n'], '-' + MARK[k], color='blue', mfc='white', ms=3.5)
    ax[0, 0].set(xlim=(30, 60), ylim=(-1, 12), xlabel='z [nm]', ylabel=r'$\phi$ [V]', title='(a) potential (black), field (red)')
    aF.set(ylim=(-1, 12), ylabel='F [10$^6$ V cm$^{-1}$]')
    ax[0, 0].legend(frameon=False, fontsize=7)
    ax[0, 1].set(xlim=(30, 60), ylim=(-12, 0), xlabel='z [nm]', ylabel='Energy [eV]', title='(b) $E_C$ (blue), $E_V$ (red)')
    ax[1, 0].set(xlim=(30, 60), ylim=(1e-1, 1e21), xlabel='z [nm]', ylabel='$p, n$ [cm$^{-3}$]', title='(c) holes (red), electrons (blue)')
    a = ax[1, 1]
    zq_nm = zq * 1e7
    for k, V in enumerate([4, 6, 8, 10]):
        i = int(np.argmin(np.abs(d['V'] - V)))
        for y, c, ls in ((d['U_rad_qd'][i], 'red', '-'), (d['U_srh_qd'][i], 'grey', ':'), (d['U_aug_qd'][i], 'blue', '--')):
            a.semilogy(zq_nm, y, ls + MARK[k + 1], color=c, mfc='white', ms=4)
        print(f'fig2d V={V}: Urad={d["U_rad_qd"][i]}  Usrh={d["U_srh_qd"][i]}  Uaug={d["U_aug_qd"][i]}')
    a.set(xlim=(37.5, 52.5), ylim=(1e22, 1e25), xlabel='z [nm]', ylabel='U [cm$^{-3}$ s$^{-1}$]',
          title='(d) $U_{RAD}$ (red), $U_{SRH}$ (grey), $U_{AUG}$ (blue) at 4,6,8,10 V')
    fig.suptitle(f'Jung 2021 Fig. 2 (M = 2) with qdtransient ({MODE})')
    fig.savefig(os.path.join(FIG, f'fig2_{MODE}.png'), dpi=160)
    plt.close(fig)

    # Supplementary Figs. 1-5 (whole device)
    fig, ax = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    aF = ax[0, 0].twinx()
    for k, V in enumerate(Vs):
        pr = prof[V]
        z = pr['z'] * 1e7
        ax[0, 0].plot(z, pr['phi'], '-' + MARK[k], color='black', mfc='white', ms=2.5, label=f'{V:g} V')
        aF.step(pr['zl'] * 1e7, pr['E'] / 1e6, where='mid', color='red')
        ec, ev = band_edges(pr, zq)
        ax[0, 1].plot(z, ec, '-', color='blue'); ax[0, 1].plot(z, ev, '-', color='red')
        ax[1, 0].semilogy(z, pr['p'], '-' + MARK[k], color='red', mfc='white', ms=2.5)
        ax[1, 0].semilogy(z, pr['n'], '-' + MARK[k], color='blue', mfc='white', ms=2.5)
        ax[1, 1].plot(z, pr['rho'], '-', label=f'{V:g} V')
    ax[0, 0].set(xlim=(0, 90), ylim=(-1, 12), xlabel='z [nm]', ylabel=r'$\phi$ [V]', title='SI Fig. 1')
    aF.set(ylim=(-1, 12), ylabel='F [10$^6$ V/cm]')
    ax[0, 0].legend(frameon=False, fontsize=7)
    ax[0, 1].set(xlim=(0, 90), ylim=(-16, 0), xlabel='z [nm]', ylabel='Energy [eV]', title='SI Fig. 2')
    ax[1, 0].set(xlim=(0, 90), ylim=(1e-1, 1e21), xlabel='z [nm]', ylabel='$p, n$ [cm$^{-3}$]', title='SI Fig. 3')
    ax[1, 1].set(xlim=(0, 90), ylim=(-50, 50), xlabel='z [nm]', ylabel=r'$\rho$ [C cm$^{-3}$]', title='SI Fig. 4')
    ax[1, 1].legend(frameon=False, fontsize=7)
    # SI Fig. 5: hole / electron current along the device (steady -> J_p + J_n uniform)
    from jung2021_runs import make_device, RUNS  # noqa
    from qdtransient import TransientSolver
    s = TransientSolver(make_device(RUNS['fig3_M_2'][1]))
    for k, V in enumerate(Vs):
        pr = prof[V]
        Fn, Fp = s.fluxes(pr['phi'], pr['n'], pr['p'])
        q = 1.602176634e-19
        ax[2, 0].plot(pr['zl'] * 1e7, q * Fp, '-' + MARK[k], color='red', mfc='white', ms=2.5)
        ax[2, 0].plot(pr['zl'] * 1e7, -q * Fn, '-' + MARK[k], color='blue', mfc='white', ms=2.5)
    ax[2, 0].set(xlim=(0, 90), ylim=(0, 1.0), xlabel='z [nm]', ylabel='J [A cm$^{-2}$]', title='SI Fig. 5: $J_p$ (red), $J_n$ (blue)')
    ax[2, 1].axis('off')
    fig.suptitle(f'Jung 2021 Supplementary Figs. 1-5 with qdtransient ({MODE})')
    fig.savefig(os.path.join(FIG, f'si_figs_{MODE}.png'), dpi=150)
    plt.close(fig)


def summary():
    """Print the numbers the paper quotes in its text, next to ours."""
    d = load('fig3_M_2', MODE)
    V = d['V']
    i10, i4 = np.argmin(abs(V - 10)), np.argmin(abs(V - 4))
    k = np.argmax(d['EQE'])
    thr = V[np.argmax(d['L'] > 1.0)]
    print(f"[{MODE}] M=2: L(10V)={d['L'][i10]:.0f} (paper 127,226)  J(10V)={d['J'][i10]:.3f} (0.83)  "
          f"Vth(L>1cd/m2)={thr:.1f} (2.2)  EQEmax={100*d['EQE'][k]:.2f}% @ {V[k]:.1f} V (5.2 @ 4.1)  "
          f"EQE(10V)={100*d['EQE'][i10]:.2f}% (3.2)")
    for M in (3, 4):
        d = load(f'fig3_M_{M}', MODE)
        thr = d['V'][np.argmax(d['L'] > 1.0)]
        print(f"   M={M}: Vth={thr:.1f}  L(10V)={d['L'][i10]:.0f}  EQEmax={100*d['EQE'].max():.2f}% @ {d['V'][np.argmax(d['EQE'])]:.1f} V")


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    summary()
    fig2_si()
    fig3()
    fig4()
    fig5()
    print('figures written to', FIG)
