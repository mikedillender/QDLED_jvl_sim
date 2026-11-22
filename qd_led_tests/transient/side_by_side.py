"""Panel-by-panel, side-by-side comparison with Jung et al. (2021) and its SI.

Every row shows one published panel (left) next to the same quantity from
qdtransient (right), drawn with the paper's axis ranges, colours and markers.

    python side_by_side.py [Jung.pdf] [Jung_SI.pdf] [mode]

Defaults to the PDFs in ~/Documents/Papers and the stepped-transient runs
(run jung2021_runs.py first).  Needs pymupdf.  Writes figures/side_by_side_*.png;
those contain cropped figures from the paper (CC BY 4.0) and are git-ignored.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jung2021_runs import load, make_device, RUNS  # noqa: E402

PAPERS = os.path.join(os.path.expanduser('~'), 'Documents', 'Papers')
PAPER_PDF = os.path.join(PAPERS, 'Modelling charge transport and electro optical characteristics of quantum dot lightemittingdiodes.pdf')
SI_PDF = os.path.join(PAPERS, 'modelling charge transport eoc si.pdf')
FIG = os.path.join(HERE, 'figures')
Q = 1.602176634e-19
MK = ['s', 'o', '^', 'v', 'D', '<']
PINK, BLUE_BG = '#f9c9c9', '#dfe8f5'

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 1.1, 'lines.linewidth': 1.5,
                     'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': False, 'axes.labelweight': 'bold',
                     'axes.labelsize': 10})


def vlegend(ax, Vs, marks=MK, **kw):
    """Legend of black open markers, one per voltage (the paper's style)."""
    hs = [ax.plot([], [], m, color='k', mfc='white', ms=5)[0] for m in marks[:len(Vs)]]
    return ax.legend(hs, [f'{V:g} V' for V in Vs], frameon=False, fontsize=7, **kw)


# ---------------------------------------------------------------------- #
# published panels
# ---------------------------------------------------------------------- #
def raster_panel(pdf, page, box):
    """Crop (x0, y0, x1, y1) in pixels from the raster figure embedded on a page."""
    import pymupdf
    doc = pymupdf.open(pdf)
    pg = doc[page - 1]
    xref = pg.get_images()[0][0]
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1, :3]


def vector_panel(pdf, page, frac, dpi=250):
    """Render a region (page fractions) of a vector page."""
    import pymupdf
    doc = pymupdf.open(pdf)
    pg = doc[page - 1]
    r = pg.rect
    x0, y0, x1, y1 = frac
    clip = pymupdf.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)
    pix = pg.get_pixmap(dpi=dpi, clip=clip)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


C3 = [0, 658, 1317, 1975]          # Figs. 3-5: three equal columns (px)


def cell(col, r0, r1):
    return (C3[col], r0, C3[col + 1], r1)


# ---------------------------------------------------------------------- #
# helpers for our panels
# ---------------------------------------------------------------------- #
def at_int(V, y, every=1.0):
    sel = np.isclose(np.mod(V + 1e-9, every), 0, atol=1e-6)
    return V[sel], y[sel]


def mline(ax, V, y, color, mk=None, ls='-', every=1.0, **kw):
    ax.plot(V, y, ls, color=color, **kw)
    if mk:
        xv, yv = at_int(V, y, every)
        ax.plot(xv, yv, mk, color=color, mfc='white', ms=5, mew=1.1, ls='none')


def pos(y, floor=1e-30):
    y = np.asarray(y, dtype=float)
    return np.where(y > floor, y, np.nan)


def broken_log(fig, spec, top=(1e17, 1e21), bot=(1e-1, 1e1), ratio=(3.2, 1)):
    """Two stacked log axes imitating the paper's broken p,n axis."""
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=ratio, hspace=0.06)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharex=a1)
    for a, lim in ((a1, top), (a2, bot)):
        a.set_yscale('log'); a.set_ylim(*lim)
        a.minorticks_off()
    a1.set_yticks([10.0 ** k for k in range(int(np.log10(top[0])), int(np.log10(top[1])) + 1)])
    a2.set_yticks([10.0 ** k for k in range(int(np.floor(np.log10(bot[0]))), int(np.log10(bot[1])))])
    a1.spines['bottom'].set_visible(False); a2.spines['top'].set_visible(False)
    a1.tick_params(labelbottom=False, bottom=False)
    d = 0.012
    for a, y in ((a1, 0), (a2, 1)):
        kw = dict(transform=a.transAxes, color='k', clip_on=False, lw=1)
        a.plot((-d, d), (y - d * 3, y + d * 3), **kw)
        a.plot((1 - d, 1 + d), (y - d * 3, y + d * 3), **kw)
    return a1, a2


def regions(ax, z_htl=(0, 40), z_etl=(50, 90)):
    ax.axvspan(*z_htl, color=PINK, alpha=0.6, lw=0, zorder=0)
    ax.axvspan(*z_etl, color=BLUE_BG, alpha=0.8, lw=0, zorder=0)


EDGES = {'HIL': (-3.6, -5.17), 'HTL': (-2.6, -5.6), 'ETL': (-4.0, -7.4), 'QD': (-3.66, -5.94)}


def bands(pr, zq):
    z, phi = pr['z'] * 1e7, pr['phi']
    ec = np.empty_like(z); ev = np.empty_like(z)
    for i, zz in enumerate(z):
        k = 'QD' if np.any(np.isclose(zz, zq * 1e7)) else ('HIL' if zz <= 20 + 1e-6 else ('HTL' if zz <= 40 + 1e-6 else 'ETL'))
        ec[i], ev[i] = EDGES[k]
    return ec - phi, ev - phi


# ---------------------------------------------------------------------- #
# our panels (each draws into a SubplotSpec)
# ---------------------------------------------------------------------- #
def ours_fig2(key, fig, spec, mode):
    d = load('fig3_M_2', mode)
    prof, zq = d['profiles'], d['z_qd']
    Vs = sorted(prof)
    if key == 'c':
        a1, a2 = broken_log(fig, spec)
        for a in (a1, a2):
            regions(a, (30, 40), (50, 60))
            for k, V in enumerate(Vs):
                pr = prof[V]; z = pr['z'] * 1e7
                a.plot(z, pos(pr['p']), '-' + MK[k], color='red', mfc='white', ms=4)
                a.plot(z, pos(pr['n']), '-' + MK[k], color='blue', mfc='white', ms=4)
            a.set_xlim(30, 60)
        a2.set_xlabel('z [nm]'); a1.set_ylabel('p, n [cm$^{-3}$]')
        vlegend(a1, Vs, loc='upper right')
        return a1
    ax = fig.add_subplot(spec)
    if key == 'a':
        regions(ax, (30, 40), (50, 60))
        aF = ax.twinx()
        for k, V in enumerate(Vs):
            pr = prof[V]; z = pr['z'] * 1e7
            ax.plot(z, pr['phi'], '-' + MK[k], color='black', mfc='white', ms=4, label=f'{V:g} V')
            zl, E = pr['zl'] * 1e7, pr['E'] / 1e6
            aF.plot(zl, E, MK[k], color='red', mfc='white', ms=4)
            aF.step(zl, E, where='mid', color='red')
        ax.set(xlim=(30, 60), ylim=(-1, 12), xlabel='z [nm]', ylabel=r'$\phi$ [V]')
        aF.set_ylim(-0.5, 12); aF.set_ylabel(r'F [10$^6$ V cm$^{-1}$]', color='red')
        aF.tick_params(axis='y', colors='red')
        vlegend(ax, Vs, loc='upper right')
    elif key == 'b':
        regions(ax, (30, 40), (50, 60))
        for k, V in enumerate([v for v in Vs if v <= 6]):
            pr = prof[V]; z = pr['z'] * 1e7
            ec, ev = bands(pr, zq)
            ax.plot(z, ec, '-' + MK[k], color='blue', mfc='white', ms=4)
            ax.plot(z, ev, '-' + MK[k], color='red', mfc='white', ms=4, label=f'{V:g} V')
        ax.set(xlim=(30, 60), ylim=(-12, 0), xlabel='z [nm]', ylabel='Energy-level [eV]')
        vlegend(ax, [0, 2, 4, 6], loc='lower right')
    elif key == 'd':
        zqn = zq * 1e7
        mk = ['s', '^', 'o', 'D']          # paper: 4 V square, 6 V triangle, 8 V circle, 10 V diamond
        for k, V in enumerate([4, 6, 8, 10]):
            i = int(np.argmin(np.abs(d['V'] - V)))
            ax.semilogy(zqn, d['U_rad_qd'][i], '-' + mk[k], color='red', mfc='white', ms=5)
            ax.semilogy(zqn, d['U_srh_qd'][i], ':' + mk[k], color='grey', mfc='white', ms=5)
            ax.semilogy(zqn, d['U_aug_qd'][i], '--' + mk[k], color='blue', mfc='white', ms=5)
            ax.plot([], [], mk[k], color='k', mfc='white', label=f'{V} V')
        ax.plot([], [], '-', color='red', label='$U_{RAD}$')
        ax.plot([], [], ':', color='grey', label='$U_{SRH}$')
        ax.plot([], [], '--', color='blue', label='$U_{AUG}$')
        ax.set(xlim=(37.5, 52.5), ylim=(5.5e22, 1e25), xlabel='z [nm]', ylabel='U [cm$^{-3}$ s$^{-1}$]')
        ax.set_xticks(np.arange(37.5, 52.6, 2.5))
        ax.legend(frameon=False, fontsize=7, loc='upper left', ncol=2)
    return ax


def ours_fig3(key, fig, spec, mode):
    ax = fig.add_subplot(spec)
    sets = {'a': ([f'fig3_tau_{t}' for t in ['0.5', '1.5', '5', '10']], ['0.5', '1.5', '5.0', '10.0'], r'$\tau$ [$\mu$s]'),
            'c': ([f'fig3_C_{c}' for c in ['0', '1', '2', '5', '10', '20']], ['0.0', '1.0', '2.0', '5.0', '10.0', '20.0'], r'C [10$^{-31}$cm$^6$s$^{-1}$]'),
            'e': ([f'fig3_M_{m}' for m in ['2', '3', '4']], ['2', '3', '4'], 'M')}
    sets['b'], sets['d'], sets['f'] = sets['a'], sets['c'], sets['e']
    runs, labs, title = sets[key]
    mks = ['s', '^', 'D', '<'] if key in 'ab' else MK     # the paper's symbol order per sweep
    if key in 'ace':
        aL = ax.twinx()
        for k, (r, lab) in enumerate(zip(runs, labs)):
            d = load(r, mode)
            mline(ax, d['V'], d['J'], 'red', mks[k])
            mline(aL, d['V'], pos(d['L'], 1e-3), 'black', mks[k])
            ax.plot([], [], '-' + mks[k], color='black', mfc='white', ms=5, label=lab)
        aL.set_yscale('log'); aL.set_ylim(1, 1e6); aL.set_ylabel('L [cd m$^{-2}$]')
        ax.set(xlim=(0, 10), ylim=(0, 1.0), xlabel='Voltage [V]')
        ax.set_ylabel('J [A cm$^{-2}$]', color='red'); ax.tick_params(axis='y', colors='red')
        ax.legend(title=title, frameon=False, fontsize=7, title_fontsize=7, loc='upper left')
    else:
        for k, (r, lab) in enumerate(zip(runs, labs)):
            d = load(r, mode)
            mline(ax, d['V'], 100 * d['EQE'], 'black', mks[k], label=None)
            ax.plot([], [], '-' + mks[k], color='black', mfc='white', ms=5, label=lab)
        ax.set(xlim=(0, 10), ylim=(0, 7 if key == 'f' else 20), xlabel='Voltage [V]', ylabel='EQE [%]')
        ax.set_yticks(np.arange(0, 7.1, 1) if key == 'f' else np.arange(0, 20.1, 5))
        ax.legend(title=title, frameon=False, fontsize=7, title_fontsize=7,
                  loc='lower right' if key == 'b' else 'upper left')
    return ax


def ours_fig4(key, fig, spec, mode):
    ax = fig.add_subplot(spec)
    sets = {'d': [('fig4_balanced', '-', None)],
            'g': [('fig4_alpha_0.5_1.5', '-', '^'), ('fig4_balanced', ':', None), ('fig4_alpha_1.5_0.5', '-', 'v')],
            'j': [('fig4_delta_+0.4_gated', '-', '^'), ('fig4_balanced', ':', None), ('fig4_delta_-0.4_gated', '-', 'v')]}
    row = {'d': 'd', 'e': 'd', 'f': 'd', 'g': 'g', 'h': 'g', 'i': 'g', 'j': 'j', 'k': 'j', 'l': 'j'}[key]
    kind = {'d': 'dens', 'g': 'dens', 'j': 'dens', 'e': 'rate', 'h': 'rate', 'k': 'rate'}.get(key, 'eqe')
    for run, ls, mk in sets[row]:
        d = load(run, mode)
        V = d['V']
        if kind == 'dens':
            mline(ax, V, pos(d['p_qd_mean'], 1e10), 'red', mk, ls)
            mline(ax, V, pos(d['n_qd_mean'], 1e10), 'blue', mk, ls)
        elif kind == 'rate':
            mline(ax, V, pos(d['R_rad'], 1e10), 'red', mk, ls)
            mline(ax, V, pos(d['R_srh'], 1e10), 'black', mk, ls)
            mline(ax, V, pos(d['R_aug'], 1e10), 'blue', mk, ls)
        else:
            mline(ax, V, 100 * d['EQE'], 'black', mk, ls)
    if row == 'j' and kind == 'eqe':
        for run, c in (('fig4_delta_+0.4_alpha_0.5_1.5_gated', 'blue'), ('fig4_delta_-0.4_alpha_1.5_0.5_gated', 'red')):
            d = load(run, mode)
            ax.plot(d['V'], 100 * d['EQE'], '-', color=c)
        ax.text(6.3, 3.6, r'$\alpha_p:\alpha_n$=1.0:1.0', fontsize=7, weight='normal')
        ax.text(6.6, 1.75, r'0.5:1.5 (blue), 1.5:0.5 (red)', fontsize=7, weight='normal')
    if kind == 'dens':
        ax.set_yscale('log'); ax.set(ylim=(1e15, 1e19), ylabel='p, n [cm$^{-3}$]')
        ax.plot([], [], '-', color='red', label='p'); ax.plot([], [], '-', color='blue', label='n')
        ax.legend(frameon=False, fontsize=7, loc='lower right')
    elif kind == 'rate':
        ax.set_yscale('log'); ax.set(ylim=(1e14, 1e19), ylabel='R [cm$^{-2}$ s$^{-1}$]')
        for c, l in (('red', 'Radiative'), ('black', 'SRH'), ('blue', 'Auger')):
            ax.plot([], [], '-', color=c, label=l)
        ax.legend(frameon=False, fontsize=7, loc='lower right')
    else:
        ax.set(ylim=(0, 6), ylabel='EQE [%]')
    note = {'d': r'$\alpha_p:\alpha_n$ = 1.0:1.0, $\delta$ = 0',
            'g': r'$\alpha_p:\alpha_n$ = 0.5:1.5 ($\triangle$), 1.0:1.0 (dotted), 1.5:0.5 ($\nabla$)',
            'j': r'$\delta$ = +0.4 ($\triangle$), 0 (dotted), $-$0.4 eV ($\nabla$); gated capture'}[row]
    ax.text(0.03, 0.95, note, transform=ax.transAxes, fontsize=7, va='top', weight='normal')
    ax.set(xlim=(0, 10), xlabel='Voltage [V]')
    return ax


def ours_fig5(key, fig, spec, mode):
    ax = fig.add_subplot(spec)
    cols = {'red': 'red', 'green': 'green', 'blue': 'blue'}
    vmax = {'red': 6.0, 'green': 7.6, 'blue': 8.0}
    if key == 'c':
        aL = ax.twinx()
        for c, col in cols.items():
            d = load(f'fig5_{c}', mode)
            s = d['V'] <= vmax[c] + 1e-9
            ax.plot(d['V'][s], d['J'][s], ':', color=col)
            aL.semilogy(d['V'][s], pos(d['L'][s], 1e-3), '-', color=col, label=c.capitalize())
        ax.set(xlim=(0, 8), ylim=(0, 0.6), xlabel='Voltage [V]', ylabel='J [A cm$^{-2}$]')
        aL.set_ylim(1, 1e6); aL.set_ylabel('L [cd m$^{-2}$]', color='red'); aL.tick_params(axis='y', colors='red')
        aL.plot([], [], '-k', label='L'); aL.plot([], [], ':k', label='J')
        aL.legend(frameon=False, fontsize=7, loc='upper left')
    else:
        for c, col in cols.items():
            d = load(f'fig5_{c}', mode)
            s = d['V'] <= vmax[c] + 1e-9
            ax.plot(d['V'][s], 100 * d['EQE'][s], '-', color=col, label=c.capitalize())
        ax.set(xlim=(0, 8), ylim=(0, 15), xlabel='Voltage [V]', ylabel='EQE [%]')
        ax.legend(frameon=False, fontsize=7, loc='upper left')
    return ax


def ours_si(key, fig, spec, mode):
    d = load('fig3_M_2', mode)
    prof, zq = d['profiles'], d['z_qd']
    Vs = sorted(prof)
    if key == '3':
        a1, a2 = broken_log(fig, spec)
        for a in (a1, a2):
            for k, V in enumerate(Vs):
                pr = prof[V]; z = pr['z'] * 1e7
                a.plot(z, pos(pr['p']), '-' + MK[k], color='red', mfc='white', ms=3, markevery=1)
                a.plot(z, pos(pr['n']), '-' + MK[k], color='blue', mfc='white', ms=3)
            a.set_xlim(0, 90)
        a2.set_xlabel('z [nm]'); a1.set_ylabel('p, n [cm$^{-3}$]')
        vlegend(a1, Vs, loc='upper right')
        return a1
    ax = fig.add_subplot(spec)
    if key == '1':
        aF = ax.twinx()
        for k, V in enumerate(Vs):
            pr = prof[V]
            ax.plot(pr['z'] * 1e7, pr['phi'], '-' + MK[k], color='black', mfc='white', ms=3, label=f'{V:g} V')
            aF.step(pr['zl'] * 1e7, pr['E'] / 1e6, where='mid', color='red')
        ax.set(xlim=(0, 90), ylim=(-1, 12), xlabel='z [nm]', ylabel=r'$\phi$ [V]')
        aF.set_ylim(-0.5, 12); aF.set_ylabel(r'F [10$^6$ V cm$^{-1}$]', color='red'); aF.tick_params(axis='y', colors='red')
        vlegend(ax, Vs, loc='upper right')
    elif key == '2':
        for k, V in enumerate(Vs):
            pr = prof[V]
            ec, ev = bands(pr, zq)
            ax.plot(pr['z'] * 1e7, ec, '-' + MK[k], color='blue', mfc='white', ms=3)
            ax.plot(pr['z'] * 1e7, ev, '-' + MK[k], color='red', mfc='white', ms=3, label=f'{V:g} V')
        ax.set(xlim=(0, 90), ylim=(-16, 0), xlabel='z [nm]', ylabel='Energy-level [eV]')
        vlegend(ax, Vs, loc='lower right')
    elif key == '4':
        cols = ['red', 'orange', 'gold', 'green', 'dodgerblue', 'blueviolet']
        for k, V in enumerate(Vs):
            pr = prof[V]
            ax.plot(pr['z'] * 1e7, pr['rho'], '-', color=cols[k], label=f'{V:g} V')
        ax.set(xlim=(0, 90), ylim=(-50, 50), xlabel='z [nm]', ylabel=r'$\rho$ [C cm$^{-3}$]')
        ax.set_yticks(np.arange(-50, 51, 10))
        ax.legend(frameon=False, fontsize=7, loc='upper right')
    elif key == '5':
        from qdtransient import TransientSolver
        s = TransientSolver(make_device(RUNS['fig3_M_2'][1]))
        for k, V in enumerate(Vs):
            pr = prof[V]
            Fn, Fp = s.fluxes(pr['phi'], pr['n'], pr['p'])
            zl = pr['zl'] * 1e7
            ax.plot(zl, Q * Fp, '-' + MK[k], color='red', mfc='white', ms=3, markevery=2)
            ax.plot(zl, -Q * Fn, '-' + MK[k], color='blue', mfc='white', ms=3, markevery=2)
        ax.set(xlim=(0, 90), ylim=(0, 1.0), xlabel='z [nm]', ylabel='J [A cm$^{-2}$]')
        leg = vlegend(ax, Vs, loc='center right', ncol=2)
        ax.add_artist(leg)
        hp, = ax.plot([], [], '-', color='red'); hn, = ax.plot([], [], '-', color='blue')
        ax.legend([hp, hn], ['$J_p$', '$J_n$'], frameon=False, fontsize=7, loc='upper right', ncol=2)
    return ax


# ---------------------------------------------------------------------- #
# figure definitions: (title, [(panel id, published image getter, our drawer, our key)])
# ---------------------------------------------------------------------- #
def figures(paper, si):
    F2 = [(k, (lambda b: (lambda: raster_panel(paper, 3, b)))(b), ours_fig2, k)
          for k, b in (('a', (0, 0, 672, 500)), ('b', (672, 0, 1300, 500)),
                       ('c', (0, 500, 672, 1000)), ('d', (672, 500, 1300, 1000)))]
    F3 = []
    for k, (col, r0, r1) in {'a': (0, 0, 492), 'b': (0, 492, 1012), 'c': (1, 0, 492), 'd': (1, 492, 1012),
                             'e': (2, 0, 492), 'f': (2, 492, 1012)}.items():
        F3.append((k, (lambda b: (lambda: raster_panel(paper, 4, b)))(cell(col, r0, r1)), ours_fig3, k))
    F4 = []
    rows = {'d': 1, 'g': 2, 'j': 3}
    for first, r in rows.items():
        for j, k in enumerate([first, chr(ord(first) + 1), chr(ord(first) + 2)]):
            b = cell(j, int(r * 527.75) + 10, int((r + 1) * 527.75) - 4)
            F4.append((k, (lambda b: (lambda: raster_panel(paper, 6, b)))(b), ours_fig4, k))
    # Fig. 5c's luminance axis labels extend past the 1/3 column line, so split at x = 708
    F5 = [(k, (lambda b: (lambda: raster_panel(paper, 7, b)))(b), ours_fig5, k)
          for k, b in (('c', (0, 563, 708, 1126)), ('d', (718, 563, 1345, 1126)))]
    SI = [(f'Supplementary Fig. {k}', (lambda pg, fr: (lambda: vector_panel(si, pg, fr)))(pg, fr), ours_si, k)
          for k, pg, fr in (('1', 3, (0.20, 0.155, 0.80, 0.445)), ('2', 3, (0.20, 0.56, 0.80, 0.855)),
                            ('3', 4, (0.20, 0.125, 0.80, 0.42)), ('4', 4, (0.20, 0.50, 0.80, 0.79)),
                            ('5', 5, (0.20, 0.12, 0.80, 0.41)))]
    return {'fig2': ('Fig. 2 (M = 2, 0-10 V in 2 V steps)', F2),
            'fig3': ('Fig. 3 a-f', F3),
            'fig4': ('Fig. 4 d-l (M = 3; j-l with gated capture)', F4),
            'fig5': ('Fig. 5 c,d (Table 1 parameters)', F5),
            'si': ('Supplementary Figs. 1-5 (M = 2)', SI)}


def build(name, title, panels, mode, ncols=1):
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    W, H = 4.6, 3.4                      # inches per panel
    FW, FH = 2 * W * ncols + 0.6, H * nrows + 1.0
    fig = plt.figure(figsize=(FW, FH))
    outer = fig.add_gridspec(nrows, 2 * ncols, left=0.01, right=1 - 0.55 / FW, top=1 - 0.85 / FH,
                             bottom=0.6 / FH, wspace=0.32, hspace=0.36)
    for i, (pid, getimg, draw, key) in enumerate(panels):
        r, c = divmod(i, ncols)
        a = fig.add_subplot(outer[r, 2 * c])
        a.imshow(getimg()); a.axis('off')
        lab = pid if pid.startswith('Supp') else f'Fig. {name[3:]}{pid}'
        a.set_title(f'{lab}: published', fontsize=9, loc='left')
        top = draw(key, fig, outer[r, 2 * c + 1], mode)
        top.set_title(f'{lab}: qdtransient', fontsize=9, loc='left')
    how = '0.1 V staircase, 10 us per step' if mode == 'stepped' else 'history-connected steady state'
    fig.suptitle(f'Jung et al., npj Comput. Mater. 7, 122 (2021): {title}\n'
                 f'left: published (CC BY 4.0)    right: qdtransient, published parameters, {how}',
                 fontsize=10)
    out = os.path.join(FIG, f'side_by_side_{name}.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    paper = sys.argv[1] if len(sys.argv) > 1 else PAPER_PDF
    si = sys.argv[2] if len(sys.argv) > 2 else SI_PDF
    mode = sys.argv[3] if len(sys.argv) > 3 else 'stepped'
    os.makedirs(FIG, exist_ok=True)
    for name, (title, panels) in figures(paper, si).items():
        build(name, title, panels, mode, ncols=2 if len(panels) > 4 else 1)
