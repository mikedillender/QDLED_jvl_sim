"""Decoupled-emulation of Jung et al. (2021) Figure 3.

Extends `recreate_fig3.py` by referencing the built-in potential to the ITO/Al
*electrode work functions* (V_bi = 4.70 - 4.06 = 0.64 V) instead of the degenerate
PEDOT Fermi level (V_bi = 1.52 V that a self-consistent Ohmic contact produces).
That single reference change is the entire ~1 V turn-on offset (Vth = (2M/q)max(dEc,dEv) + V_bi).

WHY A REFERENCE SHIFT RATHER THAN A REAL DECOUPLED CONTACT:
JMK inject holes as if from a degenerate PEDOT reservoir *and* anchor V_bi to ITO -- two
mutually inconsistent statements a rate/reservoir model can assert but a self-consistent
drift-diffusion contact cannot.  A genuine decoupled contact ('cam': WF-referenced potential
+ ohmic injection) was implemented and *diverges at V=0*: forcing the reservoir hole density
at the shallow ITO potential requires an unbounded quasi-Fermi splitting at equilibrium.
So the honest, convergent emulation is: solve self-consistently (Ohmic, offset-removed build),
then correct the built-in-potential reference by dVbi = V_bi(PEDOT) - V_bi(ITO/Al) ~ 0.88 V,
i.e. plot vs (V - dVbi).  V_bi enters Vth as a rigid additive constant, so this is an exact
reference correction, not a fit.

Run `recreate_fig3.py` first to generate fig3_data/, then this script to plot the shifted curves.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DVBI = 0.88   # V_bi reference correction: PEDOT Fermi (1.52) -> ITO/Al electrodes (0.64)
JLEAK = 8e-3  # constant shunt leakage [A/cm2] not captured by the ideal simulation (smooths EQE turn-on)
Km, h, cc, q, lam, Vl = 683.0, 6.62607015e-34, 2.998e8, 1.602176634e-19, 545e-9, 0.984

def _lum(j, e):
    J = np.abs(j) * 1e4; E = np.clip(e, 0, None); g = np.abs(j) > 1e-6
    return Km * Vl * (h * cc / lam) * np.where(g, J, 0) * np.where(g, E, 0) / (q * np.pi)

def _load(f):
    d = np.load(f); V = d['V'] - DVBI; j = np.abs(d['j']); m = np.isfinite(j)
    return V[m], j[m], d['eqe'][m]

def plot(datadir='fig3_data', out='recreate_fig3_decoupled.png'):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    def panel(aJ, aE, files, labs, titles, cbar, elim):
        gs = plt.cm.Greys(np.linspace(0.42, 0.93, len(files))); aR = aJ.twinx()
        for f, lab, g in zip(files, labs, gs):
            V, j, e = _load(f); aJ.plot(V, j, '-', color=g, lw=1.6, label=lab)
            aR.semilogy(V, np.clip(_lum(j, e), 1e-1, None), '-', color='#d62728', lw=1.3)
            aE.plot(V, np.where(np.abs(j) > 0, e * 100 * np.abs(j) / (np.abs(j) + JLEAK), 0), '-', color=g, lw=1.8, label=lab)
        aJ.set(xlabel='Voltage [V]', ylabel='$J$ [A cm$^{-2}$]', xlim=(0, 9), ylim=(0, 1.0)); aJ.set_title(titles[0])
        aR.set(ylabel='$L$ [cd m$^{-2}$]', ylim=(1, 1e6)); aR.set_yscale('log'); aJ.legend(title=cbar, frameon=False, fontsize=7)
        aE.set(xlabel='Voltage [V]', ylabel='EQE [%]', xlim=(0, 9), ylim=(0, elim)); aE.legend(title=cbar, frameon=False, fontsize=7); aE.set_title(titles[1])
    panel(ax[0,0], ax[1,0], ['%s/tau_%g.npz'%(datadir,t) for t in [0.5,1.5,5,10]], ['0.5','1.5','5.0','10.0'], ('(a) J-L: tau sweep (C=0)','(b) EQE: tau sweep'), 'tau [us]', 20)
    panel(ax[0,1], ax[1,1], ['%s/C_%g.npz'%(datadir,c) for c in [0.01,1,2,5,10,20]], ['0','1','2','5','10','20'], ('(c) J-L: C sweep','(d) EQE: C sweep'), 'C [1e-31]', 20)
    panel(ax[0,2], ax[1,2], ['%s/M_%d.npz'%(datadir,m) for m in [2,3,4]], ['2','3','4'], ('(e) J-L: M sweep','(f) EQE: M sweep'), 'M', 7)
    fig.suptitle('Decoupled emulation of Jung 2021 Fig 3 (V_bi referenced to ITO/Al electrodes)', fontsize=12.5)
    fig.savefig(out, dpi=150); print('saved', out)

if __name__ == '__main__':
    plot()
