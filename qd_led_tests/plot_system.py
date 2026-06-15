"""Clean diagnostic plots for a saved Sesame QD-LED solution.

Edit the USER SETTINGS block below, then run this file directly:

    python banddiag.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sesame
from sesame.observables import get_jn, get_jp

# =============================================================================
# USER SETTINGS
# =============================================================================
# Edit these values, then run: python banddiag.py
SIM_FILE = "compare_symmetric_m/m2/1dQD_V_55.gzip"
SAVE_PATH = None          # e.g. "banddiag_m2_55.png" or None
SHOW_FIGURE = True
FIGURE_TITLE = None       # e.g. "m = 2, V index 55" or None
# =============================================================================


def _finite_positive_min(*arrays, floor=1e-30):
    vals = []
    for arr in arrays:
        a = np.asarray(arr)
        good = np.isfinite(a) & (a > 0)
        if np.any(good):
            vals.append(np.nanmin(a[good]))
    if not vals:
        return floor
    return max(min(vals), floor)


def qd_region_edges(sys):
    """Return approximate left/right physical QD-stack edges in cm."""
    if getattr(sys, "has_qd", False) and hasattr(sys, "qd_sites"):
        qd_sites = np.asarray(sys.qd_sites, dtype=int)
        rqd = getattr(sys, "rqd", None)
        if rqd is not None:
            return sys.xpts[qd_sites[0]] - rqd, sys.xpts[qd_sites[-1]] + rqd
        return sys.xpts[qd_sites[0]], sys.xpts[qd_sites[-1]]
    if hasattr(sys, "eml_sites") and len(sys.eml_sites) > 0:
        eml = np.asarray(sys.eml_sites, dtype=int)
        return sys.xpts[eml[0]], sys.xpts[eml[-1]]
    return None, None


def add_region_markers(ax, sys, label=True):
    left, right = qd_region_edges(sys)
    if left is None or right is None:
        return
    left_nm, right_nm = left * 1e7, right * 1e7
    ax.axvspan(left_nm, right_nm, color="0.92", zorder=0, label="QDL" if label else None)
    ax.axvline(left_nm, color="0.25", linestyle=":", linewidth=1)
    ax.axvline(right_nm, color="0.25", linestyle=":", linewidth=1)




def _equilibrium_contact_potential(sys, side):
    """Return the equilibrium electrostatic potential at a contact.

    This mirrors Sesame's Solver.make_guess/contact Dirichlet convention so that
    old .gzip files, which usually do not store the applied voltage explicitly,
    can still be labeled by applied voltage.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    i = 0 if side == "left" else sys.nx - 1
    contact_idx = 0 if side == "left" else 1
    bc = sys.contacts_bcs[contact_idx]

    if bc in ("Schottky", "cam"):
        return -sys.contacts_WF[contact_idx] / sys.scaling.energy

    if bc in ("Ohmic", "Neutral"):
        if sys.rho[i] < 0:  # p-type
            return -sys.Eg[i] - np.log(abs(sys.rho[i]) / sys.Nv[i]) - sys.bl[i]
        return np.log(sys.rho[i] / sys.Nc[i]) - sys.bl[i]

    # Last-resort fallback: assume the saved contact value itself is equilibrium.
    return np.nan


def applied_voltage(sys, result):
    """Infer applied voltage in volts from a saved Sesame result.

    Newer saved results may include an explicit value.  For older files, infer it
    from the right contact potential, because IVcurve applies the bias on the
    right contact using

        v_right = v_right_eq + q * Vapp / Vt

    with q = +1 for p-type right contacts and q = -1 for n-type right contacts.
    """
    for key in ("applied_voltage", "voltage", "Vapp", "V"):
        if key in result:
            try:
                return float(np.asarray(result[key]).squeeze())
            except Exception:
                pass

    rc = sys.nx - 1
    v_eq = _equilibrium_contact_potential(sys, "right")
    if not np.isfinite(v_eq):
        return np.nan

    qsign = 1.0 if sys.rho[rc] < 0 else -1.0
    return float((result["v"][rc] - v_eq) / qsign * sys.scaling.energy)


def voltage_label(sys, result, precision=3):
    """Human-readable applied-voltage label for titles and legends."""
    V = applied_voltage(sys, result)
    if np.isfinite(V):
        return f"{V:.{precision}g} V"
    return "unknown V"


def link_currents(sys, result):
    efn = result.get("efn", np.zeros_like(result["v"]))
    efp = result.get("efp", np.zeros_like(result["v"]))
    v = result["v"]
    sites_i = np.arange(sys.nx - 1, dtype=int)
    sites_ip1 = sites_i + 1
    dl = sys.dx
    jn = get_jn(sys, efn, v, sites_i, sites_ip1, dl) * sys.scaling.current
    jp = get_jp(sys, efp, v, sites_i, sites_ip1, dl) * sys.scaling.current
    x_link = 0.5 * (sys.xpts[:-1] + sys.xpts[1:]) * 1e7
    return x_link, jn, jp, jn + jp


def compute_quantities(sys, result):
    az = sesame.Analyzer(sys, result)
    v = result["v"]
    efn = result.get("efn", np.zeros_like(v))
    efp = result.get("efp", np.zeros_like(v))
    vt = sys.scaling.energy

    x_nm = sys.xpts * 1e7
    Ec = -vt * (v + sys.bl)
    Ev = -vt * (v + sys.bl + sys.Eg)
    Efn = vt * efn
    Efp = vt * efp

    n = az.electron_density() * sys.scaling.density
    p = az.hole_density() * sys.scaling.density
    rho = (sys.rho - az.electron_density() + az.hole_density()) * sys.scaling.density

    potential = (v - v[0]) * vt
    x_link = 0.5 * (sys.xpts[:-1] + sys.xpts[1:]) * 1e7
    field = -np.diff(v * vt) / sys.dx
    xj, jn, jp, jtot = link_currents(sys, result)

    return {
        "x_nm": x_nm,
        "Ec": Ec,
        "Ev": Ev,
        "Efn": Efn,
        "Efp": Efp,
        "n": n,
        "p": p,
        "rho": rho,
        "potential": potential,
        "x_link_nm": x_link,
        "field": field,
        "xj_nm": xj,
        "jn": jn,
        "jp": jp,
        "jtot": jtot,
    }


def plot_banddiag(sim_file, save=None, show=True, title=None):
    sys, result = sesame.load_sim(str(sim_file))
    q = compute_quantities(sys, result)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    axes = axes.ravel()
    V_label = voltage_label(sys, result)
    fig.suptitle(title or f"Sesame diagnostic: {Path(sim_file).name} ({V_label})", fontsize=14)

    # 1. Band diagram
    ax = axes[0]
    add_region_markers(ax, sys)
    ax.plot(q["x_nm"], q["Ec"], color="tab:blue", lw=2, label=r"$E_C$")
    ax.plot(q["x_nm"], q["Ev"], color="tab:red", lw=2, label=r"$E_V$")
    ax.plot(q["x_nm"], q["Efn"], color="tab:blue", lw=1.7, ls="--", label=r"$E_{F,n}$")
    ax.plot(q["x_nm"], q["Efp"], color="tab:red", lw=1.7, ls="--", label=r"$E_{F,p}$")
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel("Energy [eV]")
    ax.set_title("Band diagram")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # 2. Carrier densities
    ax = axes[1]
    add_region_markers(ax, sys)
    ax.semilogy(q["x_nm"], np.maximum(q["n"], 1e-30), color="tab:blue", lw=2, label="n")
    ax.semilogy(q["x_nm"], np.maximum(q["p"], 1e-30), color="tab:red", lw=2, label="p")
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel(r"Carrier density [cm$^{-3}$]")
    ax.set_title("Carrier densities")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # 3. Electrostatic potential
    ax = axes[2]
    add_region_markers(ax, sys)
    ax.plot(q["x_nm"], q["potential"], color="tab:purple", lw=2, label=r"$\phi-\phi(0)$")
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel("Potential [V]")
    ax.set_title("Electrostatic potential")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # 4. Electric field
    ax = axes[3]
    add_region_markers(ax, sys)
    ax.plot(q["x_link_nm"], q["field"], color="tab:green", lw=2, label="E")
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel(r"Electric field [V cm$^{-1}$]")
    ax.set_title("Electric field")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # 5. Charge density
    ax = axes[4]
    add_region_markers(ax, sys)
    ax.plot(q["x_nm"], q["rho"], color="tab:orange", lw=2, label=r"$p-n+N_D-N_A$")
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel(r"Charge density [cm$^{-3}$]")
    ax.set_title("Net charge density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # 6. Currents
    ax = axes[5]
    add_region_markers(ax, sys)
    ax.plot(q["xj_nm"], q["jn"], color="tab:blue", lw=2, label=r"$J_n$")
    ax.plot(q["xj_nm"], q["jp"], color="tab:red", lw=2, label=r"$J_p$")
    ax.plot(q["xj_nm"], q["jtot"], color="k", lw=2, ls="--", label=r"$J_n+J_p$")
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel(r"Current density [A cm$^{-2}$]")
    ax.set_title("Link currents")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def main():
    """Run the plot using the USER SETTINGS at the top of this file."""
    plot_banddiag(SIM_FILE, save=SAVE_PATH, show=SHOW_FIGURE, title=FIGURE_TITLE)


if __name__ == "__main__":
    main()
