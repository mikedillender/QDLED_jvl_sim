"""Overlay diagnostic quantities from multiple saved Sesame .gzip files.

Edit the USER SETTINGS block below, then run this file directly:

    python banddiag_overlay.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sesame
from sesame.observables import get_jn, get_jp

# =============================================================================
# USER SETTINGS
# =============================================================================
# Edit this list, then run: python banddiag_overlay.py
SIM_FILES = [
    "compare_symmetric_m/m2/1dQD_V_40.gzip",
    "compare_symmetric_m/m2/1dQD_V_60.gzip",
    "compare_symmetric_m/m2/1dQD_V_80.gzip",
    "compare_symmetric_m/m2/1dQD_V_100.gzip",
    "compare_symmetric_m/m2/1dQD_V_120.gzip",
    "compare_symmetric_m/m2/1dQD_V_140.gzip",
]

# Optional labels. Set to None to auto-label by inferred applied voltage.
LABELS = None
# Example manual labels:
# LABELS = ["2.0 V", "2.75 V", "3.5 V"]

SAVE_PATH = None          # e.g. "banddiag_overlay.png" or None
SHOW_FIGURE = True
FIGURE_TITLE = "Overlayed Sesame diagnostics"
# =============================================================================


def qd_region_edges(sys):
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


def add_region_markers(ax, sys):
    left, right = qd_region_edges(sys)
    if left is None or right is None:
        return
    left_nm, right_nm = left * 1e7, right * 1e7
    ax.axvspan(left_nm, right_nm, color="0.94", zorder=0)
    ax.axvline(left_nm, color="0.35", linestyle=":", linewidth=1)
    ax.axvline(right_nm, color="0.35", linestyle=":", linewidth=1)




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


def make_labels(loaded, labels):
    """Return legend labels, using applied voltage when labels are not supplied."""
    if labels:
        if len(labels) != len(loaded):
            raise ValueError("Number of labels must match number of files.")
        return labels

    out = []
    for f, sys, result, _q in loaded:
        V = applied_voltage(sys, result)
        if np.isfinite(V):
            out.append(f"{V:.3g} V")
        else:
            out.append(Path(f).stem.replace("1dQD_V_", "V index "))
    return out


def plot_overlay(sim_files, labels=None, save=None, show=True, title="Overlayed Sesame diagnostics"):
    loaded = []
    for f in sim_files:
        sys, result = sesame.load_sim(str(f))
        loaded.append((f, sys, result, compute_quantities(sys, result)))

    if not loaded:
        raise ValueError("No simulation files were supplied.")

    labels = make_labels(loaded, labels)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    axes = axes.ravel()
    fig.suptitle(title, fontsize=14)

    # Draw QDL region from first file only, assuming same mesh/device.
    first_sys = loaded[0][1]
    for ax in axes:
        add_region_markers(ax, first_sys)

    cmap = plt.get_cmap("viridis")
    denom = max(len(loaded) - 1, 1)
    colors = [cmap(i / denom) for i in range(len(loaded))]

    for (f, sys, result, q), label, color in zip(loaded, labels, colors):
        # Band diagram: Ec solid, Ev dashed for each voltage/file.
        axes[0].plot(q["x_nm"], q["Ec"], color=color, lw=2, label=label)
        axes[0].plot(q["x_nm"], q["Ev"], color=color, lw=2, ls="--")

        # Densities: n solid, p dashed.
        axes[1].semilogy(q["x_nm"], np.maximum(q["n"], 1e-30), color=color, lw=2, label=label)
        axes[1].semilogy(q["x_nm"], np.maximum(q["p"], 1e-30), color=color, lw=2, ls="--")

        axes[2].plot(q["x_nm"], q["potential"], color=color, lw=2, label=label)
        axes[3].plot(q["x_link_nm"], q["field"], color=color, lw=2, label=label)
        axes[4].plot(q["x_nm"], q["rho"], color=color, lw=2, label=label)
        axes[5].plot(q["xj_nm"], q["jtot"], color=color, lw=2, label=label)

    axes[0].set_title("Band diagram")
    axes[0].set_xlabel("Position [nm]")
    axes[0].set_ylabel("Energy [eV]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Applied voltage", fontsize=8)
    axes[0].text(0.02, 0.03, "solid: $E_C$, dashed: $E_V$", transform=axes[0].transAxes, fontsize=9)

    axes[1].set_title("Carrier densities")
    axes[1].set_ylim(1e10, 1e21)
    axes[1].set_xlabel("Position [nm]")
    axes[1].set_ylabel(r"Carrier density [cm$^{-3}$]")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(title="Applied voltage", fontsize=8)
    axes[1].text(0.02, 0.03, "solid: n, dashed: p", transform=axes[1].transAxes, fontsize=9)

    axes[2].set_title("Electrostatic potential")
    axes[2].set_xlabel("Position [nm]")
    axes[2].set_ylabel("Potential [V]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    axes[3].set_title("Electric field")
    axes[3].set_xlabel("Position [nm]")
    axes[3].set_ylabel(r"Electric field [V cm$^{-1}$]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8)

    axes[4].set_title("Net charge density")
    axes[4].set_xlabel("Position [nm]")
    axes[4].set_ylabel(r"Charge density [cm$^{-3}$]")
    axes[4].axhline(0, color="0.2", lw=0.8)
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(fontsize=8)

    axes[5].set_title("Total link current")
    axes[5].set_xlabel("Position [nm]")
    axes[5].set_ylabel(r"$J_n+J_p$ [A cm$^{-2}$]")
    axes[5].axhline(0, color="0.2", lw=0.8)
    axes[5].grid(True, alpha=0.3)
    axes[5].legend(fontsize=8)

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def main():
    """Run the overlay using the USER SETTINGS at the top of this file."""
    plot_overlay(SIM_FILES, labels=LABELS, save=SAVE_PATH, show=SHOW_FIGURE, title=FIGURE_TITLE)


if __name__ == "__main__":
    main()
