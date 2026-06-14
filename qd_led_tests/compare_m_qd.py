import os
import numpy as np
import matplotlib.pyplot as plt

from symmetric import run_iv


def first_crossing(v, y, threshold):
    """Linearly interpolate the first voltage where y crosses threshold."""
    y = np.asarray(y)
    ok = np.isfinite(y)
    v = np.asarray(v)[ok]
    y = y[ok]
    idx = np.where(y >= threshold)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return v[0]
    # Interpolate in log-current because the traces are plotted semilog.
    y0, y1 = np.log10(y[i - 1]), np.log10(y[i])
    yt = np.log10(threshold)
    return v[i - 1] + (v[i] - v[i - 1]) * (yt - y0) / (y1 - y0)


def main():
    voltages = np.linspace(0, 6, 100)
    export_root = "compare_m_qd"
    os.makedirs(export_root, exist_ok=True)

    results = {}
    for m_qd in (1, 2, 3):
        _, result = run_iv(m_qd=m_qd, voltages=voltages, export_root=export_root)
        results[m_qd] = result

    fig, ax = plt.subplots()
    for m_qd, result in results.items():
        ax.plot(result['v'], result['j'], '-o', markersize=3, label=f"m = {m_qd}")
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('Current [A/cm$^2$]')
    ax.set_yscale('log')
    ax.set_ylim(1e-12, 1)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(export_root, "current_vs_voltage_m_compare.png"), dpi=300)

    fig, ax = plt.subplots()
    for m_qd, result in results.items():
        ax.plot(result['v'], result['l'], '-o', markersize=3, label=f"m = {m_qd}")
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('EQE / emission proxy')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(export_root, "emission_vs_voltage_m_compare.png"), dpi=300)

    thresholds = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    with open(os.path.join(export_root, "turn_on_summary.csv"), "w") as f:
        f.write("m_qd,threshold_A_per_cm2,voltage_V\n")
        for m_qd, result in results.items():
            for threshold in thresholds:
                f.write(f"{m_qd},{threshold},{first_crossing(result['v'], result['j'], threshold)}\n")

    plt.show()


if __name__ == "__main__":
    main()
