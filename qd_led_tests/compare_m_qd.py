import os

import numpy as np
import matplotlib.pyplot as plt

from symmetric import run_iv


def first_crossing(v, y, threshold):
    """Log-linearly interpolate the first voltage where y crosses threshold."""
    y = np.asarray(y)
    v = np.asarray(v)
    ok = np.isfinite(y) & np.isfinite(v) & (y > 0)
    v = v[ok]
    y = y[ok]
    idx = np.where(y >= threshold)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return v[0]
    y0, y1 = np.log10(y[i - 1]), np.log10(y[i])
    yt = np.log10(threshold)
    return v[i - 1] + (v[i] - v[i - 1]) * (yt - y0) / (y1 - y0)


def plot_current_and_emission(results, export_root):
    fig, ax = plt.subplots(figsize=(4, 3))
    #ax2 = ax.twinx()

    lines = []
    labels = []
    for m_qd, result in results.items():
        line, = ax.plot(result['v'], result['j'], '-', markersize=3, label=f"J, m = {m_qd}")
        lines.append(line)
        labels.append(f"J, m = {m_qd}")
        line2, = ax.plot(result['v'], result['jem'], '--', markersize=3, label=f"Jem, m = {m_qd}")
        lines.append(line2)
        labels.append(f"Jem, m = {m_qd}")

    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('Total current [A/cm$^2$]')
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1)
    ax.set_xlim(2,8)
    ax.grid(True)

    #ax2.set_ylabel('Emissive current [A/cm$^2$]')
    #ax2.set_yscale('log')
    all_jem = np.concatenate([r['jem'][np.isfinite(r['jem']) & (r['jem'] > 0)] for r in results.values()])
    #if len(all_jem) > 0:
    #    ax2.set_ylim(max(np.nanmin(all_jem) / 3, 1e-16), max(np.nanmax(all_jem) * 3, 1e-12))

    ax.legend(lines, labels, loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(export_root, 'current_and_emissive_current_vs_voltage_m_compare.png'), dpi=300)


def plot_eqe(results, export_root):
    fig, ax = plt.subplots(figsize=(4, 3))
    for m_qd, result in results.items():
        ax.plot(result['v'], result['eqe'], '-o', markersize=3, label=f"m = {m_qd}")
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('EQE')
    ax.grid(True)
    ax.set_xlim(2,8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(export_root, 'eqe_vs_voltage_m_compare.png'), dpi=300)


def main():
    voltages = np.linspace(0, 8, 150)
    export_root = 'compare_symmetric_m'
    os.makedirs(export_root, exist_ok=True)

    results = {}
    for m_qd in (1, 2, 3, 4):
        _, result = run_iv(m_qd=m_qd, voltages=voltages, export_root=export_root)
        results[m_qd] = result

    plot_current_and_emission(results, export_root)
    plot_eqe(results, export_root)

    thresholds = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    with open(os.path.join(export_root, 'turn_on_summary.csv'), 'w') as f:
        f.write('m_qd,quantity,threshold_A_per_cm2,voltage_V\n')
        for m_qd, result in results.items():
            for threshold in thresholds:
                f.write(f"{m_qd},total_current,{threshold},{first_crossing(result['v'], result['j'], threshold)}\n")
                f.write(f"{m_qd},emissive_current,{threshold},{first_crossing(result['v'], result['jem'], threshold)}\n")

    plt.show()


if __name__ == '__main__':
    main()
