import os

import numpy as np
import sesame


def build_system(m_qd=2):
    """Build the QD-LED structure with a user-selectable number of discrete QD sites."""
    if m_qd < 1:
        raise ValueError("m_qd must be >= 1")

    t_hil = 30e-7
    t_htl = 25e-7
    t_etl = 20e-7
    t_bqd = t_hil + t_htl

    r_qdc = 4e-7
    r_qds = 1e-7
    r_qd = r_qdc + r_qds

    t_qdl = 2 * m_qd * r_qd
    t_aqd = t_bqd + t_qdl
    t_total = t_aqd + t_etl

    print(
        "m_qd =", m_qd,
        "| t_total =", t_total,
        "| rqd =", r_qd,
        "| taqd =", t_aqd,
        "| tbqd =", t_bqd,
    )

    # Heterojunctions require dense mesh near the interface.
    dd = 4e-7
    dd2 = 1.5e-7
    qd_centers = t_bqd + (2 * np.arange(m_qd) + 1) * r_qd

    x = np.concatenate((
        np.linspace(0, dd, 30, endpoint=False),
        np.linspace(dd, t_hil - dd2, 40, endpoint=False),
        np.linspace(t_hil - dd2, t_hil + dd2, 20, endpoint=False),
        np.linspace(t_hil + dd2, t_bqd - dd, 40, endpoint=False),
        np.linspace(t_bqd - dd, t_bqd, 20, endpoint=False),
        qd_centers,
        np.linspace(t_aqd, t_aqd + dd, 20, endpoint=False),
        np.linspace(t_aqd + dd, t_total - dd, 40, endpoint=False),
        np.linspace(t_total - dd, t_total, 30),
    ))

    sys = sesame.Builder(x)

    qd_mnc, qd_mpc = 0.2, 0.45
    qd_mns, qd_mps = 0.19, 0.60

    hil = {
        'Nc': 2.5e19, 'Nv': 2.5e19, 'Eg': 1.57, 'epsilon': 4, 'Et': 0,
        'mu_e': 0.00322, 'mu_h': 0.00322, 'tau_e': 1.2e-6, 'tau_h': 1.2e-6,
        'affinity': 3.6,
    }
    htl = {
        'Nc': 2.5e19, 'Nv': 2.5e19, 'Eg': 3.0, 'epsilon': 4, 'Et': 0,
        'mu_e': 0.002, 'mu_h': 0.002, 'tau_e': 1.2e-6, 'tau_h': 1.2e-6,
        'affinity': 2.6,
    }
    qdc = {
        'Nc': 2.5e19 * pow(qd_mnc, 1.5),
        'Nv': 2.5e19 * pow(qd_mpc, 1.5),
        'Eg': 2.28, 'epsilon': 7, 'Et': 0,
        'mu_e': 1e-6, 'mu_h': 1e-6, 'tau_e': 1.2e-6, 'tau_h': 1.2e-6,
        'affinity': 3.66, 'Cn': 1e-31, 'Cp': 1e-31, 'B': 0.58e-12,
    }
    etl = {
        'Nc': 2.5e19 * pow(0.24, 1.5),
        'Nv': 2.5e19 * pow(0.59, 1.5),
        'Eg': 3.4, 'epsilon': 7, 'Et': 0,
        'mu_e': 0.002, 'mu_h': 0.002, 'tau_e': 1.2e-6, 'tau_h': 1.2e-6,
        'affinity': 4.0,
    }

    hil_region = lambda x: x <= t_hil
    htl_region = lambda x: np.logical_and(x > t_hil, x <= t_bqd)
    qd_region = lambda x: np.logical_and(t_bqd < x, x < t_aqd)
    etl_region = lambda x: x >= t_aqd

    sys.add_material(etl, etl_region)
    sys.add_material(htl, htl_region)
    sys.add_material(hil, hil_region)
    sys.add_material(qdc, qd_region)

    dEc = etl['affinity'] - qdc['affinity']
    dEv = qdc['affinity'] + qdc['Eg'] - (htl['affinity'] + htl['Eg'])
    print("dEc, dEv =", dEc, dEv, "| delta =", dEc - dEv)

    sys.add_qd(
        1.0,
        qd_mns=qd_mns,
        qd_mps=qd_mps,
        dEc=dEc,
        dEv=dEv,
        location=qd_region,
        r_qd=r_qd,
    )

    sys.add_donor(1e17, etl_region)
    sys.add_acceptor(1e17, htl_region)
    sys.add_acceptor(2e19, hil_region)

    Lcontact_type, Rcontact_type = 'Ohmic', 'Schottky'
    Lcontact_workFcn, Rcontact_workFcn = 4.7, 4.15
    sys.contact_type(Lcontact_type, Rcontact_type, Lcontact_workFcn, Rcontact_workFcn)

    Scontact = 1.16e4  # cm/s
    sys.contact_S(Scontact, Scontact, Scontact, Scontact)

    return sys


def run_iv(m_qd=2, voltages=None, export_root="qd_small_variable_m", maxiter=1000, tol=1e-5):
    if voltages is None:
        voltages = np.linspace(0, 7, 300)

    sys = build_system(m_qd=m_qd)
    export_folder = os.path.join(export_root, f"m{m_qd}")
    os.makedirs(export_folder, exist_ok=True)

    j, l = sesame.IVcurve(
        sys,
        voltages,
        os.path.join(export_folder, "1dQD_V"),
        tol=tol,
        htp=1,
        maxiter=maxiter,
    )
    j = j * sys.scaling.current

    result = {'v': voltages, 'j': j, 'l': l, 'm_qd': m_qd}
    np.save(os.path.join(export_folder, "qd_iv"), result)
    return sys, result


if __name__ == "__main__":
    voltages = np.linspace(0, 7, 300)
    sys, result = run_iv(m_qd=2, voltages=voltages)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(result['v'], result['j'], '-o')
        ax.set_ylim(1e-10, 1)
        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('Current [A/cm$^2$]')
        ax.set_yscale('log')
        ax.grid(True)

        fig, ax = plt.subplots()
        ax.plot(result['v'], result['l'], '-o')
        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('EQE')
        ax.grid(True)

        plt.show()

    except ImportError:
        print("Matplotlib not installed, can't make plot")
