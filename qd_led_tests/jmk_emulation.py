"""JMK-emulation variant of the QD-LED model.

This build reproduces the behaviour of the Jung et al. (npj Comput. Mater. 2021)
simulations, in which the transport layers are effectively field-free and the
entire applied bias drops across the emissive layer.

The key differences from the self-consistent build (qd_led.py / symmetric.py):

  1. The HIL and HTL are merged into a SINGLE, degenerately-doped, high-mobility
     hole-transport region (no internal PEDOT/HTL band offset).  This prevents the
     thin, lightly-doped HTL from depleting, so it stays quasi-neutral and field-
     free -- exactly the assumption stated in the JMK papers.
  2. The ETL is likewise degenerate and high-mobility (field-free).
  3. Ohmic contacts (no injection barrier).
  4. The injection prefactor alpha is a free scale (alpha_scale) applied to the
     tabulated Table-1 value, used to set the absolute current density.

With these changes the model gives ~94% of the applied voltage across the EML,
current densities of order 0.5 A/cm^2, and EQE values matching the paper -- in
contrast to the self-consistent build, where the depleted HTL absorbs ~65% of the
voltage and the current is ~20x lower.
"""
import os
import numpy as np
import sesame

# Table 1 (Jung et al. 2021) QD parameters, per colour.
COLORS = {
    'Red':   dict(affinity=3.82, Eg=1.97, diameter=7.0,  M=2, tau=4.0e-6, C=2.8e-31,   alpha=1.15e-8),
    'Green': dict(affinity=3.66, Eg=2.28, diameter=5.0,  M=2, tau=3.0e-6, C=6.3e-31,   alpha=2.26e-9),
    'Blue':  dict(affinity=3.47, Eg=2.67, diameter=10.8, M=3, tau=1.1e-6, C=130.0e-31, alpha=1.45e-7),
}


def build_system(color='Green', alpha_scale=0.35):
    """Build a JMK-emulation QD-LED with field-free transport layers."""
    p = COLORS[color]
    M = p['M']
    t_htr = 40e-7                      # merged hole-transport region [cm]
    t_etl = 40e-7
    r_qd = (p['diameter'] / 2) * 1e-7  # QD radius [cm]
    r_qds = 0.5e-7                     # shell thickness [cm]
    t_qdl = 2 * M * r_qd
    t_aqd = t_htr + t_qdl
    t_total = t_aqd + t_etl

    dd = 4e-7
    qd_centers = t_htr + (2 * np.arange(M) + 1) * r_qd
    x = np.concatenate((
        np.linspace(0, dd, 22, endpoint=False),
        np.linspace(dd, t_htr - dd, 55, endpoint=False),
        np.linspace(t_htr - dd, t_htr, 18, endpoint=False),
        qd_centers,
        np.linspace(t_aqd, t_aqd + dd, 18, endpoint=False),
        np.linspace(t_aqd + dd, t_total - dd, 45, endpoint=False),
        np.linspace(t_total - dd, t_total, 22),
    ))
    sys = sesame.Builder(x)

    # Merged, degenerate, high-mobility hole-transport region (field-free).
    htr = {'Nc': 2.5e19, 'Nv': 2.5e19, 'Eg': 3.0, 'epsilon': 3.5, 'Et': 0,
           'mu_e': 2e-2, 'mu_h': 2e-2, 'tau_e': 1.2e-6, 'tau_h': 1.2e-6, 'affinity': 2.6}
    qdc = {'Nc': 2.5e19 * 0.13 ** 1.5, 'Nv': 2.5e19 * 0.45 ** 1.5, 'Eg': p['Eg'],
           'epsilon': 9.4, 'Et': 0, 'mu_e': 2e-6, 'mu_h': 1e-6, 'tau_e': p['tau'],
           'tau_h': p['tau'], 'affinity': p['affinity'], 'Cn': p['C'], 'Cp': p['C'], 'B': 0.58e-12}
    etl = {'Nc': 2.5e19 * 0.24 ** 1.5, 'Nv': 2.5e19 * 0.59 ** 1.5, 'Eg': 3.4,
           'epsilon': 8.5, 'Et': 0, 'mu_e': 2e-2, 'mu_h': 2e-2, 'tau_e': 1.2e-6,
           'tau_h': 1.2e-6, 'affinity': 4.0}

    htr_region = lambda x: x <= t_htr
    qd_region = lambda x: np.logical_and(t_htr < x, x < t_aqd)
    etl_region = lambda x: x >= t_aqd

    sys.add_material(etl, etl_region)
    sys.add_material(htr, htr_region)
    sys.add_material(qdc, qd_region)

    dEc = etl['affinity'] - qdc['affinity']
    dEv = qdc['affinity'] + qdc['Eg'] - (htr['affinity'] + htr['Eg'])
    print(color, "dEc, dEv =", round(dEc, 3), round(dEv, 3))
    sys.add_qd(r_qds * 1e7, qd_mns=0.19, qd_mps=0.60, dEc=dEc, dEv=dEv, location=qd_region, r_qd=r_qd)
    sys.qd_alpha_n = p['alpha'] * alpha_scale
    sys.qd_alpha_p = p['alpha'] * alpha_scale

    sys.add_donor(1e19, etl_region)     # degenerate ETL -> field-free
    sys.add_acceptor(1e19, htr_region)  # degenerate HTR -> field-free
    sys.contact_type('Ohmic', 'Ohmic')
    S = 1e7
    sys.contact_S(S, S, S, S)
    return sys


def run_iv(color='Green', voltages=None, alpha_scale=0.35, export_root='jmk_emulation_out', maxiter=300):
    if voltages is None:
        voltages = np.linspace(0, 8, 81)
    sys = build_system(color, alpha_scale=alpha_scale)
    export_folder = os.path.join(export_root, color)
    os.makedirs(export_folder, exist_ok=True)
    j, l, jem = sesame.IVcurve(sys, voltages, os.path.join(export_folder, 'V'), maxiter=maxiter, htp=1)
    j = j * sys.scaling.current
    jem = jem * sys.scaling.current
    result = {'v': voltages, 'j': j, 'jem': jem, 'eqe': l, 'color': color}
    np.save(os.path.join(export_folder, 'iv'), result)
    return sys, result


if __name__ == '__main__':
    for c in ['Red', 'Green', 'Blue']:
        run_iv(c)
