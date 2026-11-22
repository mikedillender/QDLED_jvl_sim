"""Parameter sets from Jung et al. (2021) and Jo et al. (2024)."""
from .device import Device, Layer, QDLayer


def jung2021(M=2, d_nm=5.0, LUMO=-3.66, Eg=2.28, tau=1.5e-6, C=1e-30,
             alpha_p=3e-9, alpha_n=3e-9, dEv=None, dEc=None, wavelength=545.0,
             hil_spacing=0.5e-7, tl_spacing=1e-7, **dev_kw):
    """Jung et al., npj Comput. Mater. 7, 122 (2021), Supplementary Tables 1-2.

    Transport-layer minority mobilities are not given in the paper; they only
    matter for carriers the model never injects, so they are set equal to the
    majority mobility of the same layer."""
    hil = Layer('PEDOT:PSS', 20e-7, eps=3.0, mu_n=0.322e-3, mu_p=0.322e-3, Na=2.81e19,
                affinity=3.6, Eg=1.57, me=1.0, mh=1.0, spacing=hil_spacing)
    htl = Layer('TFB', 20e-7, eps=3.5, mu_n=2.0e-3, mu_p=2.0e-3, Na=1.0e17,
                affinity=2.6, Eg=3.0, me=1.0, mh=1.0, spacing=tl_spacing)
    etl = Layer('ZnO', 40e-7, eps=8.5, mu_n=2.0e-3, mu_p=2.0e-3, Nd=1.0e17,
                affinity=4.0, Eg=3.4, me=0.24, mh=0.59, spacing=tl_spacing)
    qd = QDLayer(M=M, d=d_nm * 1e-7, eps=9.4, t_shell=0.5e-7, LUMO=LUMO, Eg=Eg,
                 tau_n=tau, tau_p=tau, Cn=C, Cp=C, B=0.58e-12,
                 alpha_p=alpha_p, alpha_n=alpha_n, wavelength=wavelength)
    return Device(hil=hil, htl=htl, qd=qd, etl=etl, wf_anode=4.70, wf_cathode=4.06,
                  Vbi=0.64, LEE=0.20, dEv=dEv, dEc=dEc, **dev_kw)


# Jung 2021 Table 1 (Fig. 5 emulations)
JUNG_TABLE1 = {
    'red':   dict(LUMO=-3.82, Eg=1.97, d_nm=7.0, M=2, tau=4.0e-6, C=2.8e-31, alpha=1.15e-8, wavelength=628.0),
    'green': dict(LUMO=-3.66, Eg=2.28, d_nm=5.0, M=2, tau=3.0e-6, C=6.3e-31, alpha=2.26e-9, wavelength=545.0),
    'blue':  dict(LUMO=-3.47, Eg=2.67, d_nm=10.8, M=3, tau=1.1e-6, C=130e-31, alpha=1.45e-7, wavelength=465.0),
}


def jung2021_table1(color, **kw):
    p = dict(JUNG_TABLE1[color])
    a = p.pop('alpha')
    p.update(kw)
    return jung2021(alpha_p=a, alpha_n=a, **p)


def jo2024(delta=0.0, A=1e3, B=1e-14, C=1e-32, S=0.5, T_shell=1.0, mu_tl=2e-3, **dev_kw):
    """Jo et al., Mater. Today Phys. 46, 101492 (2024), Tables S1-S2.

    UNVALIDATED starting point (not compared against that paper's figures).
    The paper gives A = 1/(tau_p + tau_n) (tau_p = tau_n = 1/(2A)) and C = Cp + Cn.
    Its capture prefactor q r S sigma T mu E^1.5 is printed without the F0^1/2 of
    Jung 2021 and without a value for the shell tunnelling factor T; here it is
    taken in the Jung 2021 form with alpha = S * T_shell * mu_QD and F0 = 5 MV/cm,
    so the injection magnitude (T_shell) is uncalibrated.  delta = dEv - dEc."""
    hil = Layer('PEDOT:PSS', 20e-7, eps=3.0, mu_n=3.2e-4, mu_p=3.2e-4, Na=2.8e19,
                affinity=3.6, Eg=1.57, spacing=1e-7)
    htl = Layer('TFB', 40e-7, eps=5.0, mu_n=mu_tl, mu_p=mu_tl, Na=1.0e17,
                affinity=2.6, Eg=3.0, spacing=1e-7)
    etl = Layer('ZnO', 40e-7, eps=5.0, mu_n=mu_tl, mu_p=mu_tl, Nd=1.0e17,
                affinity=4.0, Eg=3.4, me=0.24, mh=0.59, spacing=1e-7)
    tau = 1.0 / (2 * A)
    qd = QDLayer(M=2, d=7.5e-7, eps=9.4, t_shell=1.0e-7, LUMO=-3.63, Eg=2.34,
                 tau_n=tau, tau_p=tau, B=B, Cn=C / 2, Cp=C / 2,
                 alpha_p=S * T_shell * 1e-6, alpha_n=S * T_shell * 1e-6, F0=5e6, wavelength=530.0)
    base = 0.37  # symmetric offset scale; only delta matters for the balance study
    dEv = base + delta / 2
    dEc = base - delta / 2
    return Device(hil=hil, htl=htl, qd=qd, etl=etl, wf_anode=4.7, wf_cathode=4.0,
                  Vbi=0.7, LEE=0.20, dEv=dEv, dEc=dEc, **dev_kw)
