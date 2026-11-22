"""Device description and finite-volume mesh for the JMK-style QD-LED model.

The model follows Jung et al., npj Comput. Mater. 7, 122 (2021) and
Jo et al., Mater. Today Phys. 46, 101492 (2024):

* Transport layers (HIL/HTL/ETL) are drift-diffusion regions whose drift term
  uses the *electrostatic* field only, F = -dphi/dz.  Band offsets between
  transport layers (e.g. PEDOT:PSS/TFB) therefore do not enter the transport
  equations, and carrier densities are continuous across those interfaces.
  This is what the published hole profile (Jung SI Fig. 3) and the ~0.1 V
  p+/p potential bump at the HIL/HTL interface (Jung SI Fig. 1) require.
* The emissive layer is M discrete QD nodes, one per QD layer, placed at the QD
  centres.  They couple to the HTL/ETL surface nodes only through the
  field-dependent capture current of Jung Eq. (1), and to each other through the
  tunnelling (hopping) current of Jung Eq. (5).  There is no drift-diffusion
  link and no thermal emission between a QD and a transport layer.
* Poisson's equation and all continuity equations use the standard box
  (finite-volume) method on the node set.  For the QD nodes this gives control
  volumes bounded by link midpoints (1.5 r for the outer QDs, 2 r for inner ones).
  With these volumes the Fig. 2d recombination rates reproduce the published
  J(10 V) = 0.83 A/cm^2 and L(10 V) = 1.27e5 cd/m^2 (M = 2), which is how
  we know the original code used them.

Units: cm, s, V, cm^-3 throughout.
"""
from dataclasses import dataclass, field
import numpy as np

Q = 1.602176634e-19        # C
EPS0 = 8.8541878128e-14    # F/cm
KB = 1.380649e-23          # J/K
H_PLANCK = 6.62607015e-34  # J s
HBAR = H_PLANCK / (2 * np.pi)
M0 = 9.1093837015e-31      # kg

# link kinds
DD, INJ_H, HOP, INJ_E = 0, 1, 2, 3


def eff_dos(mass_ratio, T=300.0):
    """Effective density of states [cm^-3] for an effective-mass ratio."""
    return 2.0 * (2 * np.pi * mass_ratio * M0 * KB * T / H_PLANCK ** 2) ** 1.5 * 1e-6


@dataclass
class Layer:
    """A drift-diffusion transport layer."""
    name: str
    thickness: float           # cm
    eps: float                 # relative permittivity
    mu_n: float                # cm^2/Vs
    mu_p: float                # cm^2/Vs
    Nd: float = 0.0            # cm^-3
    Na: float = 0.0            # cm^-3
    affinity: float = 4.0      # eV (positive, below vacuum)
    Eg: float = 1.0            # eV
    me: float = 1.0            # effective-mass ratios (only enter Nc, Nv -> ni, contacts)
    mh: float = 1.0
    tau_n: float = 1.2e-6      # s
    tau_p: float = 1.2e-6      # s
    spacing: float = 1e-7      # target mesh spacing, cm


@dataclass
class QDLayer:
    """The emissive layer: M monolayers of core/shell QDs."""
    M: int = 2
    d: float = 5e-7            # QD diameter (core + shell), cm
    eps: float = 9.4
    t_shell: float = 0.5e-7    # cm
    LUMO: float = -3.66        # eV (QD core conduction level)
    Eg: float = 2.28           # eV (optical gap)
    shell_Ec: float = -3.28    # eV
    shell_Ev: float = -6.82    # eV
    me_core: float = 0.13
    mh_core: float = 0.45
    me_shell: float = 0.19
    mh_shell: float = 0.60
    mu_n0: float = 2e-6        # QD mobilities (enter only the Langevin gamma, if computed)
    mu_p0: float = 1e-6
    tau_n: float = 1.5e-6
    tau_p: float = 1.5e-6
    B: float = 0.58e-12        # Langevin radiative coefficient, cm^3/s
    Cn: float = 1e-30          # Auger, cm^6/s
    Cp: float = 1e-30
    alpha_p: float = 3e-9      # charge-injection mobility sigma*Tb*mu0, cm^2/Vs
    alpha_n: float = 3e-9
    F0: float = 5e6            # Poole-Frenkel reference field, V/cm
    nu_n: float = None         # hopping frequencies [1/s]; None -> Jung SI Eqs. (1)-(2)
    nu_p: float = None
    hop_barrier_thickness: float = None  # shell thickness crossed between two QDs (default 2*t_shell)
    wavelength: float = None   # nm; None -> hc/Eg

    @property
    def r(self):
        return self.d / 2

    @property
    def N_QD(self):
        return 1.0 / (4.0 / 3.0 * np.pi * self.r ** 3)

    @property
    def HOMO(self):
        return self.LUMO - self.Eg

    def hopping_frequencies(self, T=300.0):
        """Tunnelling frequencies nu = v_th/(2 d) * T_bs (Jung SI Eqs. 1-2)."""
        ts = (2 * self.t_shell if self.hop_barrier_thickness is None
              else self.hop_barrier_thickness) * 1e-2  # m

        def tb(m_shell, dE):
            if dE < 0:
                return 1.0
            kappa = np.sqrt(2 * m_shell * M0 * dE * Q) / HBAR
            return np.exp(-2 * kappa * ts)

        vth_n = np.sqrt(2 * KB * T / (self.me_core * M0)) * 100  # cm/s
        vth_p = np.sqrt(2 * KB * T / (self.mh_core * M0)) * 100
        nu_n = vth_n / (2 * self.d) * tb(self.me_shell, self.shell_Ec - self.LUMO)
        nu_p = vth_p / (2 * self.d) * tb(self.mh_shell, self.HOMO - self.shell_Ev)
        if self.nu_n is not None:
            nu_n = self.nu_n
        if self.nu_p is not None:
            nu_p = self.nu_p
        return nu_n, nu_p


@dataclass
class Device:
    """HIL / HTL / [QD x M] / ETL stack between an anode and a cathode."""
    hil: Layer
    htl: Layer
    qd: QDLayer
    etl: Layer
    wf_anode: float = 4.70          # eV
    wf_cathode: float = 4.06        # eV
    Vbi: float = None               # V; None -> (wf_anode - wf_cathode)
    T: float = 300.0
    LEE: float = 0.20
    fwhm_nm: float = 30.0
    # contact carrier densities; None -> defaults set in build()
    p_anode: float = None
    n_anode: float = None
    n_cathode: float = None
    p_cathode: float = None
    dEv: float = None               # override HTL->QD valence offset (eV)
    dEc: float = None               # override ETL->QD conduction offset (eV)
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def VT(self):
        return KB * self.T / Q

    @property
    def offsets(self):
        """(dEv, dEc) in eV, Jung definitions dEv = Ev(HTL)-Ev(QD), dEc = Ec(QD)-Ec(ETL)."""
        ev_htl = -(self.htl.affinity + self.htl.Eg)
        ec_etl = -self.etl.affinity
        dEv = ev_htl - self.qd.HOMO if self.dEv is None else self.dEv
        dEc = self.qd.LUMO - ec_etl if self.dEc is None else self.dEc
        return dEv, dEc

    def ni(self, layer):
        Nc, Nv = eff_dos(layer.me, self.T), eff_dos(layer.mh, self.T)
        return np.sqrt(Nc * Nv) * np.exp(-layer.Eg / (2 * self.VT))

    # ------------------------------------------------------------------ #
    def build(self):
        """Create the node/link arrays used by the solver."""
        qd = self.qd
        r = qd.r
        VT = self.VT

        def grid(z0, t, h):
            n = max(int(round(t / h)), 1)
            return z0 + np.linspace(0.0, t, n + 1)

        z_hil = grid(0.0, self.hil.thickness, self.hil.spacing)
        z_htl = grid(self.hil.thickness, self.htl.thickness, self.htl.spacing)
        zH = self.hil.thickness + self.htl.thickness
        zq = zH + r * (2 * np.arange(qd.M) + 1)
        zE = zH + 2 * r * qd.M
        z_etl = grid(zE, self.etl.thickness, self.etl.spacing)
        z = np.concatenate([z_hil, z_htl[1:], zq, z_etl])
        N = len(z)
        h = np.diff(z)
        L = N - 1  # number of links

        iH = len(z_hil) + len(z_htl) - 2          # HTL surface node
        iQ = np.arange(iH + 1, iH + 1 + qd.M)     # QD nodes
        iE = iQ[-1] + 1                           # ETL surface node

        link_kind = np.full(L, DD, dtype=int)
        link_kind[iH] = INJ_H
        link_kind[iQ[:-1]] = HOP
        link_kind[iQ[-1]] = INJ_E

        # link materials
        mid = 0.5 * (z[1:] + z[:-1])
        eps = np.empty(L); mun = np.empty(L); mup = np.empty(L); dop = np.empty(L)
        tau_n = np.empty(L); tau_p = np.empty(L); B = np.zeros(L); Cn = np.zeros(L); Cp = np.zeros(L)
        ni = np.empty(L)
        for j in range(L):
            if link_kind[j] != DD:
                eps[j] = qd.eps; mun[j] = 0.0; mup[j] = 0.0; dop[j] = 0.0
                tau_n[j], tau_p[j], B[j], Cn[j], Cp[j] = qd.tau_n, qd.tau_p, qd.B, qd.Cn, qd.Cp
                Ncq, Nvq = eff_dos(qd.me_core, self.T), eff_dos(qd.mh_core, self.T)
                ni[j] = np.sqrt(Ncq * Nvq) * np.exp(-qd.Eg / (2 * VT))
                continue
            if mid[j] < self.hil.thickness:
                lay = self.hil
            elif mid[j] < zH:
                lay = self.htl
            else:
                lay = self.etl
            eps[j] = lay.eps; mun[j] = lay.mu_n; mup[j] = lay.mu_p; dop[j] = lay.Nd - lay.Na
            tau_n[j], tau_p[j] = lay.tau_n, lay.tau_p
            ni[j] = self.ni(lay)

        # box widths and per-node doping charge (cm^-2)
        w = np.zeros(N)
        w[:-1] += 0.5 * h
        w[1:] += 0.5 * h
        ndop = np.zeros(N)
        ndop[:-1] += 0.5 * h * dop
        ndop[1:] += 0.5 * h * dop

        # band parameters per node (only used by the diagnostic Sesame-leak option)
        Ec0 = np.empty(N); Ev0 = np.empty(N); Nc = np.empty(N); Nv = np.empty(N)
        for i, zz in enumerate(z):
            if i in iQ:
                Ec0[i], Ev0[i] = qd.LUMO, qd.HOMO
                Nc[i], Nv[i] = eff_dos(qd.me_core, self.T), eff_dos(qd.mh_core, self.T)
                continue
            lay = self.hil if zz < self.hil.thickness - 1e-12 else (self.htl if zz <= zH + 1e-12 else self.etl)
            Ec0[i], Ev0[i] = -lay.affinity, -(lay.affinity + lay.Eg)
            Nc[i], Nv[i] = eff_dos(lay.me, self.T), eff_dos(lay.mh, self.T)
        if self.dEv is not None:
            Ev0[iQ] = Ev0[iH] - self.dEv
        if self.dEc is not None:
            Ec0[iQ] = Ec0[iE] + self.dEc

        # recombination volume of each half-cell.  The EML-side halves of the
        # HTL/ETL surface boxes store charge (Poisson, continuity) but their
        # carriers are transport-layer carriers, so they get no QD recombination.
        rec_wL = 0.5 * h.copy()
        rec_wR = 0.5 * h.copy()
        rec_wL[iH] = 0.0
        rec_wR[iQ[-1]] = 0.0

        # QD hopping and injection prefactors
        nu_n, nu_p = qd.hopping_frequencies(self.T)
        dEv, dEc = self.offsets
        kap = np.pi * r ** 3 / np.sqrt(qd.F0)   # r * pi r^2 / F0^1/2

        # contacts
        hil, etl = self.hil, self.etl
        ni_a, ni_c = self.ni(hil), self.ni(etl)
        Vbi = (self.wf_anode - self.wf_cathode) if self.Vbi is None else self.Vbi
        # Jung Eq. (10) (sign-corrected): n_s = ni exp(-(phi_m - chi - Eg/2)/kT)
        n_a_eq10 = ni_a * np.exp(-(self.wf_anode - hil.affinity - hil.Eg / 2) / VT)
        n_c_eq10 = ni_c * np.exp(-(self.wf_cathode - etl.affinity - etl.Eg / 2) / VT)
        p_a = hil.Na if self.p_anode is None else self.p_anode
        n_a = n_a_eq10 if self.n_anode is None else self.n_anode
        n_c = n_c_eq10 if self.n_cathode is None else self.n_cathode
        p_c = ni_c ** 2 / n_c if self.p_cathode is None else self.p_cathode

        return Mesh(z=z, h=h, w=w, ndop=ndop, eps=eps, mun=mun, mup=mup,
                    tau_n=tau_n, tau_p=tau_p, B=B, Cn=Cn, Cp=Cp, ni=ni,
                    rec_wL=rec_wL, rec_wR=rec_wR, Ec0=Ec0, Ev0=Ev0, Nc=Nc, Nv=Nv,
                    mu_qd_n=qd.mu_n0, mu_qd_p=qd.mu_p0,
                    link_kind=link_kind, iH=iH, iQ=iQ, iE=iE, r=r, N_QD=qd.N_QD,
                    d=qd.d, nu_n=nu_n, nu_p=nu_p, kap_p=kap * qd.alpha_p,
                    kap_n=kap * qd.alpha_n, dEv=dEv, dEc=dEc, VT=VT, Vbi=Vbi,
                    p_a=p_a, n_a=n_a, n_c=n_c, p_c=p_c)


@dataclass
class Mesh:
    z: np.ndarray
    h: np.ndarray
    w: np.ndarray
    ndop: np.ndarray
    eps: np.ndarray
    mun: np.ndarray
    mup: np.ndarray
    tau_n: np.ndarray
    tau_p: np.ndarray
    B: np.ndarray
    Cn: np.ndarray
    Cp: np.ndarray
    ni: np.ndarray
    rec_wL: np.ndarray
    rec_wR: np.ndarray
    Ec0: np.ndarray
    Ev0: np.ndarray
    Nc: np.ndarray
    Nv: np.ndarray
    mu_qd_n: float
    mu_qd_p: float
    link_kind: np.ndarray
    iH: int
    iQ: np.ndarray
    iE: int
    r: float
    N_QD: float
    d: float
    nu_n: float
    nu_p: float
    kap_p: float
    kap_n: float
    dEv: float
    dEc: float
    VT: float
    Vbi: float
    p_a: float
    n_a: float
    n_c: float
    p_c: float

    @property
    def N(self):
        return len(self.z)
