"""Fully implicit (backward-Euler) transient solver for the JMK QD-LED model.

Unknowns per node: electrostatic potential phi [V], u = ln n, v = ln p.
Log-densities keep every carrier density positive and let the Newton solver
span the 1e-40 .. 1e21 cm^-3 range that appears in these stacks.

The voltage is applied as a staircase ("stepped transient"): at each step the
anode potential jumps to V - Vbi and the full Poisson / continuity system is
integrated in time from the *previous* state, either for a fixed dwell time or
until it stops changing.  Because the QD nodes only exchange carriers with the
transport layers by one-way capture (no emission), the QD occupation is a
conserved quantity whenever injection is off, so the reached state depends on
the voltage history -- this is the history dependence that a direct
Newton steady-state solve (Sesame) cannot represent.
"""
import numpy as np
from scipy.linalg import solve_banded

from .device import Q, EPS0, DD, INJ_H, HOP, INJ_E


def bern(x):
    """Bernoulli function B(x) = x / (exp(x) - 1), evaluated stably."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-5
    xs = x[small]
    out[small] = 1.0 - 0.5 * xs + xs * xs / 12.0
    xl = x[~small]
    with np.errstate(over='ignore'):
        out[~small] = xl / np.expm1(xl)
    return out


class ConvergenceError(RuntimeError):
    pass


class TransientSolver:
    """Stepped-transient drift-diffusion / QD rate-equation solver.

    Parameters
    ----------
    device : qdtransient.device.Device
    gate_injection : bool
        If True, carriers are injected into the QDs only when *both* interface
        fields exceed their band-offset thresholds.  This is NOT physical; it is
        a diagnostic switch used to test how Jung et al. obtained a
        max(dEc, dEv)-controlled threshold in their Fig. 4j-l.
    sesame_leak : float or None
        Diagnostic.  Adds, on every EML link, the band-aware Scharfetter-Gummel
        current that the Sesame reconstruction kept after dividing it by 1e10
        (mobility = sesame_leak * QD mobility).  It is negligible on any
        experimental time scale but fixes the infinite-time steady state.
    """

    def __init__(self, device, gate_injection=False, gate_width=0.02e6, field_smooth=1e3,
                 sesame_leak=None):
        self.dev = device
        self.m = device.build()
        self.N = self.m.N
        self.gate_injection = gate_injection
        self.gate_width = gate_width      # V/cm smoothing of the (diagnostic) gate
        # max(F,0)^1.5 has an infinite second derivative at F = 0, and a QD layer
        # that is charging up sits exactly there (injection self-limits by
        # screening its own field).  A softplus of width field_smooth (V/cm,
        # ~1e-3 of the operating fields) removes the kink without changing results.
        self.field_smooth = field_smooth
        self.sesame_leak = sesame_leak
        self.fd_step = np.array([1e-7 * self.m.VT, 1e-7, 1e-7])
        k = self.m.link_kind
        self._dd = k == DD
        self._hop = k == HOP
        self.stats = dict(newton=0, steps=0, rejects=0)

    # ------------------------------------------------------------------ #
    # physics
    # ------------------------------------------------------------------ #
    def recomb_terms(self, n, p, side):
        """Per-half-cell (SRH, radiative, Auger) rates using link materials.

        side = 'L' uses node j of link j, 'R' uses node j+1."""
        m = self.m
        if side == 'L':
            nn, pp = n[:-1], p[:-1]
        else:
            nn, pp = n[1:], p[1:]
        ex = nn * pp - m.ni ** 2
        srh = ex / (m.tau_n * (pp + m.ni) + m.tau_p * (nn + m.ni))
        rad = m.B * ex
        aug = (m.Cn * nn + m.Cp * pp) * ex
        return srh, rad, aug

    def interface_fields(self, phi):
        """Effective capture fields (V/cm) at the HTL/QD and QD/ETL interfaces."""
        m = self.m
        jh, je = m.iH, m.iQ[-1]
        Fp = (-(phi[jh + 1] - phi[jh]) - m.dEv) / m.r
        Fn = (-(phi[je + 1] - phi[je]) - m.dEc) / m.r
        return Fp, Fn

    def fluxes(self, phi, n, p):
        """Particle fluxes in +z on every link: (Phi_n, Phi_p) [cm^-2 s^-1]."""
        m = self.m
        dphi = phi[1:] - phi[:-1]
        psi = dphi / m.VT
        Bp, Bm = bern(psi), bern(-psi)
        Fp = np.zeros(self.N - 1)
        Fn = np.zeros(self.N - 1)
        dd = self._dd
        cp = m.mup * m.VT / m.h
        cn = m.mun * m.VT / m.h
        Fp[dd] = (cp * (p[:-1] * Bp - p[1:] * Bm))[dd]
        Fn[dd] = (cn * (n[:-1] * Bm - n[1:] * Bp))[dd]
        hop = self._hop
        Fp[hop] = (m.nu_p * m.d * (p[:-1] - p[1:]))[hop]
        Fn[hop] = (m.nu_n * m.d * (n[:-1] - n[1:]))[hop]
        # field-dependent capture (Jung Eq. 1); no emission back to the transport layers
        Ep, En = self.interface_fields(phi)
        s = self.field_smooth
        gp = (s * np.logaddexp(0.0, Ep / s)) ** 1.5
        gn = (s * np.logaddexp(0.0, En / s)) ** 1.5
        if self.gate_injection:
            s = self.gate_width
            gp *= 0.5 * (1 + np.tanh(En / s))
            gn *= 0.5 * (1 + np.tanh(Ep / s))
        jh, je = m.iH, m.iQ[-1]
        Fp[jh] = m.kap_p * gp * p[jh] * (m.N_QD - p[jh + 1])
        Fn[je] = -m.kap_n * gn * n[je + 1] * (m.N_QD - n[je])
        if self.sesame_leak:
            eml = ~dd
            Dp = (dphi + (m.Ev0[:-1] - m.Ev0[1:])) / m.VT - np.log(m.Nv[1:] / m.Nv[:-1])
            Dn = (dphi - (m.Ec0[1:] - m.Ec0[:-1])) / m.VT + np.log(m.Nc[1:] / m.Nc[:-1])
            lp = self.sesame_leak * m.mu_qd_p * m.VT / m.h
            ln_ = self.sesame_leak * m.mu_qd_n * m.VT / m.h
            Fp[eml] += (lp * (p[:-1] * bern(Dp) - p[1:] * bern(-Dp)))[eml]
            Fn[eml] += (ln_ * (n[:-1] * bern(-Dn) - n[1:] * bern(Dn)))[eml]
        return Fn, Fp

    def residual(self, X, Xold, dt, phi_a):
        m = self.m
        phi, u, v = X[:, 0], X[:, 1], X[:, 2]
        n, p = np.exp(u), np.exp(v)
        R = np.empty_like(X)
        # Poisson (box method)
        D = m.eps * (phi[1:] - phi[:-1]) / m.h
        R[1:-1, 0] = D[1:] - D[:-1] + (Q / EPS0) * (m.w[1:-1] * (p[1:-1] - n[1:-1]) + m.ndop[1:-1])
        # continuity
        Fn, Fp = self.fluxes(phi, n, p)
        sL, rL, aL = self.recomb_terms(n, p, 'L')
        sR, rR, aR = self.recomb_terms(n, p, 'R')
        Uw = np.zeros(self.N)
        Uw[:-1] += (sL + rL + aL) * m.rec_wL
        Uw[1:] += (sR + rR + aR) * m.rec_wR
        if np.isfinite(dt):
            nold, pold = np.exp(Xold[:, 1]), np.exp(Xold[:, 2])
            sn = m.w * (n - nold) / dt
            sp = m.w * (p - pold) / dt
        else:
            sn = sp = np.zeros(self.N)
        R[1:-1, 1] = sn[1:-1] + Fn[1:] - Fn[:-1] + Uw[1:-1]
        R[1:-1, 2] = sp[1:-1] + Fp[1:] - Fp[:-1] + Uw[1:-1]
        # Dirichlet contacts: anode phi = V - Vbi, cathode phi = 0
        R[0] = [phi[0] - phi_a, u[0] - np.log(m.n_a), v[0] - np.log(m.p_a)]
        R[-1] = [phi[-1], u[-1] - np.log(m.n_c), v[-1] - np.log(m.p_c)]
        return R

    # ------------------------------------------------------------------ #
    # linear algebra
    # ------------------------------------------------------------------ #
    def jacobian(self, X, R0, Xold, dt, phi_a):
        """Banded Jacobian (5 sub/super diagonals) by colored finite differences."""
        N = self.N
        nv = 3 * N
        ab = np.zeros((11, nv))
        r0 = R0.ravel()
        for c in range(3):
            dlt = self.fd_step[c]
            for k in range(3):
                nodes = np.arange(k, N, 3)
                Xp = X.copy()
                Xp[nodes, c] += dlt
                dR = (self.residual(Xp, Xold, dt, phi_a).ravel() - r0) / dlt
                cols = 3 * nodes + c
                for off in (-1, 0, 1):
                    rn = nodes + off
                    ok = (rn >= 0) & (rn < N)
                    for rc in range(3):
                        rows = 3 * rn[ok] + rc
                        cc = cols[ok]
                        ab[5 + rows - cc, cc] = dR[rows]
        return ab

    @staticmethod
    def _row_scale(ab, rhs):
        nv = ab.shape[1]
        rmax = np.zeros(nv)
        for d in range(-5, 6):  # entry (r, r+d) stored at ab[5-d, r+d]
            row = ab[5 - d]
            r = np.arange(nv) - d
            ok = (r >= 0) & (r < nv)
            np.maximum.at(rmax, r[ok], np.abs(row[ok]))
        rmax[rmax == 0] = 1.0
        for d in range(-5, 6):
            r = np.arange(nv) - d
            ok = (r >= 0) & (r < nv)
            ab[5 - d, ok] /= rmax[r[ok]]
        return ab, rhs / rmax

    def newton(self, X0, Xold, dt, phi_a, tol=1e-9, maxit=30, maxstep=8.0):
        X = X0.copy()
        VT = self.m.VT
        scale = np.array([1.0 / VT, 1.0, 1.0])
        for it in range(1, maxit + 1):
            R = self.residual(X, Xold, dt, phi_a)
            ab = self.jacobian(X, R, Xold, dt, phi_a)
            ab, rhs = self._row_scale(ab, -R.ravel())
            try:
                dx = solve_banded((5, 5), ab, rhs, check_finite=False)
            except (np.linalg.LinAlgError, ValueError):
                return X, False, it
            dx = dx.reshape(-1, 3)
            if not np.all(np.isfinite(dx)):
                return X, False, it
            mx = np.max(np.abs(dx) * scale)
            # per-component clipping (log-densities by maxstep, phi by maxstep*VT) so a
            # single variable far from its solution cannot freeze all the others
            lim = maxstep / scale
            dx = np.clip(dx, -lim, lim)
            X = X + dx
            self.stats['newton'] += 1
            if mx < tol:
                return X, True, it
            if not np.all(np.isfinite(X)) or np.max(np.abs(X[:, 1:])) > 700:
                return X0, False, it
        return X, False, maxit

    # ------------------------------------------------------------------ #
    # time integration
    # ------------------------------------------------------------------ #
    def advance(self, X, V, t_end, dt0=1e-12, dt_max=None, steady_tol=None,
                t_min=0.0, dtmin=1e-20):
        """Integrate from state X at constant bias V for up to t_end seconds.

        If steady_tol is given, stop early once a step of length >= t/2
        changes no log-density by more than steady_tol and potential by more
        than steady_tol*VT (i.e. the state is stationary on the current time
        scale).  Returns (X, t_reached, dt_last, Xprev)."""
        phi_a = V - self.m.Vbi
        t, dt = 0.0, dt0
        Xprev = X
        if dt_max is None:
            dt_max = t_end
        while t < t_end * (1 - 1e-12):
            dt = min(dt, t_end - t, dt_max)
            Xn, ok, its = self.newton(X, X, dt, phi_a)
            if not ok:
                self.stats['rejects'] += 1
                dt /= 4.0
                if dt < dtmin:
                    raise ConvergenceError(f'time step collapsed at V={V:.3f}, t={t:.3e}')
                continue
            # stationarity measure: potential everywhere, log-densities only where the
            # density is physically meaningful (> 1 cm^-3; tiny minority densities in
            # the transport layers carry round-off noise but no current)
            dX = np.abs(Xn - X)
            dX[:, 0] /= self.m.VT
            dX[:, 1] *= Xn[:, 1] > 0.0
            dX[:, 2] *= Xn[:, 2] > 0.0
            change = np.max(dX)
            Xprev, X = X, Xn
            t += dt
            self.stats['steps'] += 1
            if steady_tol is not None and t >= t_min and dt >= 0.5 * t and change < steady_tol:
                break
            if its <= 4:
                dt *= 2.0
            elif its > 10:
                dt *= 0.5
        return X, t, dt, Xprev

    # ------------------------------------------------------------------ #
    def initial_state(self, qd_density=1.0):
        """Charge-neutral initial guess, then relaxed at V = 0."""
        m, dev = self.m, self.dev
        z = m.z
        N = self.N
        X = np.zeros((N, 3))
        zH = dev.hil.thickness + dev.htl.thickness
        Na = np.where(z <= dev.hil.thickness, dev.hil.Na, dev.htl.Na)
        p = np.where(z <= zH + 1e-12, Na, 1.0)
        n = np.where(z >= m.z[m.iE] - 1e-12, dev.etl.Nd, 1.0)
        p[m.iQ] = qd_density
        n[m.iQ] = qd_density
        # minority carriers in the transport layers
        n = np.where(z <= zH + 1e-12, np.maximum(m.n_a, 1e-30), n)
        p = np.where(z >= m.z[m.iE] - 1e-12, np.maximum(m.p_c, 1e-60), p)
        X[:, 0] = np.interp(z, [0, z[-1]], [-m.Vbi, 0.0])
        X[:, 1] = np.log(n)
        X[:, 2] = np.log(p)
        X, *_ = self.advance(X, 0.0, 1.0, dt0=1e-15, steady_tol=1e-10, t_min=1e-6)
        return X

    # ------------------------------------------------------------------ #
    # observables
    # ------------------------------------------------------------------ #
    def observe(self, X, Xprev=None, dt=None, V=None):
        m, dev = self.m, self.dev
        phi, n, p = X[:, 0], np.exp(X[:, 1]), np.exp(X[:, 2])
        Fn, Fp = self.fluxes(phi, n, p)
        Jc = Q * (Fp - Fn)                       # conduction current per link [A/cm^2]
        E = -(phi[1:] - phi[:-1]) / m.h
        if Xprev is not None and dt is not None and np.isfinite(dt):
            Eo = -(Xprev[1:, 0] - Xprev[:-1, 0]) / m.h
            Jd = EPS0 * m.eps * (E - Eo) / dt
        else:
            Jd = np.zeros_like(Jc)
        Jt = Jc + Jd
        sL, rL, aL = self.recomb_terms(n, p, 'L')
        sR, rR, aR = self.recomb_terms(n, p, 'R')
        wL, wR = m.rec_wL, m.rec_wR
        srh = np.zeros(self.N); rad = np.zeros(self.N); aug = np.zeros(self.N)
        srh[:-1] += sL * wL; srh[1:] += sR * wR
        rad[:-1] += rL * wL; rad[1:] += rR * wR
        aug[:-1] += aL * wL; aug[1:] += aR * wR
        iQ = m.iQ
        R_rad, R_srh, R_aug = rad.sum(), srh.sum(), aug.sum()
        R_tot = R_rad + R_srh + R_aug
        # below turn-on all rates are ~1e-10 cm^-2 s^-1 numerical residue; report 0
        eqe = dev.LEE * R_rad / R_tot if R_tot > 1e6 else 0.0
        Ep, En = self.interface_fields(phi)
        out = dict(
            V=V, J=float(np.median(Jt)), J_cond_anode=float(Jc[0]), J_cond_cathode=float(Jc[-1]),
            R_rad=R_rad, R_srh=R_srh, R_aug=R_aug, R_tot=R_tot,
            R_rad_qd=rad[iQ].sum(), R_srh_qd=srh[iQ].sum(), R_aug_qd=aug[iQ].sum(),
            EQE=eqe,
            EQE_current=dev.LEE * Q * R_rad / abs(np.median(Jt)) if abs(np.median(Jt)) > 0 else 0.0,
            L=luminance(R_rad, dev),
            p_qd=p[iQ].copy(), n_qd=n[iQ].copy(),
            p_qd_mean=float(np.mean(p[iQ])), n_qd_mean=float(np.mean(n[iQ])),
            pH=float(p[m.iH]), nE=float(n[m.iE]),
            F_p=float(Ep), F_n=float(En),
            U_rad_qd=rad[iQ] / m.w[iQ], U_srh_qd=srh[iQ] / m.w[iQ], U_aug_qd=aug[iQ] / m.w[iQ],
        )
        return out

    def profile(self, X):
        m = self.m
        phi, n, p = X[:, 0], np.exp(X[:, 1]), np.exp(X[:, 2])
        rho = Q * (m.w * (p - n) + m.ndop) / m.w
        E = -(phi[1:] - phi[:-1]) / m.h
        return dict(z=m.z.copy(), phi=phi.copy(), n=n, p=p, rho=rho,
                    zl=0.5 * (m.z[1:] + m.z[:-1]), E=E)

    # ------------------------------------------------------------------ #
    def sweep(self, voltages, dwell=None, steady_tol=1e-7, t_max=1e-2, X0=None,
              profiles_at=(), verbose=False, dt0=1e-12):
        """Staircase voltage sweep.

        dwell : float or None
            Fixed time spent at every voltage (s).  None -> integrate each
            step until stationary (steady_tol) or t_max.
        Returns (list of observation dicts, dict of profiles, final state).
        """
        X = self.initial_state() if X0 is None else X0.copy()
        obs, prof = [], {}
        want = [float(v) for v in profiles_at]
        for V in voltages:
            if dwell is None:
                X, t, dt, Xp = self.advance(X, V, t_max, dt0=dt0, steady_tol=steady_tol,
                                            t_min=1e-9)
            else:
                X, t, dt, Xp = self.advance(X, V, dwell, dt0=dt0)
            o = self.observe(X, Xp, dt, V)
            o['t'] = t
            obs.append(o)
            if any(abs(V - w) < 1e-9 for w in want):
                prof[round(float(V), 6)] = self.profile(X)
            if verbose:
                print(f"V={V:5.2f}  t={t:8.2e}  J={o['J']:.3e}  p={o['p_qd_mean']:.2e}  "
                      f"n={o['n_qd_mean']:.2e}  EQE={100*o['EQE']:.2f}%", flush=True)
        return obs, prof, X

    def pulse(self, X, V_on, t_on, t_off, n_out=200, V_off=0.0):
        """Square pulse (Jo et al. 2024 style). Returns time traces."""
        traces = []
        for V, T in ((V_on, t_on), (V_off, t_off)):
            tgrid = np.geomspace(1e-9, T, n_out)
            t_prev = 0.0
            for tt in tgrid:
                X, t, dt, Xp = self.advance(X, V, tt - t_prev, dt0=min(1e-12, tt - t_prev))
                t_prev = tt
                o = self.observe(X, Xp, dt, V)
                o['t'] = tt + (0 if V == V_on else t_on)
                traces.append(o)
        return traces, X


# ---------------------------------------------------------------------- #
def ybar(lam_nm):
    """CIE 1931 photopic luminosity function (Wyman, Sloan & Shirley 2013 fit)."""
    lam = np.asarray(lam_nm, dtype=float)

    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    return 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)


def luminance(R_rad, dev):
    """Luminance [cd/m^2] from the radiative rate R_rad [cm^-2 s^-1] (Jung Eqs. 13-14)."""
    qd = dev.qd
    lam0 = qd.wavelength if qd.wavelength is not None else 1239.84193 / qd.Eg
    lam = np.linspace(380, 780, 801)
    sig = dev.fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    g = np.exp(-0.5 * ((lam - lam0) / sig) ** 2)
    g /= np.trapezoid(g, lam)
    power = dev.LEE * qd.Eg * Q * R_rad * 1e4          # W/m^2 out of the device
    radiance = power / np.pi                            # Lambertian, W/m^2/sr
    return 683.002 * radiance * np.trapezoid(ybar(lam) * g, lam)
