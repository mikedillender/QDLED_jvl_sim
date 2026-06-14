from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
import scipy.constants as cts
from collections import namedtuple
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

defect = namedtuple('defect', ['sites', 'location', 'dos', 'energy', 'sigma_e', 'sigma_h', 'transition', 'perp_dl'])


class Scaling():
    def __init__(self, input_length='cm', T=300):
        self.density = 1e19 if input_length == "cm" else 1e25
        self.energy = cts.k * T / cts.e
        self.mobility = 1 if input_length == "cm" else 1e-4
        self.time = cts.epsilon_0 * 1e-2 / (self.mobility * cts.e * self.density)
        self.length = np.sqrt(cts.epsilon_0 * 1e-2 * self.energy / (cts.e * self.density))
        self.generation = (self.density * self.mobility * self.energy) / self.length ** 2
        self.velocity = self.length / self.time
        self.current = cts.k * T * self.mobility * self.density / self.length


class Builder():
    def __init__(self, xpts, input_length='cm', T=300):
        self.scaling = Scaling(input_length, T)
        self.input_length = input_length
        self.has_qd=False
        self.xpts = xpts
        self.dx = (self.xpts[1:] - self.xpts[:-1]) / self.scaling.length
        self.nx = xpts.shape[0]
        self.ny = 1
        self.dimension = 1  # System is always one-dimensional

        self.Nc = np.zeros(self.nx, dtype=float)
        self.Nv = np.zeros(self.nx, dtype=float)
        self.Eg = np.zeros(self.nx, dtype=float)
        self.epsilon = np.zeros(self.nx, dtype=float)
        self.mass_e = np.zeros(self.nx, dtype=float)
        self.mass_h = np.zeros(self.nx, dtype=float)
        self.mu_e = np.zeros(self.nx, dtype=float)
        self.mu_h = np.zeros(self.nx, dtype=float)
        self.tau_e = np.zeros(self.nx, dtype=float)
        self.tau_h = np.zeros(self.nx, dtype=float)
        self.n1 = np.zeros(self.nx, dtype=float)
        self.p1 = np.zeros(self.nx, dtype=float)
        self.bl = np.zeros(self.nx, dtype=float)
        self.rho = np.zeros(self.nx, dtype=float)
        self.g = np.zeros(self.nx, dtype=float)
        self.B = np.zeros(self.nx, dtype=float)
        self.Cn = np.zeros(self.nx, dtype=float)
        self.Cp = np.zeros(self.nx, dtype=float)
        self.Etrap = np.zeros(self.nx, dtype=float)

        self.defects_list = []

    def add_material(self, mat, location=lambda pos: True):
        s = np.where(location(self.xpts))[0]
        N, t, vt, mu = self.scaling.density, self.scaling.time, self.scaling.energy, self.scaling.mobility

        # sites belonging to the region
        # s, pos = get_sites(self, location)

        N = self.scaling.density
        t = self.scaling.time
        vt = self.scaling.energy
        mu = self.scaling.mobility

        # default material parameters
        if self.input_length == 'm':
            mt = {'Nc': 1e25, 'Nv': 1e25, 'Eg': 1, 'epsilon': 1, 'mass_e': 1, \
                  'mass_h': 1, 'mu_e': 100e-4, 'mu_h': 100e-4, 'Et': 0, 'tau_e': 1e-6, \
                  'tau_h': 1e-6, 'affinity': 0, 'B': 0, 'Cn': 0, 'Cp': 0}
        else:
            mt = {'Nc': 1e19, 'Nv': 1e19, 'Eg': 1, 'epsilon': 1, 'mass_e': 1, \
                  'mass_h': 1, 'mu_e': 100, 'mu_h': 100, 'Et': 0, 'tau_e': 1e-6, \
                  'tau_h': 1e-6, 'affinity': 0, 'B': 0, 'Cn': 0, 'Cp': 0}

        arrays = {'Nc': self.Nc, 'Nv': self.Nv, 'Eg': self.Eg, \
                  'epsilon': self.epsilon, 'mass_e': self.mass_e, \
                  'mass_h': self.mass_h, 'mu_e': self.mu_e, 'mu_h': self.mu_h, \
                  'Et': self.Etrap, 'tau_e': self.tau_e, 'tau_h': self.tau_h, \
                  'affinity': self.bl, 'B': self.B, 'Cn': self.Cn, 'Cp': self.Cp}

        for key, val in mt.items():
            if key in mat.keys():
                val = mat[key]
            if not callable(val):
                arrays[key][s] = val
            else:
                print("not callable? ")
                exit()
            #    arrays[key][s] = (val(pos) * location(pos))[s]

        self.Nc[s] /= N
        self.Nv[s] /= N
        self.Eg[s] /= vt
        self.mu_e[s] /= mu
        self.mu_h[s] /= mu
        self.Etrap[s] /= vt
        self.tau_e[s] /= t
        self.tau_h[s] /= t
        self.bl[s] /= vt
        self.B[s] /= (1. / N) / t
        self.Cn[s] /= (1. / N ** 2) / t
        self.Cp[s] /= (1. / N ** 2) / t

        self.n1[s] = np.sqrt(self.Nc[s] * self.Nv[s]) * np.exp(-self.Eg[s] / 2 + self.Etrap[s])
        self.p1[s] = np.sqrt(self.Nc[s] * self.Nv[s]) * np.exp(-self.Eg[s] / 2 - self.Etrap[s])
        self.ni = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg / 2)

    def add_defects(self, location, N, sigma_e, sigma_h=None, E=None, transition=(1, -1)):
        if E is not None:
            E /= self.scaling.energy
        s = np.where(location(self.xpts))[0]
        NN = self.scaling.density * self.scaling.length
        sigma_e, sigma_h = sigma_e * NN, (sigma_h if sigma_h else sigma_e) * NN
        f = (lambda E: N(E * self.scaling.energy) / NN) if callable(N) else N / NN
        self.defects_list.append(defect(s, location, f, E, sigma_e, sigma_h, transition, self.dx))

    def add_donor(self, density, location=lambda pos: True):
        self.rho[np.where(location(self.xpts))[0]] += density / self.scaling.density

    def add_acceptor(self, density, location=lambda pos: True):
        self.rho[np.where(location(self.xpts))[0]] -= density / self.scaling.density

    def add_qd(self, t_s_nm, qd_mns=.19, qd_mps=.6, dEc=.28, dEv=.28,
               location=lambda pos: True, temp=300, r_qd=None):
        """Add a discretized quantum-dot layer.

        Parameters
        ----------
        t_s_nm : float
            QD shell thickness in nm, used in the tunneling probability.
        qd_mns, qd_mps : float
            Electron and hole effective masses in the QD shell.
        dEc, dEv : float
            Conduction- and valence-band offsets in eV for ETL->QD and
            HTL->QD injection.
        location : callable
            Boolean mask selecting the QD center sites.
        temp : float
            Temperature in K for the thermal velocity.
        r_qd : float, optional
            Physical QD radius in cm. This is required when the mask selects a
            single QD site. For multiple equally spaced QD sites it is inferred
            as half the center-to-center spacing if not supplied.
        """
        self.qd_sites = np.asarray(np.where(location(self.xpts))[0], dtype=int)
        if len(self.qd_sites) < 1:
            raise ValueError('add_qd found no QD sites. Check the QD location mask.')
        if not np.all(np.diff(self.qd_sites) == 1):
            raise ValueError('QD sites must be consecutive mesh points. Build the mesh with the QD centers inserted as adjacent points.')

        # Store QD/transport-layer band offsets in eV. These enter the
        # JMK effective capture fields as E = -(dphi + dE/q)/r_qd.
        self.qd_dEc = dEc
        self.qd_dEv = dEv

        if r_qd is None:
            if len(self.qd_sites) < 2:
                raise ValueError('add_qd requires r_qd when only one QD site is present.')
            self.rqd = (self.xpts[self.qd_sites[1]] - self.xpts[self.qd_sites[0]]) / 2
        else:
            self.rqd = r_qd

        # EML sites are the adjacent HTL surface site, all QD sites, and the
        # adjacent ETL surface site. qd_links are the modified links spanning
        # HTL/QD, QD/QD, ..., QD/ETL.
        self.eml_sites = [self.qd_sites[0] - 1, *self.qd_sites.tolist(), self.qd_sites[-1] + 1]
        self.qd_links = np.arange(self.qd_sites[0] - 1, self.qd_sites[-1] + 1, dtype=int)
        self.qd_m = len(self.qd_sites)

        self.qd_density = 3 / ((self.rqd ** 3) * self.scaling.density * 4 * np.pi)
        print("m_qd = ", self.qd_m, " | r_qd = ", self.rqd, ', sites ', self.qd_sites,
              ", links ", self.qd_links, ", density ", self.qd_density)

        qd_mn, qd_mp = qd_mns, qd_mps
        T_bn = np.exp(-10.246 * t_s_nm * np.sqrt(qd_mn * dEc)) if dEc >= 0 else 1.0
        T_bp = np.exp(-10.246 * t_s_nm * np.sqrt(qd_mp * dEv)) if dEv >= 0 else 1.0
        vth_n = 5.505695e5 * np.sqrt(temp / qd_mn)
        vth_p = 5.505695e5 * np.sqrt(temp / qd_mp)
        print("vth_n = ", f"{vth_n:.2e}", ', vth_p = ', f"{vth_p:.2e}")
        self.qd_vd_n = vth_n * T_bn / (4 * self.rqd)
        self.qd_vd_p = vth_p * T_bp / (4 * self.rqd)
        print("T_bn = ", f"{T_bn:.2e}", ', T_bp = ', f"{T_bp:.2e}",
              ", vd_n = ", f"{self.qd_vd_n:.2e}", ", vd_p = ", f"{self.qd_vd_p:.2e}")
        self.qd_alpha_n = 0.5 * T_bn * (0.000001)
        self.qd_alpha_p = 0.5 * T_bp * (0.000001)
        print("alpha_n = ", self.qd_alpha_n, ', alpha_p = ', self.qd_alpha_p)
        self.has_qd = True
        self.set_qd_poisson_widths()

    def set_qd_poisson_widths(self):
        """Set finite-volume charge widths used by Poisson's equation.

        Sesame's standard 1D Poisson residual divides the electric-displacement
        flux imbalance by the midpoint-to-midpoint control-volume width and then
        subtracts the local charge density. That is equivalent to assuming the
        local charge density fills the full midpoint-to-midpoint volume around
        every site. For discretized QDs this is not correct at the HTL/QD and
        QD/ETL boundaries: the adjacent transport-layer surface sites should not
        contribute charge inside the QD layer.

        This array stores the *physical* charge-control-volume width for each
        site, in the same scaled length units as ``self.dx``. getF.py and
        jacobian.py multiply the local charge term and its derivatives by
        ``poisson_charge_width / dxbar``. Away from QDs this ratio is one.
        """
        x = self.xpts / self.scaling.length
        n_sites = len(x)

        if not hasattr(self, 'qd_sites') or len(self.qd_sites) < 1:
            raise ValueError('set_qd_poisson_widths requires at least one QD site.')

        w = np.zeros(n_sites, dtype=float)
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
        w[0] = 0.5 * (x[1] - x[0])
        w[-1] = 0.5 * (x[-1] - x[-2])

        qd_sites = np.asarray(self.qd_sites, dtype=int)
        q0 = qd_sites[0]
        qN = qd_sites[-1]
        htl_surf = q0 - 1
        etl_surf = qN + 1

        if htl_surf < 1 or etl_surf > n_sites - 2:
            raise ValueError('QD sites must have neighboring transport-layer sites on both sides.')

        rqd_scaled = self.rqd / self.scaling.length
        x_left_qdl = x[q0] - rqd_scaled
        x_right_qdl = x[qN] + rqd_scaled

        # Clip the adjacent transport-layer charge volumes at the physical QD
        # layer boundaries. This prevents HTL/ETL density from being integrated
        # halfway into the QD layer by the default midpoint rule.
        w[htl_surf] = x_left_qdl - 0.5 * (x[htl_surf - 1] + x[htl_surf])
        w[etl_surf] = 0.5 * (x[etl_surf] + x[etl_surf + 1]) - x_right_qdl

        # QD-site charge volumes: first/last QD cells are bounded by the physical
        # QD-layer interfaces, while interior QD cells use midpoints between QD
        # centers. For m=1, the single QD owns one QD diameter.
        for k, qi in enumerate(qd_sites):
            left = x_left_qdl if k == 0 else 0.5 * (x[qd_sites[k - 1]] + x[qi])
            right = x_right_qdl if k == len(qd_sites) - 1 else 0.5 * (x[qi] + x[qd_sites[k + 1]])
            w[qi] = right - left

        special = [htl_surf, *qd_sites.tolist(), etl_surf]
        if np.any(w[special] <= 0):
            raise ValueError('Non-positive Poisson charge width detected at the QD interfaces.')

        self.poisson_charge_width = w

    def contact_S(self, Scn_left, Scp_left, Scn_right, Scp_right):
        v = self.scaling.velocity
        self.Scn, self.Scp = [Scn_left / v, Scn_right / v], [Scp_left / v, Scp_right / v]

    def contact_type(self, left_contact, right_contact, left_wf=None, right_wf=None):
        if left_contact == 'Schottky' and left_wf is None or right_contact == 'Schottky' and right_wf is None:
            raise ValueError("Schottky contacts require work functions.")
        self.contacts_bcs, self.contacts_WF = [left_contact, right_contact], [left_wf, right_wf]
