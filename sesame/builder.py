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
        #s, pos = get_sites(self, location)

        N = self.scaling.density
        t = self.scaling.time
        vt = self.scaling.energy
        mu = self.scaling.mobility

        # default material parameters
        if self.input_length == 'm':
            mt = {'Nc': 1e25, 'Nv': 1e25, 'Eg': 1, 'epsilon': 1, 'mass_e': 1,\
              'mass_h': 1, 'mu_e': 100e-4, 'mu_h': 100e-4, 'Et': 0, 'tau_e': 1e-6,\
              'tau_h': 1e-6, 'affinity': 0, 'B': 0, 'Cn': 0, 'Cp': 0}
        else:
            mt = {'Nc': 1e19, 'Nv': 1e19, 'Eg': 1, 'epsilon': 1, 'mass_e': 1, \
                  'mass_h': 1, 'mu_e': 100, 'mu_h': 100, 'Et': 0, 'tau_e': 1e-6, \
                  'tau_h': 1e-6, 'affinity': 0, 'B': 0, 'Cn': 0, 'Cp': 0}

        arrays = {'Nc': self.Nc, 'Nv': self.Nv, 'Eg': self.Eg,\
                  'epsilon': self.epsilon, 'mass_e': self.mass_e,\
                  'mass_h': self.mass_h, 'mu_e': self.mu_e, 'mu_h': self.mu_h,\
                  'Et': self.Etrap, 'tau_e': self.tau_e, 'tau_h': self.tau_h,\
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

    def add_qd(self, location=lambda pos: True):
        self.qd_sites=(np.where(location(self.xpts))[0])
        self.rqd=(self.xpts[self.qd_sites[1]]-self.xpts[self.qd_sites[0]])/2
        self.qd_links=self.qd_sites.copy()
        self.qd_links=np.insert(self.qd_links,0,self.qd_links[0]-1)
        print("r_qd = ",self.rqd,', sites ',self.qd_sites, ", links ",self.qd_links)
        #print(self.xpts[self.qd_sites[0]-1:self.qd_sites[1]+4])
        #print(self.xpts[self.qd_sites])

    def contact_S(self, Scn_left, Scp_left, Scn_right, Scp_right):
        v = self.scaling.velocity
        self.Scn, self.Scp = [Scn_left / v, Scn_right / v], [Scp_left / v, Scp_right / v]

    def contact_type(self, left_contact, right_contact, left_wf=None, right_wf=None):
        if left_contact == 'Schottky' and left_wf is None or right_contact == 'Schottky' and right_wf is None:
            raise ValueError("Schottky contacts require work functions.")
        self.contacts_bcs, self.contacts_WF = [left_contact, right_contact], [left_wf, right_wf]
