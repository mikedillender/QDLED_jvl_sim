# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from numpy import exp
import numpy as np
import scipy.constants as cts


def _qd_local_sites_and_links(sys, sites_i):
    """Return local indices for QD sites/links inside a current vector.

    Sesame sometimes calls current functions with arrays starting at link/site 0
    and sometimes with arrays shifted by one. The original QD code handled this
    by subtracting one from hard-coded indices when sites_i[0] != 0. This helper
    generalizes the same convention to an arbitrary number of QD sites.
    """
    shift = 0 if sites_i[0] == 0 else -1
    return np.asarray(sys.qd_sites, dtype=int) + shift, np.asarray(sys.qd_links, dtype=int) + shift


def _qd_capture_mu(E_eff, alpha, rqd, scaling):
    """Field-dependent JMK capture prefactor with a hard E_eff >= 0 cutoff."""
    F0 = 5e6
    Epos = np.maximum(E_eff, 0.0)
    return np.sqrt(Epos ** 3 / F0) * alpha / scaling.current


def get_n(sys, efn, v, sites):
    """
    Compute the electron density on the given sites.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    efn: numpy array of floats
        Values of the electron quasi-Fermi level.
    v: numpy array of floats
        Values of the electrostatic potential.
    sites: list of integers
        The sites where the electron density should be computed.

    Returns
    -------
    n: numpy array
    """

    n = sys.Nc[sites] * exp(+sys.bl[sites] + efn[sites] + v[sites])
    return n


def get_p(sys, efp, v, sites):
    """
    Compute the hole density on the given sites.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    efp: numpy array of floats
        Values of the hole quasi-Fermi level.
    v: numpy array of floats
        Values of the electrostatic potential.
    sites: list of integers
        The sites where the hole density should be computed.

    Returns
    -------
    p: numpy array
    """
    bl = sys.bl[sites]
    Eg = sys.Eg[sites]
    Nv = sys.Nv[sites]
    p = Nv * exp(-Eg - bl - efp[sites] - v[sites])
    return p


def get_bulk_rr(sys, n, p):
    # Compute the bulk recombination of the entire system for SRH, radiative and
    # Auger mechanisms
    ni2 = sys.ni ** 2
    _np = n * p
    r = (_np - ni2) / (sys.tau_h * (n + sys.n1) + sys.tau_e * (p + sys.p1)) \
        + (sys.Cn * n + sys.Cp * p) * (_np - ni2) \
        + sys.B * (_np - ni2)
    return r


def get_bulk_rr_derivs(sys, n, p):
    ni2 = sys.ni ** 2
    _np = n * p

    defn = (_np * (sys.tau_h * (n + sys.n1) + sys.tau_e * (p + sys.p1)) - (_np - ni2) * n * sys.tau_h) \
           / (sys.tau_h * (n + sys.n1) + sys.tau_e * (p + sys.p1)) ** 2 \
           + sys.Cn * n * (2 * _np - ni2) + sys.Cp * _np * p \
           + sys.B * _np

    defp = -(_np * (sys.tau_h * (n + sys.n1) + sys.tau_e * (p + sys.p1)) - (_np - ni2) * p * sys.tau_e) \
           / (sys.tau_h * (n + sys.n1) + sys.tau_e * (p + sys.p1)) ** 2 \
           - sys.Cn * n * _np - sys.Cp * p * (2 * _np - ni2) \
           - sys.B * _np

    dv = (_np - ni2) * (sys.tau_e * p - sys.tau_h * n) \
         / (sys.tau_h * (n + sys.n1) + sys.tau_e * (p + sys.p1)) ** 2 \
         + sys.Cn * n * (_np - ni2) - sys.Cp * p * (_np - ni2)

    return defn, defp, dv


def get_jn(sys, efn, v, sites_i, sites_ip1, dl):
    """
    Compute the electron current between sites ``site_i`` and ``sites_ip1``.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    efn: numpy array of floats
        Values of the electron quasi-Fermi level for the entire system (as given
        by the drift diffusion Poisson solver).
    v: numpy array of floats
        Values of the electrostatic potential for the entire system (as given
        by the drift diffusion Poisson solver).
    sites_i: list of integers
        Indices of the sites the current is coming from.
    sites_ip1: list of integers
        Indices of the sites the current is going to.
    dl: numpy arrays of floats
        Lattice distances between sites ``sites_i`` and sites ``sites_ip1``.

    Returns
    -------
    jn: numpy array of floats
    """
    #print(sites_i[0])
    # tol1 controls the minimum value of dv.  all values less than tol1 are set equal to tol1
    tol1 = 1e-12
    # tol2 controls threshold for taylor series expansion of jp in terms of dv0: series expansion is used if dv0<tol2
    tol2 = 1e-5
    # tol3 controls threshold for taylor series expansion of jp in terms of defp: series expansion is used if defp<tol3
    tol3 = 1e-9
    # this description of tol variables applies for the jp function, and jn and jp derivative functions

    vp0 = v[sites_i] + sys.bl[sites_i] + np.log(sys.Nc[sites_i]) # psi_n=q*phi+chi+kbT*ln(Nc)
    dv = vp0 - (v[sites_ip1] + sys.bl[sites_ip1] + np.log(sys.Nc[sites_ip1]))
    dv0 = dv
    dv = dv + (np.abs(dv) < tol1) * tol1

    efnp0 = efn[sites_i]
    efnp1 = efn[sites_ip1]
    defn = efnp1 - efnp0
    mu = sys.mu_e[sites_i]

    if (sys.has_qd and len(sites_i) > 1):
        qd_sites_l, qd_links = _qd_local_sites_and_links(sys, sites_i)
        vd = (4e10 * sys.rqd * 2) * (cts.e * 1e19)
        vd = vd / sys.scaling.current
        n_qd = exp(efnp0[qd_sites_l] + vp0[qd_sites_l])
        n_etl = exp(efnp0[qd_sites_l[-1] + 1] + vp0[qd_sites_l[-1] + 1])

        dphi = (v[sys.qd_sites[-1] + 1] - v[sys.qd_sites[-1]]) * sys.scaling.energy
        # JMK effective ETL->QD electron injection field includes the
        # conduction-band offset: E_n = -(dphi + dEc) / r_qd.
        lambdae = (((cts.e * sys.scaling.density) *
                    (np.pi * sys.scaling.density * sys.rqd ** 3))) * sys.qd_alpha_n
        E_etl = -(dphi + getattr(sys, 'qd_dEc', 0.0)) / sys.rqd
        mu_E = _qd_capture_mu(E_etl, lambdae, sys.rqd, sys.scaling)
        jni_qd = mu_E * (sys.qd_density - n_qd[-1]) * n_etl

    jn = (    mu * exp(efnp1)*(1 - exp(efnp0-efnp1)) / dl * dv / (-exp(-vp0) * (1 - exp(dv))) * (np.abs(dv0) >= tol2) + \
         -1 * mu * exp(efnp1)*(1 - exp(efnp0-efnp1)) / dl / (-exp(-vp0) * (1 + .5 * dv0 + 1/6.*(dv0)**2)) * (np.abs(dv0) < tol2)) * (np.abs(defn)>=tol3) + \
         (    mu * exp(efnp1)*(-(efnp0 - efnp1))     / dl * dv / (-exp(-vp0) * (1 - exp(dv))) * (np.abs(dv0) >= tol2) + \
         -1 * mu * exp(efnp1)*(-(efnp0 - efnp1))     / dl / (-exp(-vp0) * (1 + .5 * dv0 + 1 / 6. * (dv0) ** 2)) * (np.abs(dv0) < tol2)) * (np.abs(defn) < tol3)

    if (sys.has_qd and len(sites_i) > 1):
        jn[qd_links] /= 1e8
        for k in range(len(qd_sites_l) - 1):
            jn[qd_sites_l[k]] += vd * (n_qd[k + 1] - n_qd[k])
        jn[qd_sites_l[-1]] += jni_qd




    #if(len(sites_i)>1):
    #    print('hi',vd,jn[qd1_i])

    return jn


def get_jp(sys, efp, v, sites_i, sites_ip1, dl):
    """
    Compute the hole current between sites ``site_i`` and ``sites_ip1``.

    Parameters
    ----------
    sys: Builder
        The discretized system.
    efp: numpy array of floats
        Values of the hole quasi-Fermi level for the entire system (as given
        by the drift diffusion Poisson solver).
    v: numpy array of floats
        Values of the electrostatic potential for the entire system (as given
        by the drift diffusion Poisson solver).
    sites_i: list of integers
        Indices of the sites the current is coming from.
    sites_ip1: list of integers
        Indices of the sites the current is going to.
    dl: numpy arrays of floats
        Lattice distances between sites ``sites_i`` and sites ``sites_ip1``.

    Returns
    -------
    jp: numpy array of floats
    """
    tol1 = 1e-12
    tol2 = 1e-5
    tol3 = 1e-9

    vp0 = v[sites_i] + sys.bl[sites_i] + sys.Eg[sites_i] - np.log(sys.Nv[sites_i])
    dv = vp0 - (v[sites_ip1] + sys.bl[sites_ip1] + sys.Eg[sites_ip1] - np.log(sys.Nv[sites_ip1]))
    dv0 = dv
    dv = dv + (np.abs(dv) < tol1) * tol1

    efpp0 = -efp[sites_i]
    efpp1 = -efp[sites_ip1]
    defp = efpp1 - efpp0

    mu = sys.mu_h[sites_i]
    #V=sys.V
    if (sys.has_qd and len(sites_i) > 1):
        qd_sites_l, qd_links = _qd_local_sites_and_links(sys, sites_i)
        vd = (7.1e9 * sys.rqd * 2) * (cts.e * sys.scaling.density)
        vd = -vd / sys.scaling.current
        p_qd = exp(efpp0[qd_sites_l] - vp0[qd_sites_l])
        p_htl = exp(efpp0[qd_sites_l[0] - 1] - vp0[qd_sites_l[0] - 1])

        dphi = (v[sys.qd_sites[0]] - v[sys.qd_sites[0] - 1])
        lambdah = (((cts.e * sys.scaling.density) *
                    (np.pi * sys.scaling.density * sys.rqd ** 3))) * sys.qd_alpha_p
        # JMK effective HTL->QD hole injection field includes the
        # valence-band offset: E_p = -(dphi + dEv/q) / r_qd.
        E_htl = -(dphi * sys.scaling.energy + getattr(sys, 'qd_dEv', 0.0)) / sys.rqd
        mu_E = _qd_capture_mu(E_htl, lambdah, sys.rqd, sys.scaling)
        jpi_qd = mu_E * (sys.qd_density - p_qd[0]) * p_htl


    jp = (mu * exp(efpp1) * (1 - exp(efpp0-efpp1)) / dl * dv / (-exp(vp0) * (1 - exp(-dv))) * (np.abs(dv0) >= tol2) + \
          mu * exp(efpp1) * (1 - exp(efpp0-efpp1)) / dl * 1 / (-exp(vp0) * (1 - .5*(dv0) + 1/6.*(dv0)**2.)) * (np.abs(dv0) < tol2)) * (np.abs(defp) >= tol3) + \
         (mu * exp(efpp1) * ( -(efpp0 - efpp1))    / dl * dv / (-exp(vp0) * (1 - exp(-dv))) * (np.abs(dv0) >= tol2) + \
          mu * exp(efpp1) * ( -(efpp0 - efpp1))    / dl * 1 / (-exp(vp0) * (1 - .5 * (dv0) + 1 / 6. * (dv0) ** 2.)) * (np.abs(dv0) < tol2)) * (np.abs(defp) < tol3)

    if (sys.has_qd and len(sites_i) > 1):
        jp[qd_links] /= 1e8
        for k in range(len(qd_sites_l) - 1):
            jp[qd_sites_l[k]] += vd * (p_qd[k + 1] - p_qd[k])
        jp[qd_sites_l[0] - 1] *= mu[qd_sites_l[0]] / mu[qd_sites_l[0] - 1]
        jp[qd_sites_l[0] - 1] += jpi_qd
        #jp[sites_i >= qd2_i] = 0
    #if(len(sites_i)>1):
    #    print('hi',jpt_qd,jp[qd1_i])

    return jp


def get_jn_derivs(sys, efn, v, sites_i, sites_ip1, dl):
    tol1 = 1e-12
    tol2 = 1e-5
    tol3 = 1e-9


    vp0 = v[sites_i] + sys.bl[sites_i] + np.log(sys.Nc[sites_i])
    vp1 = v[sites_ip1] + sys.bl[sites_ip1] + np.log(sys.Nc[sites_ip1])
    dv = vp0 - vp1
    dv0 = dv
    dv = dv + (np.abs(dv) < tol1) * tol1

    efnp0 = efn[sites_i]
    efnp1 = efn[sites_ip1]
    defn = efnp1 - efnp0
    mu = sys.mu_e[sites_i]
    ev0 = exp(-vp0)

    if (sys.has_qd and len(sites_i) > 1):
        qd_sites_l, qd_links = _qd_local_sites_and_links(sys, sites_i)
        vd = (4e10 * sys.rqd * 2) * (cts.e * 1e19)
        vd = vd / sys.scaling.current
        n_qd = exp(efnp0[qd_sites_l] + vp0[qd_sites_l])
        n_etl = exp(efnp0[qd_sites_l[-1] + 1] + vp0[qd_sites_l[-1] + 1])

        dphi = (v[sys.qd_sites[-1] + 1] - v[sys.qd_sites[-1]])
        lambdae = (((cts.e * sys.scaling.density) *
                    (np.pi * sys.scaling.density * (sys.rqd ** 3)))) * sys.qd_alpha_n
        u_etl = dphi + getattr(sys, 'qd_dEc', 0.0) / sys.scaling.energy
        E_etl = -(u_etl * sys.scaling.energy) / sys.rqd
        mu_E = _qd_capture_mu(E_etl, lambdae, sys.rqd, sys.scaling)
        inj_denom_etl = max(abs(u_etl), 1e-300)
        jni_qd = mu_E * (sys.qd_density - n_qd[-1]) * n_etl



    defn_i = (1. / dl * exp(efnp0 + vp0) * (dv) / (1 - exp(dv)) * (np.abs(dv0) >= tol2) + \
             -1. / dl * exp(efnp0 + vp0) / (1 + .5*dv0 + 1/6.*dv0**2) * (np.abs(dv0) < tol2)) * (np.abs(defn) >= tol3) + \
             (1. * exp(efnp1) / dl * exp(vp0) * (dv) / (1 - exp(dv)) * (np.abs(dv0) >= tol2) + \
             -1. * exp(efnp1) / dl * exp(vp0) / (1 + .5 * dv0 + 1 / 6. * dv0 ** 2) * (np.abs(dv0) < tol2)) * (np.abs(defn) < tol3)

    defn_ip1 = (-1. / dl * exp(efnp1 + vp0) * (dv) / (1 - exp(dv)) * (np.abs(dv0) >= tol2) + \
                 1. / dl * exp(efnp1 + vp0) / (1 + .5*dv0 + 1/6.*dv0**2) * (np.abs(dv0) < tol2)) * (np.abs(defn) >= tol3) + \
               (-1. * exp(efnp1) *(1-(efnp0 - efnp1))/ dl * exp(vp0) * (dv) / (1 - exp(dv)) * (np.abs(dv0) >= tol2) + \
                 1. * exp(efnp1) *(1-(efnp0 - efnp1))/ dl * exp(vp0) / (1 + .5 * dv0 + 1 / 6. * dv0 ** 2) * (np.abs(dv0) < tol2)) * (np.abs(defn) < tol3)

    dv_i = (-exp(efnp1)*(1 - exp(efnp0-efnp1)) / dl * ev0 * (1 + dv - exp(dv)) / (ev0 ** 2 * (exp(dv) - 1) ** 2) * (np.abs(dv0) >= tol2) + \
           -6*exp(vp0) * exp(efnp1)*(1 - exp(efnp0-efnp1)) / dl * (3 + vp0 + vp0**2 - 2*vp0*vp1 + vp1*(-1 + vp1)) \
           / (6 + vp0**2 + vp0*(3 - 2*vp1) + vp1*(-3 + vp1))**2 * (np.abs(dv0) < tol2)) * (np.abs(defn)>=tol3) + \
           (-exp(efnp1) * ( -(efnp0 - efnp1)) / dl * ev0 * (1 + dv - exp(dv)) / (ev0 ** 2 * (exp(dv) - 1) ** 2) * (np.abs(dv0) >= tol2) + \
            -6 * exp(vp0) * exp(efnp1) * (-(efnp0 - efnp1)) / dl * (3 + vp0 + vp0 ** 2 - 2 * vp0 * vp1 + vp1 * (-1 + vp1)) \
            / (6 + vp0 ** 2 + vp0 * (3 - 2 * vp1) + vp1 * (-3 + vp1)) ** 2 * (np.abs(dv0) < tol2)) * (np.abs(defn) < tol3)

    dv_ip1 = (-1. / dl * exp(efnp1)*(1 - exp(efnp0-efnp1)) * exp(-vp1) * (1 - dv - exp(-dv)) / (exp(-2 * vp1) * (1 - exp(-dv)) ** 2) * (np.abs(dv0) >= tol2) + \
             -6 * exp(vp0) * exp(efnp1)*(1 - exp(efnp0-efnp1)) / dl * (3 + 2*vp0 - 2*vp1)\
             / (6 + vp0**2 + vp0*(3 - 2*vp1) + vp1*(-3 + vp1))**2 * (np.abs(dv0) < tol2)) * (np.abs(defn) >= tol3) + \
             (-1. / dl * exp(efnp1) * (-(efnp0 - efnp1)) * exp(-vp1) * (1 - dv - exp(-dv)) / (exp(-2 * vp1) * (1 - exp(-dv)) ** 2) * (np.abs(dv0) >= tol2) + \
              -6 * exp(vp0) * exp(efnp1) * (-(efnp0 - efnp1)) / dl * (3 + 2 * vp0 - 2 * vp1) \
              / (6 + vp0 ** 2 + vp0 * (3 - 2 * vp1) + vp1 * (-3 + vp1)) ** 2 * (np.abs(dv0) < tol2)) * (np.abs(defn) < tol3)

    defn_i, defn_ip1, dv_i, dv_ip1=mu * defn_i, mu * defn_ip1, mu * dv_i, mu * dv_ip1

    if (sys.has_qd and len(sites_i) > 1):
        dv_i[qd_links] /= 1e8
        dv_ip1[qd_links] /= 1e8
        defn_i[qd_links] /= 1e8
        defn_ip1[qd_links] /= 1e8

        # QD-to-QD electron hopping links.
        for k in range(len(qd_sites_l) - 1):
            link = qd_sites_l[k]
            dv_i[link] += -vd * n_qd[k]
            dv_ip1[link] += +vd * n_qd[k + 1]
            defn_i[link] += -vd * n_qd[k]
            defn_ip1[link] += +vd * n_qd[k + 1]

        # ETL -> last QD electron capture link.
        link = qd_sites_l[-1]
        dv_i[link] += +(3 * jni_qd / (2 * inj_denom_etl) - mu_E * n_qd[-1] * n_etl)
        dv_ip1[link] += +jni_qd * (-3 / (2 * inj_denom_etl) + 1)
        defn_i[link] += -mu_E * n_qd[-1] * n_etl
        defn_ip1[link] += +jni_qd


    return defn_i, defn_ip1, dv_i, dv_ip1


def get_jp_derivs(sys, efp, v, sites_i, sites_ip1, dl):
    tol1 = 1e-12
    tol2 = 1e-5
    tol3 = 1e-9

    vp0 = v[sites_i] + sys.bl[sites_i] + sys.Eg[sites_i] - np.log(sys.Nv[sites_i])
    vp1 = v[sites_ip1] + sys.bl[sites_ip1] + sys.Eg[sites_ip1] - np.log(sys.Nv[sites_ip1])
    dv = vp0 - vp1
    dv0 = dv
    dv = dv + (np.abs(dv) < tol1) * tol1

    efpp0 = -efp[sites_i]
    efpp1 = -efp[sites_ip1]
    defp = efpp1 - efpp0
    mu = sys.mu_h[sites_i]

    ev0 = exp(vp0)


    if (sys.has_qd and len(sites_i) > 1):
        qd_sites_l, qd_links = _qd_local_sites_and_links(sys, sites_i)
        vd = (7.1e9 * sys.rqd * 2) * (cts.e * 1e19)
        vd = -vd / sys.scaling.current

        p_qd = exp(efpp0[qd_sites_l] - vp0[qd_sites_l])
        p_htl = exp(efpp0[qd_sites_l[0] - 1] - vp0[qd_sites_l[0] - 1])

        dphi = (v[sys.qd_sites[0]] - v[sys.qd_sites[0] - 1])
        lambdah = (((cts.e * sys.scaling.density) *
                    (np.pi * sys.scaling.density * sys.rqd ** 3))) * sys.qd_alpha_p
        u_htl = dphi + getattr(sys, 'qd_dEv', 0.0) / sys.scaling.energy
        E_htl = -(u_htl * sys.scaling.energy) / sys.rqd
        mu_E = _qd_capture_mu(E_htl, lambdah, sys.rqd, sys.scaling)
        inj_denom_htl = max(abs(u_htl), 1e-300)
        jpi_qd = mu_E * (sys.qd_density - p_qd[0]) * p_htl


    defp_i = -(exp(efpp0 - vp0) * dv / (dl * (1 - exp(-dv))) * (np.abs(dv0) >= tol2) + \
              exp(efpp0 - vp0) / (dl) / (1 - .5*(vp0-vp1) + 1/6.*(vp0-vp1)**2.) * (np.abs(dv0) < tol2)) * (np.abs(defp)>=tol3) + \
             -(exp(efpp1) * exp(-vp0) * dv / (dl * (1 - exp(-dv))) * (np.abs(dv0) >= tol2) + \
              exp(efpp1) * exp(-vp0) / (dl) / (1 - .5 * (vp0 - vp1) + 1 / 6. * (vp0 - vp1) ** 2.) * (np.abs(dv0) < tol2)) * (np.abs(defp) < tol3)

    defp_ip1 = -(-exp(efpp1 - vp0) * dv / (dl * (1 - exp(-dv))) * (np.abs(dv0) >= tol2) + \
                -exp(efpp1 - vp0)  / (dl) / (1 - .5*(vp0-vp1) + 1/6.*(vp0-vp1)**2.) * (np.abs(dv0) < tol2)) * (np.abs(defp)>=tol3) + \
               -(-exp(efpp1) * exp(-vp0)*(1-(efpp0 - efpp1)) * dv / (dl * (1 - exp(-dv))) * (np.abs(dv0) >= tol2) + \
                -exp(efpp1) * exp(-vp0)*(1-(efpp0 - efpp1)) / (dl) / (1 - .5*(vp0-vp1) + 1/6.*(vp0-vp1)**2.) * (np.abs(dv0) < tol2)) * (np.abs(defp) < tol3)

    dv_i = (-exp(efpp0)*(1 - exp(efpp1-efpp0)) * ev0 * (exp(-dv) + (-1 + dv)) / (dl * exp(2 * vp0) * (1 - exp(-dv)) ** 2) * (np.abs(dv0) >= tol2) + \
           -6* exp(efpp0)*(1 - exp(efpp1-efpp0)) / dl * (-exp(-vp0)) * (3 + (-1 + vp0)*vp0 + vp1 - 2*vp0*vp1 + vp1**2) \
           / (6 + vp0**2 + vp1*(3+vp1) - vp0*(3 + 2*vp1)) ** 2 * (np.abs(dv0) < tol2)) * (np.abs(defp) >= tol3) + \
           (-exp(efpp0) * (-(efpp1 - efpp0)) * ev0 * (exp(-dv) + (-1 + dv)) / (dl * exp(2 * vp0) * (1 - exp(-dv)) ** 2) * (np.abs(dv0) >= tol2) + \
            -6 * exp(efpp0) * (-(efpp1 - efpp0)) / dl * (-exp(-vp0)) * (3 + (-1 + vp0) * vp0 + vp1 - 2 * vp0 * vp1 + vp1 ** 2) \
            / (6 + vp0 ** 2 + vp1 * (3 + vp1) - vp0 * (3 + 2 * vp1)) ** 2 * (np.abs(dv0) < tol2)) * (np.abs(defp) < tol3)

    dv_ip1 = (-exp(efpp0)*(1 - exp(efpp1-efpp0)) * ev0 * (1 + exp(-dv) * (-1 - dv)) / (dl * exp(2 * vp0) * (1 - exp(-dv)) ** 2) * (np.abs(dv0) >= tol2) + \
             6 * exp(efpp0)*(1 - exp(efpp1-efpp0)) / dl * (-exp(-vp0)) * (-3 + 2*vp0 - 2*vp1) \
                / (6 + vp0 ** 2 + vp1 * (3 + vp1) - vp0 * (3 + 2 * vp1)) ** 2 * (np.abs(dv0) < tol2)) * (np.abs(defp) >= tol3) + \
             (-exp(efpp0) * (-(efpp1 - efpp0)) * ev0 * (1 + exp(-dv) * (-1 - dv)) / (dl * exp(2 * vp0) * (1 - exp(-dv)) ** 2) * (np.abs(dv0) >= tol2) + \
              6 * exp(efpp0) * (-(efpp1 - efpp0)) / dl * (-exp(-vp0)) * (-3 + 2 * vp0 - 2 * vp1) \
              / (6 + vp0 ** 2 + vp1 * (3 + vp1) - vp0 * (3 + 2 * vp1)) ** 2 * (np.abs(dv0) < tol2)) * (np.abs(defp) < tol3)

    defp_i, defp_ip1, dv_i, dv_ip1=mu * defp_i, mu * defp_ip1, mu * dv_i, mu * dv_ip1
    ''''''
    if (sys.has_qd and len(sites_i) > 1):
        dv_i[qd_links] /= 1e8
        dv_ip1[qd_links] /= 1e8
        defp_i[qd_links] /= 1e8
        defp_ip1[qd_links] /= 1e8

        htl_link = qd_sites_l[0] - 1
        dv_i[htl_link] *= mu[qd_sites_l[0]] / mu[htl_link]
        dv_ip1[htl_link] *= mu[qd_sites_l[0]] / mu[htl_link]
        defp_i[htl_link] *= mu[qd_sites_l[0]] / mu[htl_link]
        defp_ip1[htl_link] *= mu[qd_sites_l[0]] / mu[htl_link]

        # QD-to-QD hole hopping links.
        for k in range(len(qd_sites_l) - 1):
            link = qd_sites_l[k]
            dv_i[link] += +vd * p_qd[k]
            dv_ip1[link] += -vd * p_qd[k + 1]
            defp_i[link] += +vd * p_qd[k]
            defp_ip1[link] += -vd * p_qd[k + 1]

        # HTL -> first QD hole capture link.
        dv_i[htl_link] += -jpi_qd * (-3 / (2 * inj_denom_htl) + 1)
        dv_ip1[htl_link] += -3 * jpi_qd / (2 * inj_denom_htl) + mu_E * p_qd[0] * p_htl
        defp_i[htl_link] += -jpi_qd
        defp_ip1[htl_link] += mu_E * p_htl * p_qd[0]



    return defp_i, defp_ip1, dv_i, dv_ip1


def get_srh_rr_derivs(sys, n, p, n1, p1, tau_e, tau_h):
    ni2 = n1 * p1
    _np = n * p

    defn = (_np * (tau_h * (n + n1) + tau_e * (p + p1)) - (_np - ni2) * n * tau_h) \
           / (tau_h * (n + n1) + tau_e * (p + p1)) ** 2
    defp = -(_np * (tau_h * (n + n1) + tau_e * (p + p1)) - (_np - ni2) * p * tau_e) \
           / (tau_h * (n + n1) + tau_e * (p + p1)) ** 2
    dv = (_np - ni2) * (tau_e * p - tau_h * n) / (tau_h * (n + n1) + tau_e * (p + p1)) ** 2

    return defn, defp, dv

