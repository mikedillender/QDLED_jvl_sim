# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import numpy as np
from scipy.io import savemat
from . import analyzer
from .utils import save_sim
import matplotlib.pyplot as plt

from .analyzer import Analyzer

import scipy.sparse.linalg as lg
from scipy.sparse import coo_matrix, csr_matrix
from .getFandJ_eq import getFandJ_eq
from .getF import getF
from .jacobian import getJ

import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').setLevel(logging.ERROR)

__all__ = ['solve', 'IVcurve']

# check if MUMPS is available
mumps_available = False
try:
    from . import mumps

    mumps_available = True
except:
    pass
print("mumps available = ", mumps_available)


class NewtonError(Exception):
    pass


class SparseSolverError(Exception):
    pass


class BCsError(Exception):
    def __init__(self, BCs):
        msg = "\n*********************************************" + \
              "\n*  Unknown contacts boundary conditions     *" + \
              "\n*********************************************"
        logging.error(msg)
        logging.error(
            "Contacts boundary conditions: '{0}' is different from 'Ohmic', 'Schottky', or 'Neumann'.\n".format(BCs))


class Solver():
    """
    An object that creates an interface for the equilibrium and nonequilibrium
    solvers of Sesame, and stores the equilibrium electrostatic potential once
    computed.

    Parameters
    ----------
    use_mumps: boolean
        Flag for the use of the MUMPS library if available. The flag is set to
        True by default. If the MUMPS library is absent, the flag has no effect.

    Attributes
    ----------
    equilibrium: numpy array of floats
        Electrostatic potential computed at thermal equilibrium.
    """

    def __init__(self, use_mumps=True):
        self.equilibrium = None
        self.use_mumps = use_mumps

    def make_guess(self, system):
        # Make a linear assumption based on Dirichlet contacts
        nx = system.nx
        # determine what the potential on the left might be
        if system.contacts_bcs[0] == 'Ohmic' or system.contacts_bcs[0] == 'Neutral':
            if system.rho[0] < 0:  # p-doped
                v_left = -system.Eg[0] - np.log(abs(system.rho[0]) / system.Nv[0]) - system.bl[0]
            else:  # n-doped
                v_left = np.log(system.rho[0] / system.Nc[0]) - system.bl[0]
        if system.contacts_bcs[0] == 'Schottky':
            v_left = -system.contacts_WF[0] / system.scaling.energy

        if system.contacts_bcs[0] == 'cam':
            v_left = -(system.contacts_WF[0]) / system.scaling.energy

        # determine what the potential on the right might be
        if system.contacts_bcs[1] == 'Ohmic' or system.contacts_bcs[1] == 'Neutral':
            if system.rho[nx - 1] < 0:
                v_right = -system.Eg[nx - 1] - np.log(abs(system.rho[nx - 1]) / system.Nv[nx - 1]) - system.bl[nx - 1]
            else:
                v_right = np.log(system.rho[nx - 1] / system.Nc[nx - 1]) - system.bl[nx - 1]
        if system.contacts_bcs[1] == 'Schottky':
            v_right = -system.contacts_WF[1] / system.scaling.energy
        if system.contacts_bcs[1] == 'cam':
            v_right = -system.contacts_WF[1] / system.scaling.energy

        # Make a linear guess for the equilibrium potential
        v = np.linspace(v_left, v_right, system.nx)
        if system.dimension == 2:
            # replicate the guess in the y-direction
            v = np.tile(v, system.ny)
        return v

    def solve(self, system, compute='all', guess=None, tol=1e-6, periodic_bcs=True, \
              maxiter=300, verbose=True, htp=1):
        """
        Solve the drift diffusion Poisson equation on a given discretized
        system out of equilibrium. If the equilibrium electrostatic potential is
        not yet computed, the routine will compute it and save it for further
        computations.

        Parameters
        ----------
        system: Builder
            The discretized system.
        compute: string
            Set to 'all' to solve the full drift-diffusion-Poisson equations, or
            to 'Poisson' to only solve the Poisson equation. Default is set to
            'all'.
        guess: dictionary of numpy arrays of floats
            Contains the one-dimensional arrays of the initial guesses for the
            electron quasi-Fermi level, the hole quasi-Fermi level and the
            electrostatic potential. Keys should be 'efn', 'efp' and 'v'.
        tol: float
            Accepted error made by the Newton-Raphson scheme.
        periodic_bcs: boolean
            Defines the choice of boundary conditions in the y-direction. True
            (False) corresponds to periodic (abrupt) boundary conditions.
        maxiter: integer
            Maximum number of steps taken by the Newton-Raphson scheme.
        verbose: boolean
            The solver returns the step number and the associated error at every
            step if set to True (default).
        htp: integer
            Number of homotopic Newton loops to perform.

        Returns
        -------

        solution: dictionary with  numpy arrays of floats
            Dictionary containing the one-dimensional arrays of the solution. The
            keys are the same as the ones for the guess. An exception is raised
            if no solution could be found.
        """

        # Check if we only want the electrostatic potential
        if compute == 'Poisson':  # Only Poisson is solved
            self.equilibrium = None  # delete it to force its computation

        if self.equilibrium is None:
            if verbose == True:
                logging.info("Solving for the equilibrium electrostatic potential")

            if guess is None:
                guess = self.make_guess(system)
            else:
                # testing of the data type of guess.
                if type(guess) is dict:
                    guess = guess['v']

            # Compute the potential (Newton returns an array)
            self.equilibrium = self._newton(system, guess, tol=tol, \
                                            periodic_bcs=periodic_bcs, \
                                            maxiter=maxiter, verbose=verbose, htp=htp)

            if self.equilibrium is None:
                return None
        # print(guess['v'])
        # Return now if the electrostatic potential is all we wanted
        if compute == 'Poisson':
            efn = np.zeros_like(self.equilibrium)
            efp = np.zeros_like(self.equilibrium)
            return {'efn': efn, 'efp': efp, 'v': np.copy(self.equilibrium)}

        # Otherwise, keep going with the full problem
        if compute == 'all':
            # array to pass to Newton routine
            x = np.zeros((3 * system.nx * system.ny,), dtype=np.float64)
            if guess is None:  # I will try with equilibrium
                x[2::3] = np.copy(self.equilibrium)
            else:
                x[0::3] = guess['efn']
                x[1::3] = guess['efp']
                x[2::3] = guess['v']
            '''fig = plt.figure()
            X0 = system.xpts
            ax = fig.add_subplot(111)
            vx, vefn, vefp = x[2::3], x[0::3], x[1::3]
            vt = system.scaling.energy
            l1, = ax.plot(X0, vt*vefn, lw=2, color='#2e89cf', ls='--')
            l2, = ax.plot(X0, vt*vefp, lw=2, color='#cf392e', ls='--')
            l3, = ax.plot(X0, -vt * (vx + system.bl), lw=2, color='k', ls='-')
            l4, = ax.plot(X0, -vt * (vx + system.bl + system.Eg), lw=2, color='k', ls='-')
            l4, = ax.plot(X0, vt * (vx-vx[0]), lw=2, color='g', ls='--')
            plt.title("initial equilibrium condition")
            plt.show()'''
            # Compute solution (Newton returns an array)
            x = self._newton(system, x, tol=tol, periodic_bcs=periodic_bcs, \
                             maxiter=maxiter, verbose=verbose, htp=htp)

            if x is not None:
                return {'efn': x[0::3], 'efp': x[1::3], 'v': x[2::3]}
            else:
                return None

    def _damping(self, dx):
        # This damping procedure is inspired from Solid-State Electronics, vol. 19,
        # pp. 991-992 (1976).

        b = np.abs(dx) > 1
        dx[b] = np.log(1 + np.abs(dx[b]) * 1.72) * np.sign(dx[b])

    def _sparse_solver(self, J, f):
        spsolve = lg.spsolve
        # print("f: ",len(f))
        # print("J: ",J.shape)
        # print(type(J))
        if self.use_mumps and mumps_available:
            # print("using mumps")
            spsolve = mumps.spsolve
        else:
            J = J.tocsr()
        # print("J: ",J)
        # print("F: ",f)
        dx = spsolve(J, f)
        return dx

    def _get_system(self, x, system, periodic_bcs):
        # Compute the right hand side of J * x = f
        if self.equilibrium is None:  # this is where the guess comes from i think
            size = system.nx * system.ny
            f, rows, columns, data = getFandJ_eq(system, x)
        else:
            f = getF(system, x[2::3], x[0::3], x[1::3], self.equilibrium)
            rows, columns, data = getJ(system, x[2::3], x[0::3], x[1::3])

        # form the Jacobian
        if self.use_mumps and mumps_available:
            J = coo_matrix((data, (rows, columns)), dtype=np.float64)
        else:
            J = csr_matrix((data, (rows, columns)), dtype=np.float64)

        return f, J

    def _newton(self, system, x, tol=1e-6, periodic_bcs=True, maxiter=300, verbose=True, htp=1):

        htpy = np.linspace(1. / htp, 1, htp)

        for gdx, gamma in enumerate(htpy):
            if verbose:
                logging.info("Newton loop {0}/{1}".format(gdx + 1, htp))

            if gamma < 1:
                htol = 1
            else:
                htol = tol

            cc = 0
            converged = False
            if gamma != 1:
                f0, _ = self._get_system(x, system, periodic_bcs)
            while not converged:
                cc = cc + 1
                # break if no solution found after maxiterations

                if cc == 100:
                    print("try decreasing voltage :D")
                    if (x[system.qd_sites[0] * 3 + 2] > x[(system.qd_sites[0] - 1) * 3 + 2]):
                        x[system.qd_sites[0] * 3 + 2] -= 1
                if cc == 101:
                    if (x[system.qd_sites[1] * 3 + 2] > x[(system.qd_sites[0]) * 3 + 2]):
                        x[system.qd_sites[1] * 3 + 2] -= 2
                if cc == 102:
                    if (x[(system.qd_sites[1] + 1) * 3 + 2] > x[(system.qd_sites[1]) * 3 + 2]):
                        x[(system.qd_sites[1] + 1) * 3 + 2] -= 3

                # solve linear system
                f, J = self._get_system(x, system, periodic_bcs)
                # kam = np.argmax(abs(f))
                # print('max f is ', max(abs(f)), " at ", kam, " (", f[kam], ")")

                if gamma != 1:
                    f -= (1 - gamma) * f0

                try:
                    dx = self._sparse_solver(J, -f)

                    if dx is None:
                        raise SparseSolverError
                        break
                    else:
                        dx.transpose()
                        # compute error
                        error = max(np.abs(dx))

                        if cc > maxiter:
                            msg = "**  Maximum number of iterations reached  **"
                            logging.error(msg)
                            kam = np.argmax(abs(dx))
                            print('max value ', error, " at ", kam, " (", dx[kam], ")")
                            fig = plt.figure()
                            X0 = system.xpts
                            ax = fig.add_subplot(111)
                            vx, vefn, vefp = f[2::3], f[0::3], f[1::3]
                            # l1, = ax.plot(X0 * 1e7, -system.bl - system.Eg - vx, lw=2, color='k', ls='-')
                            # l2, = ax.plot(X0 * 1e7, -system.bl - vx, lw=2, color='k', ls='-')
                            l1, = ax.plot(X0 * 1e7, vx, lw=2, color='k', ls='-')
                            l2, = ax.plot(X0 * 1e7, vefn, lw=2, color='#cf392e', ls='-')
                            l3, = ax.plot(X0 * 1e7, vefp, lw=2, color='#2e89cf', ls='-')
                            plt.title("error")
                            fig.legend([l1, l2, l3],
                                       [r'$\mathregular{V}$', r'$\mathregular{E_{fn}}$',
                                        r'$\mathregular{E_{fp}}$'])
                            plt.show()
                            fig = plt.figure()
                            X0 = system.xpts
                            ax = fig.add_subplot(111)
                            vx, vefn, vefp = dx[2::3], dx[0::3], dx[1::3]
                            # l1, = ax.plot(X0 * 1e7, -system.bl - system.Eg - vx, lw=2, color='k', ls='-')
                            # l2, = ax.plot(X0 * 1e7, -system.bl - vx, lw=2, color='k', ls='-')
                            l1, = ax.plot(X0 * 1e7, vx, lw=2, color='k', ls='-')
                            l2, = ax.plot(X0 * 1e7, vefn, lw=2, color='#cf392e', ls='-')
                            l3, = ax.plot(X0 * 1e7, vefp, lw=2, color='#2e89cf', ls='-')
                            plt.title("error")
                            fig.legend([l1, l2, l3],
                                       [r'$\mathregular{V}$', r'$\mathregular{E_{fn}}$',
                                        r'$\mathregular{E_{fp}}$'])
                            plt.show()

                            fig = plt.figure()
                            X0 = system.xpts
                            ax = fig.add_subplot(111)
                            vx, vefn, vefp = x[2::3], x[0::3], x[1::3]
                            vt = system.scaling.energy
                            l1, = ax.plot(X0, vt * vefn, lw=2, color='#2e89cf', ls='--')
                            l2, = ax.plot(X0, vt * vefp, lw=2, color='#cf392e', ls='--')
                            l3, = ax.plot(X0, -vt * (vx + system.bl), lw=2, color='k', ls='-')
                            l4, = ax.plot(X0, -vt * (vx + system.bl + system.Eg), lw=2, color='k', ls='-')
                            l4, = ax.plot(X0, -vt * (vx - vx[0]), lw=2, color='g', ls='--')
                            plt.title("stuck equilibrium")
                            plt.show()
                            break
                        if np.isnan(error) or error > 1e30:
                            kam = np.argmax(abs(dx))
                            print('max value ', error, " at ", kam, " (", dx[kam], ")")
                            fig = plt.figure()
                            X0 = system.xpts
                            ax = fig.add_subplot(111)
                            vx, vefn, vefp = f[2::3], f[0::3], f[1::3]

                            l1, = ax.plot(X0 * 1e7, vx, lw=2, color='k', ls='-')
                            l2, = ax.plot(X0 * 1e7, vefn, lw=2, color='#cf392e', ls='-')
                            l3, = ax.plot(X0 * 1e7, vefp, lw=2, color='#2e89cf', ls='-')
                            plt.title("error")
                            fig.legend([l1, l2, l3],
                                       [r'$\mathregular{V}$', r'$\mathregular{E_{fn}}$',
                                        r'$\mathregular{E_{fp}}$'])
                            plt.show()
                            raise NewtonError
                            break
                        if error < htol:
                            converged = True
                        else:
                            # damping and new value of x
                            # print(dx[system.eml_sites])
                            # dx[system.qd_sites[1]]
                            self._damping(dx)
                            x += dx
                        # print status of solution procedure
                        if verbose:
                            logging.info('step {0}, error = {1}'.format(cc, error))
                except SparseSolverError:
                    msg = "**  The linear system could not be solved  **"
                    logging.error(msg)
                    break

                except NewtonError:
                    msg = "**  The Newton-Raphson algorithm diverged, try a better guess or finer grid  **"
                    logging.error(msg)
                    fig = plt.figure()
                    X0 = system.xpts
                    ax = fig.add_subplot(111)
                    vx, vefn, vefp = x[2::3], x[0::3], x[1::3]
                    l1, = ax.plot(X0 * 1e7, -system.bl - system.Eg - vx, lw=2, color='k', ls='-')
                    l2, = ax.plot(X0 * 1e7, -system.bl - vx, lw=2, color='k', ls='-')
                    l3, = ax.plot(X0 * 1e7, vefn, lw=2, color='#cf392e', ls='-')
                    l4, = ax.plot(X0 * 1e7, vefp, lw=2, color='#2e89cf', ls='-')
                    plt.title("error")
                    fig.legend([l1, l2, l3, l4],
                               [r'$\mathregular{E_v}$', r'$\mathregular{E_c}$', r'$\mathregular{E_{fn}}$',
                                r'$\mathregular{E_{fp}}$'])
                    plt.show()

                    break

        if converged:
            return x
        else:
            return None

    def IVcurve(self, system, voltages, file_name, guess=None, tol=1e-6,
                periodic_bcs=True, maxiter=300, verbose=True, htp=1, fmt='npz'):
        """
        Solve the Drift Diffusion Poisson equations for the voltages provided. The
        results are stored in files with ``.npz`` format by default (See below for
        saving in Matlab format). The steady state current is computed at the
        end of the voltage loop and returned. Note that the
        potential is always applied on the right contact.

        Parameters
        ----------
        system: Builder
            The discretized system.
        voltages: array-like
            List of voltages for which the current should be computed.
        file_name: string
            Name of the file to write the data to. The file name will be appended
            the index of the voltage list, e.g. ``file_name_0.npz``.
        guess: dictionary of numpy arrays of floats (optional)
            Starting point of the solver. Keys of the dictionary must be 'efn',
            'efp', 'v' for the electron and quasi-Fermi levels, and the
            electrostatic potential respectively.
        tol: float
            Accepted error made by the Newton-Raphson scheme.
        periodic_bcs: boolean
            Defines the choice of boundary conditions in the y-direction. True
            (False) corresponds to periodic (abrupt) boundary conditions.
        maxiter: integer
            Maximum number of steps taken by the Newton-Raphson scheme.
        verbose: boolean
            The solver returns the step number and the associated error at every
            step, and this function prints the current applied voltage if set to True (default).
        htp: integer
            Number of homotopic Newton loops to perform.
        fmt: string
            Format string for the data files. Use ``mat`` to save the data in a
            Matlab format (version 5 and above).

        Returns
        -------
        J: numpy array of floats
            Steady state current computed for each voltage value.

        Notes
        -----
        The data files can be loaded and used as follows:

        >>> results = np.load('file.npz')
        >>> efn = results['efn']
        >>> efp = results['efp']
        >>> v = results['v']
        """
        # create a dictionary 'result' with efn and efp
        if guess is None:
            result = self.solve(system, compute='Poisson', tol=tol,
                                periodic_bcs=periodic_bcs, maxiter=maxiter,
                                verbose=verbose, htp=htp)
        else:
            result = guess
        # sites of the right contact
        nx = system.nx

        # qd2=system.qd_sites[1]
        rc = nx - 1
        # rc = range(qd2+20,nx)
        # print(rc)

        # sign of the voltage to apply
        if system.rho[nx - 1] < 0:
            q = 1
        else:
            q = -1

        # Solving equilbrium potential first
        if self.equilibrium is not None:
            if verbose:
                logging.info("Equilibrium potential already computed. Moving on.")
        else:
            self.solve(system, compute='Poisson', tol=tol,
                       periodic_bcs=periodic_bcs, maxiter=maxiter,
                       verbose=verbose, htp=htp)

        # Applied potentials made dimensionless
        Vapp = [i / system.scaling.energy for i in voltages]
        # Arrays of the steady-state terminal current, EQE proxy, and
        # emissive current. J and Jem are dimensionless current-equivalents;
        # multiply both by system.scaling.current to convert to A/cm^2.
        J = np.zeros((len(Vapp),))
        EQE = np.zeros((len(Vapp),))
        Jem = np.zeros((len(Vapp),))
        J[:] = np.nan
        EQE[:] = np.nan
        Jem[:] = np.nan
        r1 = result.copy()

        # v0=Vapp[0]
        # self.equilibrium=np.linspace(-system.contacts_WF[0] / system.scaling.energy, -system.contacts_WF[1] / system.scaling.energy+v0*q, system.nx)
        vapp0 = 0
        # nq1
        for idx, vapp in enumerate(Vapp):

            if verbose:
                logging.info("Applied voltage: {0} V".format(voltages[idx]))

            # Apply the voltage on the right contact
            if (voltages[idx] > 1 and idx>0):
                result['v'] = result['v'] + drv
                result['efn'] = result['efn'] + drfn
                result['efp'] = result['efp'] + drfp
            result['v'][rc] = self.equilibrium[rc] + q * vapp

            # Call the Drift Diffusion Poisson solver
            result = self.solve(system, guess=result, tol=tol, periodic_bcs=periodic_bcs, \
                                maxiter=maxiter, verbose=verbose, htp=htp)
            if result is None:
                logging.info("The solver failed to converge for the applied voltage"
                             + " {0} V (index {1}).".format(voltages[idx], idx))
                return J, EQE, Jem

            drv = result['v'] - r1['v']
            drfn = result['efn'] - r1['efn']
            drfp = result['efp'] - r1['efp']
            if result is not None:
                # 1. Save efn, efp, v
                name = file_name + "_{0}".format(idx)
                # add some system settings to the saved results
                if fmt == 'mat':
                    save_sim(system, result, name, fmt='mat')
                else:
                    filename = "%s.gzip" % name
                    save_sim(system, result, filename)
                # 2. Compute the steady state current
                try:
                    az = Analyzer(system, result)
                    J[idx] = az.full_current()
                    EQE[idx] = az.full_emission()
                    Jem[idx] = az.full_emissive_current()
                    r1 = result.copy()
                    logging.info(
                        "For {0} V, J = {1}, Jem = {2}, EQE = {3}.".format(
                            voltages[idx],
                            system.scaling.current * J[idx],
                            system.scaling.current * Jem[idx],
                            EQE[idx],
                        )
                    )
                except Exception:
                    logging.info("Could not compute the current for the applied voltage" \
                                 + " {0} V (index {1}).".format(voltages[idx], idx))

            else:
                logging.info("The solver failed to converge for the applied voltage" \
                             + " {0} V (index {1}).".format(voltages[idx], idx))
                return J, EQE, Jem
                break
        return J, EQE, Jem


default = Solver()
solve = default.solve
IVcurve = default.IVcurve
