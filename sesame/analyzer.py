# Copyright 2017 University of Maryland.
#
# This file is part of Sesame. It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from scipy.interpolate import InterpolatedUnivariateSpline as spline, interp2d
import scipy.constants as cts
from .utils import Bresenham, get_indices
from .observables import *
from .defects import defectsF

try:
    import matplotlib.pyplot as plt

    mpl_enabled = True
    try:
        from mpl_toolkits import mplot3d

        has3d = True
    except:
        has3d = False
except:
    mpl_enabled = False


class Analyzer():
    """
    Object that simplifies the extraction of physical data (densities, currents,
    recombination) across the system.

    Parameters
    ----------

    sys: Builder
        A discretized system.
    data: dictionary
        Dictionary containing 1D arrays of electron and hole quasi-Fermi levels
        and the electrostatic potential across the system. Keys must be 'efn',
        'efp', and/or 'v'.
    """

    def __init__(self, sys, data):
        self.sys = sys
        self.v = data['v']

        # check for efn
        if 'efn' in data.keys():
            self.efn = data['efn']
        else:
            self.efn = 0 * self.v

        # check for efp
        if 'efp' in data.keys():
            self.efp = data['efp']
        else:
            self.efp = 0 * self.v

        # sites of the system
        self.sites = np.arange(sys.nx * sys.ny, dtype=int)

    @staticmethod
    def line(system, p1, p2):
        """
        Compute the path and sites between two points.

        Parameters
        ----------
        system: Builder
            The discretized system.
        p1, p2: array-like (x, y)
            Two points defining a line.

        Returns
        -------
        s, sites: numpay arrays
            Curvilinear abscissa and sites of the line.

        Notes
        -----
        This method can be used with an instance of the Analyzer():

        >>> az = sesame.Analyzer(sys, res)
        >>> X, sites = az.line(sys, p1, p2)

        or without it:

        >>> X, sites = sesame.Analyzer.line(sys, p1, p2)
        """

        p1 = (p1[0], p1[1], 0)
        p2 = (p2[0], p2[1], 0)
        s, x, _, _ = Bresenham(system, p1, p2)
        return x, s

    def band_diagram(self, location, fig=None):
        """
        Compute the band diagram between two points defining a line. Display a
        plot if fig is None.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute a band
            diagram.

        fig: Maplotlib figure
            A plot is added to it if given. If not given, a new one is created and
            displayed.

        """
        p1, p2 = location
        if self.sys.dimension == 1:
            idx1, _ = get_indices(self.sys, (p1[0], 0, 0))
            idx2, _ = get_indices(self.sys, (p2[0], 0, 0))
            X = self.sys.xpts[idx1:idx2]
            sites = np.arange(idx1, idx2, 1, dtype=int)
        if self.sys.dimension == 2:
            X, sites = self.line(self.sys, p1, p2)

        show = False
        if fig is None:
            fig = plt.figure()
            show = True

        # add axis to figure
        ax = fig.add_subplot(221)

        X = X * 1e7  # in um
        vt = self.sys.scaling.energy
        l1, = ax.plot(X, vt * self.efn[sites], lw=2, color='#2e89cf', ls='--')
        l2, = ax.plot(X, vt * self.efp[sites], lw=2, color='#cf392e', ls='--')
        l3, = ax.plot(X, -vt * (self.v[sites] + self.sys.bl[sites]), lw=2, color='k', ls='-')
        l4, = ax.plot(X, -vt * (self.v[sites] + self.sys.bl[sites] + self.sys.Eg[sites]), lw=2, color='k', ls='-')
        # print("chi :")
        # print(self.sys.bl[sites]*vt)
        # fig.legend([l1, l2], [r'$\mathregular{E_{F_n}}$',\
        #                      r'$\mathregular{E_{F_p}}$'])

        # ax.set_xlabel(r'Position [$\mathregular{nm}$]')
        ax.set_ylabel('Energy [eV]')

        # if show:
        #    plt.show()

    def saveCSV(self):
        X0 = self.sys.xpts  # *1e7
        # vt = self.sys.scaling.energy
        V = self.v  # *vt
        Ev = -self.sys.bl - self.sys.Eg  # *vt
        Nv = np.log(self.sys.Nv)
        efp = self.efp  # *self.sys.scaling.energy
        stacked = np.stack((X0, V, Ev, Nv, efp), axis=0)
        np.savetxt("p_data.csv", stacked, delimiter=",", fmt="%.10f")

    def field_diagram(self, fig=None):

        sites = self.sites
        X0 = self.sys.xpts
        X1 = self.sys.xpts[:-1] + self.sys.dx * self.sys.scaling.length / 2
        V = self.v * self.sys.scaling.energy
        dv = V[1:] - V[:-1]
        E = -dv / self.sys.dx
        V = V - V[0]
        show = False
        if fig is None:
            fig = plt.figure()
            show = True

        borders_x = np.array([X0[self.sys.eml_sites[0]], X0[self.sys.eml_sites[0]], X0[self.sys.eml_sites[3]],
                              X0[self.sys.eml_sites[3]]])
        borders_y = 1e6 * np.array([-1, 1, 1, -1])

        # add axis to figure
        ax = fig.add_subplot(234)
        ax.plot(borders_x * 1e7, borders_y, lw=1, color='#000000', ls='--')
        l1, = ax.plot(X0 * 1e7, V, lw=2, color='#2e89cf', ls='-')
        ax.set_ylim(min(V) - .1, max(V) + .1)
        ax.set_title(r'$\mathregular{V(x)}$')
        ax.set_xlabel(r'Position [$\mathregular{nm}$]')

        ax = fig.add_subplot(235)
        ax.plot(borders_x * 1e7, borders_y, lw=1, color='#000000', ls='--')
        l2, = ax.plot(X1 * 1e7, E, lw=2, color='#2e89cf', ls='-')
        ax.set_ylim(min(E) - .01, max(E) + .01)
        ax.set_title(r'$\mathregular{E(x)}$')
        ax.set_xlabel(r'Position [$\mathregular{nm}$]')

        ax = fig.add_subplot(236)
        p = self.hole_density()
        n = self.electron_density()
        rho = self.sys.rho - n + p
        ax.plot(borders_x * 1e7, borders_y, lw=1, color='#000000', ls='--')
        l3, = ax.plot(X0 * 1e7, rho, lw=2, color='#cf392e', ls='-')
        ax.set_ylim(min(rho) - .01, max(rho) + .01)
        ax.set_title(r'$\mathregular{\rho(x)}$')
        ax.set_xlabel(r'Position [$\mathregular{nm}$]')

        if show:
            plt.show()

    def current_diagram(self, fig=None):

        sites = self.sites[1:-2].flatten()
        dx = self.sys.dx[sites]
        X0 = self.sys.xpts[sites]
        X1 = X0 + (dx / 2) * self.sys.scaling.length

        show = False
        if fig is None:
            fig = plt.figure()
            show = True

        # add axis to figure
        ax = fig.add_subplot(111)

        X = X1 * 1e7  # in nm

        p1 = self.sys.scaling.current * get_jp(self.sys, self.efp, self.v, sites, sites + 1, dx)
        n1 = self.sys.scaling.current * get_jn(self.sys, self.efn, self.v, sites, sites + 1, dx)
        # print(n1)
        # print(p1)
        # print("current^")
        l1, = ax.plot(X, (n1), lw=2, color='#2e89cf', ls='-')
        l2, = ax.plot(X, (p1), lw=2, color='#cf392e', ls='-')

        fig.legend([l1, l2], [r'$\mathregular{n}$', r'$\mathregular{p}$'])

        ax.set_xlabel(r'Position [$\mathregular{nm}$]')

        if show:
            plt.show()

    def electron_density(self, location=None):
        """
        Compute the electron density across the system or on a line defined by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the electron
            density.

        Returns
        -------
        n: numpy array of floats

        See also
        --------
        hole_density

        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)
        n = get_n(self.sys, self.efn, self.v, sites)
        return n

    def hole_density(self, location=None):
        """
        Compute the hole density across the system or  on a line defined by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the hole
            density.

        Returns
        -------
        p: numpy array of floats

        See also
        --------
        electron_density

        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)
        p = get_p(self.sys, self.efp, self.v, sites)
        return p

    def total_charge(self):
        dx = self.sys.dx
        L = (dx[0:-2] + dx[1:-1]) / 2
        p = self.hole_density()
        n = self.electron_density()
        rho = self.sys.rho - n + p
        print(len(L), len(rho))
        trho = L * rho[1:-2]
        net = trho.sum()
        tot = abs(trho).sum()
        return [net, tot]

    def total_charge2(self):
        dx = self.sys.dx
        L = (dx[0:-2] + dx[1:-1]) / 2
        p = self.hole_density()
        n = self.electron_density()
        nrho = self.sys.rho - n + p
        nrho = L * nrho[1:-2]
        n = L * n[1:-2]
        p = L * p[1:-2]
        rho1 = L * self.sys.rho[1:-2]
        space1 = nrho[np.logical_and(rho1 > 0, abs(rho1) > n)].sum() - nrho[
            np.logical_and(rho1 + p < 0, rho1 < 0)].sum()
        carriers = nrho[np.logical_and(rho1 > 0, abs(rho1) < n)].sum() - nrho[
            np.logical_and(rho1 + p > 0, rho1 < 0)].sum() + nrho[rho1 == 0].sum()
        # trho = L * rho[1:-2]
        # net = trho.sum()
        # tot = abs(trho).sum()
        return [space1, carriers]

    def carrier_densities(self, location, fig=None):

        sites = self.sites
        X = self.sys.xpts[sites]
        # print("sites",sites,len(sites))
        # print("X",X,len(X))

        show = False
        if fig is None:
            fig = plt.figure()
            show = True

        # add axis to figure
        ax = fig.add_subplot(222)

        X = X * 1e7  # in nm

        p = self.sys.scaling.density * self.hole_density()
        n = self.sys.scaling.density * self.electron_density()
        rho = self.sys.scaling.density * self.sys.rho
        l1, = ax.plot(X, np.log10(n), lw=2, color='#2e89cf', ls='-')
        l2, = ax.plot(X, np.log10(p), lw=2, color='#cf392e', ls='-')
        l3, = ax.plot(X, np.log10(np.abs(rho) + pow(10, -30)), lw=2, color='k', ls='--')
        ax.set_ylim(ymin=10, ymax=20)

        # fig.legend([l1, l2], [r'$\mathregular{n}$', r'$\mathregular{p}$'])

        # ax.set_xlabel(r'Position [$\mathregular{nm}$]')

        if show:
            plt.show()

    def bulk_srh_rr(self, location=None):
        """
        Compute the bulk Shockley-Read-Hall recombination across the system or
        on a line defined by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            TUple of two points defining a line over which to compute the recombination.

        Returns
        -------
        r: numpy array
            An array with the values of recombination.
        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)
        p = get_p(self.sys, self.efp, self.v, sites)

        ni2 = self.sys.ni[sites] ** 2
        n1 = self.sys.n1[sites]
        p1 = self.sys.p1[sites]
        tau_h = self.sys.tau_h[sites]
        tau_e = self.sys.tau_e[sites]
        n = get_n(self.sys, self.efn, self.v, sites)
        p = get_p(self.sys, self.efp, self.v, sites)
        r = (n * p - ni2) / (tau_h * (n + n1) + tau_e * (p + p1))
        return r

    def auger_rr(self, location=None):
        """
        Compute the Auger recombination across the system or on a line defined
        by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the recombination.

        Returns
        -------
        r: numpy array
            An array with the values of recombination.
        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)

        n = get_n(self.sys, self.efn, self.v, sites)
        p = get_p(self.sys, self.efp, self.v, sites)
        ni2 = self.sys.ni[sites] ** 2
        r = self.sys.Cn[sites] * n * (n * p - ni2) + self.sys.Cp[sites] * p * (n * p - ni2)
        return r

    def radiative_rr(self, location=None):
        """
        Compute the radiative recombination across the system or on a line defined by two points.

        Parameters
        ----------
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the recombination.

        Returns
        -------
        r: numpy array
            An array with the values of recombination.
        """
        if location is None:
            sites = self.sites
        else:
            p1, p2 = location
            _, sites = self.line(self.sys, p1, p2)

        n = get_n(self.sys, self.efn, self.v, sites)
        p = get_p(self.sys, self.efp, self.v, sites)
        ni2 = self.sys.ni[sites] ** 2
        r = self.sys.B[sites] * (n * p - ni2)
        return r

    def defect_rr(self, defect):
        """
        Compute the recombination for all sites of a defect (2D and 3D).

        Parameters
        ----------
        defect: named tuple
            Container with the properties of a defect. The expected field names
            of the named tuple are sites, location, dos, energy, sigma_e,
            sigma_h, transition, perp_dl.

        Returns
        -------
        r: numpy array of floats
            An array with the values of recombination at each sites.
        """

        # Create arrays to pass to defectsF
        n = self.electron_density()
        p = self.hole_density()
        rho = np.zeros_like(n)
        r = np.zeros_like(n)

        # Update r (and rho but we don't use it)
        defectsF(self.sys, [defect], n, p, rho, r=r)
        r = np.multiply(r[defect.sites], defect.perp_dl)

        return r

    def total_rr(self):
        """
        Compute the sum of all the recombination sources for all sites of the
        system.

        Returns
        -------
        r: numpy array of floats
            An array with the values of the total recombination at each sites.
        """

        srh = self.bulk_srh_rr()
        radiative = self.radiative_rr()
        auger = self.auger_rr()
        defects = np.zeros_like(srh)
        for defect in self.sys.defects_list:
            sites = defect.sites
            defects[sites] += self.defect_rr(defect)

        return srh + radiative + auger + defects

    def electron_current(self, component='x', location=None):
        """
        Compute the electron current either by component (x or y) across the
        entire system, or on a line defined by two points.

        Parameters
        ----------
        component: string
            Current direction ``'x'`` or ``'y'``. By default returns all currents
            in the x-direction.
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the electron
            current.

        Returns
        -------
        jn: numpy array of floats
        """

        if location is not None:
            p1, p2 = location
            X, sites = self.line(self.sys, p1, p2)
            jn = get_jn(self.sys, self.efn, self.v, sites[:-1], sites[1:], X[1:] - X[:-1])
        else:
            Nx, Ny = self.sys.nx, self.sys.ny
            sites = self.sites.reshape(Ny, Nx)
            if component == 'x':
                sites = sites[:Ny, :Nx - 1].flatten()
                dx = np.tile(self.sys.dx, Ny)
                jn = get_jn(self.sys, self.efn, self.v, sites, sites + 1, dx)
            if component == 'y':
                sites = sites[:Ny - 1, :Nx].flatten()
                dy = np.repeat(self.sys.dy, Nx)
                jn = get_jn(self.sys, self.efn, self.v, sites, sites + Nx, dy)
        return jn

    def hole_current(self, component='x', location=None):
        """
        Compute the hole current either by component (x or y) across the entire
        system, or on a line defined by two points.

        Parameters
        ----------
        component: string
            Current direction ``'x'`` or ``'y'``. By default returns all currents
            in the x-direction.
        location: array-like ((x1,y1), (x2,y2))
            Tuple of two points defining a line over which to compute the hole
            current.

        Returns
        -------
        jp: numpy array of floats
        """

        if location is not None:
            p1, p2 = location
            X, sites = self.line(self.sys, p1, p2)
            jp = get_jp(self.sys, self.efp, self.v, sites[:-1], sites[1:], X[1:] - X[:-1])
        else:
            Nx, Ny = self.sys.nx, self.sys.ny
            sites = self.sites.reshape(Ny, Nx)
            if component == 'x':
                sites = sites[:Ny, :Nx - 1].flatten()
                dx = np.tile(self.sys.dx, Ny)
                jp = get_jp(self.sys, self.efp, self.v, sites, sites + 1, dx)
            if component == 'y':
                sites = sites[:Ny - 1, :Nx].flatten()
                dy = np.repeat(self.sys.dy, Nx)
                jp = get_jp(self.sys, self.efp, self.v, sites, sites + Nx, dy)
        return jp

    def electron_current_map(self, cmap='gnuplot', scale=1e4):
        """
        Compute a 2D map of the electron current.

        Parameters
        ----------
        cmap: Matplotlib color map
            Color map used for the plot.
        scale: float
            Scale to apply to the axes of the plot.

        """
        self.current_map(True, cmap, scale)

    def hole_current_map(self, cmap='gnuplot', scale=1e4):
        """
        Compute a 2D map of the hole current of a 2D system.

        Parameters
        ----------
        cmap: Matplotlib color map
            Color map used for the plot.
        scale: float
            Scale to apply to the axes of the plot.

        """
        self.current_map(False, cmap, scale)

    def current_map(self, electron, cmap, scale, fig=None):

        if not mpl_enabled:
            raise RuntimeError("matplotlib was not found, but is required "
                               "for plotting.")

        show = False
        if fig is None:
            fig = plt.figure()
            show = True

        # add axis to figure
        ax = fig.add_subplot(111)

        Lx = self.sys.xpts[-2] * scale
        Ly = self.sys.ypts[-2] * scale

        x, y = self.sys.xpts[:-1], self.sys.ypts[:-1]
        nx, ny = len(x), len(y)

        s = np.asarray([i + j * self.sys.nx for j in range(self.sys.ny - 1) \
                        for i in range(self.sys.nx - 1)])
        dx = np.tile(self.sys.dx, ny)
        dy = np.repeat(self.sys.dy[:-1], nx)

        if electron:
            Jx = get_jn(self.sys, self.efn, self.v, s, s + 1, dx)
            Jy = get_jn(self.sys, self.efn, self.v, s, s + (nx + 1), dy)
            title = r'$\mathregular{J_{n}\ [mA\cdot cm^{-2}]}$'
        else:
            Jx = get_jp(self.sys, self.efp, self.v, s, s + 1, dx)
            Jy = get_jp(self.sys, self.efp, self.v, s, s + (nx + 1), dy)
            title = r'$\mathregular{J_{p}\ [mA\cdot cm^{-2}]}$'

        Jx = np.reshape(Jx, (ny, nx)) * self.sys.scaling.current * 1e3
        Jy = np.reshape(Jy, (ny, nx)) * self.sys.scaling.current * 1e3

        jx = interp2d(x * scale, y * scale, Jx, kind='linear')
        jy = interp2d(x * scale, y * scale, Jy, kind='linear')

        xx, yy = np.linspace(0, Lx, 100), np.linspace(0, Ly, 100)
        jnx, jny = jx(xx, yy), jy(xx, yy)
        norm = np.sqrt(jnx ** 2 + jny ** 2)

        y, x = np.mgrid[0:Ly:100j, 0:Lx:100j]
        p = ax.pcolor(x, y, norm, cmap=cmap, rasterized=True)
        cbar = fig.colorbar(p, ax=ax)

        ax.streamplot(x, y, jnx, jny, linewidth=1, color='#a9a9a9', density=2)
        ax.set_xlim(xmax=Lx, xmin=0)
        ax.set_ylim(ymin=0, ymax=Ly)

        ax.set_xlabel(r'x [$\mathregular{\mu m}$]')
        ax.set_ylabel(r'y [$\mathregular{\mu m}$]')
        ax.set_title(title)

        if show:
            plt.show()

    def map3D(self, data, cmap='gnuplot', scale=1e-6):
        """
        Plot a 3D map of data across the entire system.

        Parameters
        ----------

        data: numpy array
            One-dimensional array of data with size equal to the size of the system.
        cmap: string
            Name of the colormap used by Matplolib.
        scale: float
            Relevant scaling to apply to the axes.
        """

        if not mpl_enabled:
            raise RuntimeError("matplotlib was not found, but is required "
                               "for map3D()")
        if not has3d:
            raise RuntimeError("Installed matplotlib does not support 3d plotting")

        xpts, ypts = self.sys.xpts / scale, self.sys.ypts / scale
        nx, ny = len(xpts), len(ypts)
        data_xy = data.reshape(ny, nx).T
        X, Y = np.meshgrid(xpts, ypts)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        Z = data_xy.T
        ax.plot_surface(X, Y, Z, cmap=cmap)
        ax.mouse_init(rotate_btn=1, zoom_btn=3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def integrated_bulk_srh_recombination(self):
        """
        Integrate the bulk Shockley-Read-Hall recombination over an entire
        system.

        Returns
        -------
        JR: float
            The integrated bulk recombination.

        Warnings
        --------
        Not implemented in 3D.
        """
        return self.integrated_recombination('srh')

    def integrated_auger_recombination(self):
        """
        Integrate the Auger recombination over an entire system.

        Returns
        -------
        JR: float
            The integrated Auger recombination.

        Warnings
        --------
        Not implemented in 3D.
        """
        return self.integrated_recombination('auger')

    def integrated_radiative_recombination(self):
        """
        Integrate the radiative recombination over an entire system.

        Returns
        -------
        JR: float
            The integrated radiative recombination.

        Warnings
        --------
        Not implemented in 3D.
        """
        return self.integrated_recombination('radiative')

    def integrated_recombination(self, mec):
        # Compute recombination averywhere
        if mec == 'srh':
            r = self.bulk_srh_rr()
        if mec == 'auger':
            r = self.auger_rr()
        if mec == 'radiative':
            r = self.radiative_rr()

        # Integrate along x for each y (if any
        x = self.sys.xpts / self.sys.scaling.length
        if self.sys.ny > 1:
            y = self.sys.ypts / self.sys.scaling.length
        u = []
        for j in range(self.sys.ny):
            # List of sites
            s = [i + j * self.sys.nx for i in range(self.sys.nx)]
            sp = spline(x, r[s])
            u.append(sp.integral(x[0], x[-1]))
        if self.sys.ny == 1:
            JR = u[-1]
        if self.sys.ny > 1:
            sp = spline(y, u)
            JR = sp.integral(y[0], y[-1])
        return JR

    def integrated_defect_recombination(self, defect):
        """
        Integrate the recombination along a defect in 2D.

        Returns
        -------
        JD: float
            The recombination integrated along the line of the defect.

        Warnings
        --------
        Not implemented in 3D.
        """
        # Find the path along which to integrate
        p1 = defect.location[0]
        p2 = defect.location[1]
        X, _ = self.line(self.sys, p1, p2)

        # interpolate recombination and integrate
        r = self.defect_rr(defect)
        sp = spline(X, r)
        JD = sp.integral(X[0], X[-1])

        return JD

    def full_current(self):
        """
        Compute the steady state current in 1D and 2D.

        Returns
        -------
        J: float
            The integrated full steady state current.
        """

        # System number of sites
        nx, ny = self.sys.nx, self.sys.ny

        # Define the sites between which computing the currents
        sites_i = [nx // 2 + j * nx for j in range(ny)]
        sites_ip1 = [nx // 2 + 1 + j * nx for j in range(ny)]
        # And the corresponding lattice dimensions
        dl = self.sys.dx[self.sys.nx // 2]
        # print(sites_i)

        # Compute the electron and hole currents
        jn = get_jn(self.sys, self.efn, self.v, sites_i, sites_ip1, dl)
        jp = get_jp(self.sys, self.efp, self.v, sites_i, sites_ip1, dl)

        if ny == 1:
            j = jn[0] + jp[0]
        if ny > 1:
            # Interpolate the results and integrate over the y-direction
            y = self.sys.ypts / self.sys.scaling.length
            j = spline(y, jn + jp).integral(y[0], y[-1])

        return j

    def _emission_sites(self, sites=None):
        """Return the sites used for LED emission accounting.

        For QD-LED simulations, emission should only be counted from the
        discrete QD sites.  Falling back to ``eml_sites`` preserves the old
        behavior for non-QD simulations.
        """
        if sites is not None:
            return np.asarray(sites, dtype=int)
        if getattr(self.sys, 'has_qd', False) and hasattr(self.sys, 'qd_sites'):
            return np.asarray(self.sys.qd_sites, dtype=int)
        return np.asarray(self.sys.eml_sites, dtype=int)

    def _site_widths(self, sites):
        """Continuity-equation control-volume widths for site-local rates.

        For the continuity equations Sesame discretizes a local rate term R_i as

            (J_i - J_{i-1}) / dxbar_i - R_i

        where dxbar_i = (dx_{i-1} + dx_i)/2 for an interior site.  Therefore,
        to convert a site-local recombination rate into a current-equivalent
        flux that is consistent with f_n/f_p, the rate must be multiplied by
        this continuity control-volume width.

        This deliberately does *not* use ``poisson_charge_width``.  The Poisson
        charge volume was introduced only to integrate charge density in f_v;
        it is not the volume used by the carrier-continuity equations.
        """
        sites = np.asarray(sites, dtype=int)
        widths = np.empty(len(sites), dtype=float)
        nx = self.sys.nx

        for k, i in enumerate(sites):
            if i <= 0:
                widths[k] = 0.5 * self.sys.dx[0]
            elif i >= nx - 1:
                widths[k] = 0.5 * self.sys.dx[-1]
            else:
                widths[k] = 0.5 * (self.sys.dx[i - 1] + self.sys.dx[i])
        return widths

    def _qd_boundary_currents(self):
        """Return QD-region boundary currents from the link-current arrays.

        These are diagnostics only.  The EQE/emissive-current calculation below
        uses the recombination term from the continuity equation, not these
        boundary-current differences.  In a well-converged steady-state run,
        both approaches should agree closely.
        """
        nan_out = {
            'jp_left_qd': np.nan,
            'jp_right_qd': np.nan,
            'jn_left_qd': np.nan,
            'jn_right_qd': np.nan,
            'jrec_holes_raw': np.nan,
            'jrec_electrons_raw': np.nan,
            'jrec_qd_raw': np.nan,
            'jrec_qd_flux': np.nan,
        }
        if not (getattr(self.sys, 'has_qd', False) and hasattr(self.sys, 'qd_sites')):
            return nan_out

        nx = self.sys.nx
        if nx < 2:
            return nan_out

        sites_i = np.arange(nx - 1, dtype=int)
        sites_ip1 = sites_i + 1
        dl = self.sys.dx

        jp = get_jp(self.sys, self.efp, self.v, sites_i, sites_ip1, dl)
        jn = get_jn(self.sys, self.efn, self.v, sites_i, sites_ip1, dl)

        qd_sites = np.asarray(self.sys.qd_sites, dtype=int)
        left_link = int(qd_sites[0] - 1)  # HTL site -> first QD site
        right_link = int(qd_sites[-1])  # last QD site -> ETL site

        if not (0 <= left_link < len(jp) and 0 <= right_link < len(jp)):
            return nan_out

        jp_left = float(jp[left_link])
        jp_right = float(jp[right_link])
        jn_left = float(jn[left_link])
        jn_right = float(jn[right_link])

        jrec_h = jp_left - jp_right
        jrec_n = jn_right - jn_left

        if np.isfinite(jrec_h) and np.isfinite(jrec_n):
            jrec_raw = 0.5 * (jrec_h + jrec_n)
        elif np.isfinite(jrec_h):
            jrec_raw = jrec_h
        elif np.isfinite(jrec_n):
            jrec_raw = jrec_n
        else:
            jrec_raw = np.nan

        jrec_flux = max(float(jrec_raw), 0.0) if np.isfinite(jrec_raw) else np.nan

        return {
            'jp_left_qd': jp_left,
            'jp_right_qd': jp_right,
            'jn_left_qd': jn_left,
            'jn_right_qd': jn_right,
            'jrec_holes_raw': float(jrec_h),
            'jrec_electrons_raw': float(jrec_n),
            'jrec_qd_raw': float(jrec_raw) if np.isfinite(jrec_raw) else np.nan,
            'jrec_qd_flux': jrec_flux,
        }

    def _qd_injection_fluxes(self):
        """Backward-compatible wrapper for injection diagnostics."""
        c = self._qd_boundary_currents()
        jp_inj = c['jp_left_qd']
        jn_inj = c['jn_right_qd']
        if np.isfinite(jp_inj) and np.isfinite(jn_inj):
            bipolar_inj = min(abs(jp_inj), abs(jn_inj))
        else:
            bipolar_inj = np.nan
        return jp_inj, jn_inj, bipolar_inj

    def emission_components(self, eta_out=None, J_floor=None, sites=None,
                            clip_negative=True):
        """Return EQE/emissive-current quantities from the continuity rates.

        This implements the definition you described:

            J_rec,QD = integral_QD[(R_rad + R_SRH + R_Auger) dxbar]
            eta_rad  = integral_QD[R_rad dxbar] / J_rec,QD
            J_em     = eta_rad * J_rec,QD

        which is exactly equivalent to ``J_em = integral_QD[R_rad dxbar]``, but
        it makes the physical interpretation explicit.  The important detail is
        that ``dxbar`` is the control-volume width from the *continuity*
        equation f_n/f_p, not the Poisson charge-control volume.

        Currents/fluxes returned here are in Sesame's dimensionless internal
        current units unless the key name contains ``physical_Acm2``.  Multiply
        by ``sys.scaling.current`` to convert to A/cm^2.
        """
        if eta_out is None:
            eta_out = getattr(self.sys, 'eqe_outcoupling', .25)
        if J_floor is None:
            J_floor = getattr(self.sys, 'eqe_current_floor', 1e-10)

        sites = self._emission_sites(sites)

        n = get_n(self.sys, self.efn, self.v, sites)
        p = get_p(self.sys, self.efp, self.v, sites)
        ni2 = self.sys.ni[sites] ** 2
        np_minus_ni2 = n * p - ni2

        r_rad = self.sys.B[sites] * np_minus_ni2
        r_aug = (self.sys.Cn[sites] * n + self.sys.Cp[sites] * p) * np_minus_ni2
        r_srh = np_minus_ni2 / (
                self.sys.tau_h[sites] * (n + self.sys.n1[sites])
                + self.sys.tau_e[sites] * (p + self.sys.p1[sites])
        )

        if clip_negative:
            r_rad_i = np.maximum(r_rad, 0.0)
            r_aug_i = np.maximum(r_aug, 0.0)
            r_srh_i = np.maximum(r_srh, 0.0)
        else:
            r_rad_i, r_aug_i, r_srh_i = r_rad, r_aug, r_srh

        widths = self._site_widths(sites)

        # Current-equivalent recombination fluxes from the same source terms
        # that appear in the continuity equations.
        rad_flux = float(np.sum(r_rad_i * widths))
        auger_flux = float(np.sum(r_aug_i * widths))
        srh_flux = float(np.sum(r_srh_i * widths))
        qd_recomb_current_flux = rad_flux + auger_flux + srh_flux

        if qd_recomb_current_flux > 0 and np.isfinite(qd_recomb_current_flux):
            radiative_branching_ratio = rad_flux / qd_recomb_current_flux
        else:
            radiative_branching_ratio = 0.0

        # This is written in the requested form, although algebraically it is
        # just rad_flux when the same widths are used above.
        emissive_current_flux = radiative_branching_ratio * qd_recomb_current_flux

        terminal_current = float(self.full_current())  # dimensionless, signed
        current_flux = abs(terminal_current)
        current_physical = current_flux * self.sys.scaling.current
        emissive_current_physical = emissive_current_flux * self.sys.scaling.current

        if (not np.isfinite(current_flux)) or current_physical < J_floor or current_flux <= 0:
            iqe_current = 0.0
            eqe = 0.0
        else:
            iqe_current = emissive_current_flux / current_flux
            eqe = eta_out * iqe_current

        recomb_fraction_of_current = (
            qd_recomb_current_flux / current_flux if current_flux > 0 else 0.0
        )
        leakage_flux = current_flux - qd_recomb_current_flux

        qd_currents = self._qd_boundary_currents()
        jp_inj, jn_inj, bipolar_injection_flux = self._qd_injection_fluxes()

        out = {
            'eqe': eqe,
            'iqe_current': iqe_current,
            'eta_out': eta_out,
            'radiative_branching_ratio': radiative_branching_ratio,
            'recomb_fraction_of_current': recomb_fraction_of_current,
            'current_physical_Acm2': current_physical,
            'current_flux': current_flux,
            'total_current': terminal_current,
            'emissive_current_flux': emissive_current_flux,
            'emissive_current_physical_Acm2': emissive_current_physical,
            'qd_recomb_current_flux': qd_recomb_current_flux,
            'qd_recomb_current_physical_Acm2': qd_recomb_current_flux * self.sys.scaling.current,
            'raw_radiative_current_flux': rad_flux,
            'raw_radiative_current_physical_Acm2': rad_flux * self.sys.scaling.current,
            'rad_flux': rad_flux,
            'auger_flux': auger_flux,
            'srh_flux': srh_flux,
            'total_recomb_flux': qd_recomb_current_flux,
            'leakage_flux': leakage_flux,
            'jp_qd_injection': jp_inj,
            'jn_qd_injection': jn_inj,
            'bipolar_injection_flux': bipolar_injection_flux,
            'r_rad': r_rad,
            'r_aug': r_aug,
            'r_srh': r_srh,
            'sites': sites,
            'widths': widths,
        }
        out.update(qd_currents)
        return out

    def full_emissive_current(self, physical=False):
        """Return the QD radiative recombination current-equivalent.

        Parameters
        ----------
        physical : bool
            If False, return the dimensionless current-equivalent used internally
            by Sesame. If True, return A/cm^2 by multiplying by
            ``sys.scaling.current``.
        """
        c = self.emission_components(J_floor=0.0)
        if physical:
            return c['emissive_current_physical_Acm2']
        return c['emissive_current_flux']

    def full_emission(self, eta_out=None, J_floor=None):
        """Return the EQE proxy used by IVcurve.

        This is QD-radiative-recombination current divided by total terminal
        current, with both quantities in Sesame's dimensionless
        current-equivalent units. No explicit factor of q is needed.
        """
        return self.emission_components(eta_out=eta_out, J_floor=J_floor)['eqe']

    def print_emission(self):
        c = self.emission_components()
        print("EQE proxy:", c['eqe'])
        print("IQE/current proxy:", c['iqe_current'])
        print("Radiative branching ratio:", c['radiative_branching_ratio'])
        print("Current [A/cm^2]:", c['current_physical_Acm2'])
        print("Emissive current [A/cm^2]:", c['emissive_current_physical_Acm2'])
        print("Raw QD radiative flux:", c['rad_flux'])
        print("Auger flux:", c['auger_flux'])
        print("SRH flux:", c['srh_flux'])
        print("Total recombination flux:", c['total_recomb_flux'])
        print("Leakage flux:", c['leakage_flux'])
        print("HTL->QD hole injection:", c['jp_qd_injection'])
        print("ETL->QD electron injection:", c['jn_qd_injection'])
        print("Bipolar injection flux:", c['bipolar_injection_flux'])
        return c['eqe']
