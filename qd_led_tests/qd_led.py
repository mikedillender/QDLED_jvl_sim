import sesame
import numpy as np
import scipy.io
from scipy.io import savemat


t_hil = 20*1e-7
t_htl = 20e-7
t_etl = 40*1e-7

# Heterojunctions require dense mesh near the interface
dd = 1.5e-7   # 2*dd is the distance over which mesh is refined
# Define the mesh
x = np.concatenate((np.linspace(0, dd, 20, endpoint=False),                        # L contact interface
                    np.linspace(dd, t_hil-dd, 40, endpoint=False),                    # material 1
                    np.linspace(t_hil - dd, t_hil + dd, 20, endpoint=False),             # interface 1
                    np.linspace(t_hil + dd, (t_hil+t_htl) - dd, 40, endpoint=False),       # material 2
                    np.linspace((t_hil+t_htl) - dd, (t_hil+t_htl) + dd, 20, endpoint=False),      # htl-etl interface
                    np.linspace((t_hil+t_htl) + dd, (t_hil+t_htl+t_etl) - dd, 80, endpoint=False),       # material 2
                    np.linspace((t_hil+t_htl+t_etl) - dd, (t_hil+t_htl+t_etl), 100)))                       # R contact interface

# Build system
sys = sesame.Builder(x)

# CdS material dictionary
hil = {'Nc': 2.5e19, 'Nv':2.5e19, 'Eg':1.57, 'epsilon':3, 'Et': 0,
        'mu_e':0.000322, 'mu_h':0.000322, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
        'affinity': 3.6}
htl = {'Nc': 2.5e18, 'Nv':2.5e19, 'Eg':3, 'epsilon':3.5, 'Et': 0,
        'mu_e':0.002, 'mu_h':0.002, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
        'affinity': 2.6}
qdc = {'Nc': 2.5e19*pow(.13,1.5), 'Nv':2.5e19*pow(.45,1.5), 'Eg':2.28, 'epsilon':9.4, 'Et': 0,
        'mu_e':0.000002, 'mu_h':0.000001, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
        'affinity': 3.6}
# CdTe material dictionary
etl = {'Nc': 2.5e19*pow(.24,1.5), 'Nv': 2.5e19*pow(.59,1.5), 'Eg':3.4, 'epsilon':8.5, 'Et': 0,
        'mu_e':0.002, 'mu_h':0.002, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
        'affinity': 4}

# CdS region
hil_region = lambda x: x<=t_hil
# CdTe region
htl_region = lambda x: np.logical_and(x>t_hil, x<=t_hil+t_htl)
etl_region = lambda x: (x>t_hil+t_htl)

# Add the material to the system
sys.add_material(etl, etl_region)     # adding CdS
sys.add_material(htl, htl_region)     # adding CdTe
sys.add_material(hil, hil_region)     # adding CdTe

# Add the donors
sys.add_donor(2.81e19, etl_region)
# Add the acceptors
sys.add_acceptor(1e17, htl_region)
sys.add_acceptor(1e17, hil_region)

# Define contacts: CdS contact is Ohmic, CdTe contact is Schottky
Lcontact_type, Rcontact_type = 'Schottky', 'Schottky'
Lcontact_workFcn, Rcontact_workFcn = 4.7, 4.06   # Lcontact work function irrelevant because L contact is Ohmic
# Add the contacts
sys.contact_type(Lcontact_type, Rcontact_type, Lcontact_workFcn, Rcontact_workFcn)

# Define the surface recombination velocities for electrons and holes [m/s]
Scontact = 1.16e4  # [cm/s]
# non-selective contacts
Sn_left, Sp_left, Sn_right, Sp_right = Scontact, Scontact, Scontact, Scontact
# This function specifies the simulation contact recombination velocity
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)


# Specify the applied voltage values
voltages = np.linspace(0,3,10)
# Perform I-V calculation
j = sesame.IVcurve(sys, voltages, '1dQD_V')
j = j * sys.scaling.current

result = {'v':voltages, 'j':j}
np.save('qd_IV_values', result)

# plot I-V curve
try:
    import matplotlib.pyplot as plt
    plt.plot(voltages, j,'-o')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A/cm^2]')
    plt.grid()     # add grid
    plt.show()     # show the plot on the screen
# no matplotlib installed
except ImportError:
    print("Matplotlib not installed, can't make plot")


