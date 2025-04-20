import sesame
import numpy as np
import scipy.io
from scipy.io import savemat


t_hil = 20*1e-7
t_htl = 40e-7
t_etl = 40*1e-7
t_bqd = t_hil+t_htl
r_qdc = 2.75e-7
r_qds = 1e-7
r_qd = r_qdc+r_qds
t_aqd=t_hil+t_htl+4*(r_qd)
t_total=t_aqd+t_etl
print("t_total = ",t_total,"|  rqd = ",r_qd, "| taqd = ",t_aqd, "| tbqd = ", t_bqd)

# Heterojunctions require dense mesh near the interface
dd = 4e-7   # 2*dd is the distance over which mesh is refined
dd2 = 1.5e-7
# Define the mesh
x = np.concatenate((np.linspace(0, dd, 300, endpoint=False),                        # L contact interface
                    np.linspace(dd, t_hil-dd2, 100, endpoint=False),                    # material 1
                    np.linspace(t_hil - dd2, t_hil + dd2, 300, endpoint=False),             # interface 1
                    np.linspace(t_hil + dd2, (t_bqd) - dd, 200, endpoint=False),       # material 2
                    np.linspace((t_bqd) - dd, (t_bqd), 200, endpoint=False),      # htl-qd interface
                    [(t_bqd)+r_qd, (t_aqd)-r_qd],      # QD
                    np.linspace((t_aqd), (t_aqd) + dd, 200, endpoint=False),      # qd-etl interface
                    np.linspace((t_aqd) + dd, (t_total) - dd, 200, endpoint=False),       # material 2
                    np.linspace((t_total) - dd, (t_total), 300)))                       # R contact interface
# Build system
sys = sesame.Builder(x)
#qd_mnc, qd_mpc=.13,.45
qd_mnc, qd_mpc=.4,.4
#qd_mns, qd_mps=.19,.6
qd_mns, qd_mps=.4,.4
# CdS material dictionary
#hil = {'Nc': 2.5e19, 'Nv':2.5e19, 'Eg':1.57, 'epsilon':3, 'Et': 0,
#        'mu_e':0.000322, 'mu_h':0.000322, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
#        'affinity': 3.6}
hil = {'Nc': 2.5e19, 'Nv':2.5e19, 'Eg':1.9, 'epsilon':3, 'Et': 0,
        'mu_e':0.000322, 'mu_h':0.000322, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
        'affinity': 3.6}
htl = {'Nc': 2.5e19, 'Nv':2.5e19, 'Eg':3, 'epsilon':5, 'Et': 0,
        'mu_e':0.002, 'mu_h':0.002, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
        'affinity': 2.6}
qdc = {'Nc': 2.5e19*pow(qd_mnc,1.5), 'Nv':2.5e19*pow(qd_mpc,1.5), 'Eg':2.28, 'epsilon':9.4, 'Et': 0,
        'mu_e':0.000001, 'mu_h':0.000001, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
        'affinity': 3.6,'Cn':pow(10,-32),'Cp':pow(10,-32),'B':pow(10,-14)}
# CdTe material dictionary'''''' ''''''
'''etl = {'Nc': 2.5e19*pow(.24,1.5), 'Nv': 2.5e19*pow(.59,1.5), 'Eg':3.4, 'epsilon':5, 'Et': 0,
        'mu_e':0.002, 'mu_h':0.002, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
        'affinity': 4}
'''
etl = {'Nc': 2.5e19, 'Nv': 2.5e19, 'Eg':3.4, 'epsilon':5, 'Et': 0,
        'mu_e':0.002, 'mu_h':0.002, 'tau_e':1.2e-6, 'tau_h':1.2e-6,
        'affinity': 4}

# CdS region
hil_region = lambda x: x<=t_hil
htl_region = lambda x: np.logical_and(x>t_hil, x<=t_bqd)
qd_region = lambda x: np.logical_and(t_bqd<x, x<t_aqd)
etl_region = lambda x: (x>=t_aqd)

# Add the material to the system
sys.add_material(etl, etl_region)     # adding CdS
sys.add_material(htl, htl_region)     # adding CdTe
sys.add_material(hil, hil_region)     # adding CdTe
sys.add_material(qdc, qd_region)     # adding CdTe
sys.add_qd(1.0,qd_mns=qd_mns,qd_mps=qd_mps,location=qd_region)
# Add the donors
sys.add_donor(1e17, etl_region)
# Add the acceptors
sys.add_acceptor(1e17, htl_region)
sys.add_acceptor(5e18, hil_region)

# Define contacts: CdS contact is Ohmic, CdTe contact is Schottky
Lcontact_type, Rcontact_type = 'Schottky', 'Schottky'
#Lcontact_type, Rcontact_type = 'Ohmic', 'Ohmic'
Lcontact_workFcn, Rcontact_workFcn = 5.3, 4.06   # Lcontact work function irrelevant because L contact is Ohmic
# Add the contacts
sys.contact_type(Lcontact_type, Rcontact_type, Lcontact_workFcn, Rcontact_workFcn)

# Define the surface recombination velocities for electrons and holes [m/s]
Scontact = 1.16e4  # [cm/s]
# non-selective contacts
Sn_left, Sp_left, Sn_right, Sp_right = Scontact, Scontact, Scontact, Scontact
# This function specifies the simulation contact recombination velocity
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)


# Specify the applied voltage values
voltages = np.linspace(0,10,200)
# Perform I-V calculation
j = sesame.IVcurve(sys, voltages, 'rr2/1dQD_V',htp=1,maxiter=2000)
j = j * sys.scaling.current

result = {'v':voltages, 'j':j}
np.save('rr2/qd_IV_values', result)

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


