import sesame
import numpy as np
import matplotlib.pyplot as plt

sys, result = sesame.load_sim('t_out4/1dQD_V_15.gzip')
az = sesame.Analyzer(sys,result)
p1 = (0,0)
p2 = (90e-7,0)
fig = plt.figure()
plt.grid()
az.band_diagram((p1,p2),fig=fig)

az.carrier_densities((p1,p2),fig=fig)

az.field_diagram()
plt.show()
