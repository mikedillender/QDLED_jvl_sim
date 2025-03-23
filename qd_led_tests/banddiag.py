import sesame
import numpy as np
import matplotlib.pyplot as plt

sys, result = sesame.load_sim('t_out/1dQD_V_29.gzip')
az = sesame.Analyzer(sys,result)
p1 = (0,0)
p2 = (80e-7,0)
fig = plt.figure()
az.band_diagram((p1,p2),fig=fig)

az.carrier_densities((p1,p2),fig=fig)
plt.show()
