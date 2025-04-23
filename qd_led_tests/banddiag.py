import sesame
import numpy as np
import matplotlib.pyplot as plt
sys, result = sesame.load_sim('p517_30/1dQD_V_140.gzip')
az = sesame.Analyzer(sys,result)
p1 = (0,0)
p2 = (120e-7,0)
fig = plt.figure()
az.band_diagram((p1,p2),fig=fig)
az.carrier_densities((p1,p2),fig=fig)
az.field_diagram()
az.current_diagram()
plt.show()
