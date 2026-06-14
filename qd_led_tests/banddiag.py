import sesame
import numpy as np
import matplotlib.pyplot as plt
#sys, result = sesame.load_sim('qd_small/1dQD_V_226.gzip')
sys, result = sesame.load_sim('paperlike2/1dQD_V_200.gzip')
az = sesame.Analyzer(sys,result)
p1 = (0,0)
p2 = (150e-7,0)
fig = plt.figure()
az.print_emission()
az.band_diagram((p1,p2),fig=fig)
az.carrier_densities((p1,p2),fig=fig)
az.field_diagram(fig=fig)
az.current_diagram()
plt.show()
#az.saveCSV()
