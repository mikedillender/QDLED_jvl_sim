import os
import sesame
import numpy as np
import matplotlib.pyplot as plt
folder="pohm_th"
f_name="1dQD_V_"
num_files=145
step=1
n=int(np.floor(num_files/step))+1
j=0

sys, result = sesame.load_sim(folder + "/" + f_name + "0.gzip")
v_len=len(result['v'])
v_ar=np.zeros([n,v_len])
xpt=sys.xpts
for i in range(0,num_files+1,step):
    this_file=folder+"/"+f_name+str(i)+".gzip"
    sys, result = sesame.load_sim(this_file)
    v_ar[j,:]=result['v']
    j=j+1

np.savetxt(folder+".csv", v_ar, delimiter=",", fmt="%.10f")
np.savetxt(folder+"_x.csv", xpt, delimiter=",", fmt="%.10f")

