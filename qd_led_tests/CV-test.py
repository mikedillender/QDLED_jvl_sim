import os
import sesame
import numpy as np
import matplotlib.pyplot as plt
folder="compare_m_qd/m2"
f_name="1dQD_V_"
num_files=99
step=1
n=int(np.floor(num_files/step))+1
pcharge=np.zeros(n)
ncharge=np.zeros(n)
volt=np.zeros(n)
j=0
for i in range(0,num_files+1,step):
    this_file=folder+"/"+f_name+str(i)+".gzip"
    sys, result = sesame.load_sim(this_file)
    az = sesame.Analyzer(sys,result)
    #net, tot=az.total_charge()
    #pcharge[j]=(tot+net)/2
    #ncharge[j]=tot-pcharge[j]
    net, tot=az.total_charge2()
    pcharge[j]=net
    ncharge[j]=tot
    volt[j]=i
    j=j+1
volt=volt[0:j-1]*(10/200)
pcharge=pcharge[0:j-1]
ncharge=ncharge[0:j-1]

fig = plt.figure()
ax = fig.add_subplot(211)
l1, = ax.plot(volt,pcharge, lw=2, color='#2e89cf', ls='-')
l2, = ax.plot(volt,ncharge, lw=2, color='#cf392e', ls='--')
plt.title("charge")

pcharge=pcharge+ncharge
voltc=(volt[0:-2]+volt[1:-1])/2
cap=pcharge[1:-1]-pcharge[0:-2]

ax = fig.add_subplot(212)
l3, = ax.plot(voltc,-cap, lw=2, color='#2e89cf', ls='-')
plt.title("capacitance")
plt.show()
'''
sys, result = sesame.load_sim('p517_30/1dQD_V_140.gzip')
az = sesame.Analyzer(sys,result)
p1 = (0,0)
p2 = (120e-7,0)
fig = plt.figure()
az.band_diagram((p1,p2),fig=fig)
az.carrier_densities((p1,p2),fig=fig)
az.field_diagram(fig=fig)
az.current_diagram()
plt.show()
az.saveCSV()'''
