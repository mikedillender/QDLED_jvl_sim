import matplotlib; matplotlib.use('Agg', force=True)
import sys; sys.path.insert(0,'/tmp')
import numpy as np
import thesis_style as ts
import matplotlib.pyplot as plt
OUT='/sessions/sleepy-adoring-pasteur/mnt/Numerical_QDLED/thesis_figures'
q=1.602176634e-19; h=6.62607015e-34; cc=2.998e8; Km=683.0; lam=545e-9; Vl=0.984
def lum(j,eqe):
    J=np.abs(j)*1e4; E=np.clip(eqe,0,None); g=np.abs(j)>1e-6
    return Km*Vl*(h*cc/lam)*np.where(g,J,0)*np.where(g,E,0)/(q*np.pi)
def load(f):
    d=np.load(f); V=d['V']; j=np.abs(d['j']); m=np.isfinite(j); return V[m],j[m],d['eqe'][m]

fig,axes=plt.subplots(2,3,figsize=(15,8.2),constrained_layout=True)
# ---------- (a,b) tau sweep, C=0 ----------
taus=[('0.5','0.5'),('1.5','1.5'),('5','5.0'),('10','10.0')]
grays=plt.cm.Greys(np.linspace(0.45,0.95,len(taus)))
axA=axes[0,0]; axAr=axA.twinx()
for (fn,lab),g in zip(taus,grays):
    V,j,e=load('/tmp/f3tau_%s.npz'%fn); axA.plot(V,j,'-',color=g,lw=1.6,label=lab)
    axAr.semilogy(V,np.clip(lum(j,e),1e-1,None),'-',color='#d62728',lw=1.4)
axA.set_xlabel('Voltage [V]'); axA.set_ylabel('$J$ [A cm$^{-2}$]'); axA.set_xlim(0,10); axA.set_ylim(0,1.0)
axAr.set_ylabel('$L$ [cd m$^{-2}$]'); axAr.set_ylim(1e0,1e6); axA.legend(title=r'$\tau$ [$\mu$s]',frameon=False,fontsize=8,loc='upper left')
axA.set_title('(a) J–L vs voltage:  $\\tau$ sweep ($C=0$)')
axB=axes[1,0]
for (fn,lab),g in zip(taus,grays):
    V,j,e=load('/tmp/f3tau_%s.npz'%fn); axB.plot(V,np.where(np.abs(j)>1e-6,e*100,0),'-',color=g,lw=1.8,label=lab)
axB.set_xlabel('Voltage [V]'); axB.set_ylabel('EQE [%]'); axB.set_xlim(0,10); axB.set_ylim(0,20); axB.legend(title=r'$\tau$ [$\mu$s]',frameon=False,fontsize=8)
axB.set_title('(b) EQE vs voltage:  $\\tau$ sweep ($C=0$)')
# ---------- (c,d) C sweep, tau=1.5 ----------
Cs=[('0.01','0'),('1','1'),('2','2'),('5','5'),('10','10'),('20','20')]
grC=plt.cm.Greys(np.linspace(0.35,0.95,len(Cs)))
axC=axes[0,1]; axCr=axC.twinx()
for (fn,lab),g in zip(Cs,grC):
    V,j,e=load('/tmp/f3C_%s.npz'%fn); axC.plot(V,j,'-',color=g,lw=1.5,label=lab)
    axCr.semilogy(V,np.clip(lum(j,e),1e-1,None),'-',color='#d62728',lw=1.2)
axC.set_xlabel('Voltage [V]'); axC.set_ylabel('$J$ [A cm$^{-2}$]'); axC.set_xlim(0,10); axC.set_ylim(0,1.0)
axCr.set_ylabel('$L$ [cd m$^{-2}$]'); axCr.set_ylim(1e0,1e6); axC.legend(title='$C$ [$10^{-31}$]',frameon=False,fontsize=7,loc='upper left')
axC.set_title('(c) J–L vs voltage:  $C$ sweep ($\\tau$=1.5 $\\mu$s)')
axD=axes[1,1]
for (fn,lab),g in zip(Cs,grC):
    V,j,e=load('/tmp/f3C_%s.npz'%fn); axD.plot(V,np.where(np.abs(j)>1e-6,e*100,0),'-',color=g,lw=1.8,label=lab)
axD.set_xlabel('Voltage [V]'); axD.set_ylabel('EQE [%]'); axD.set_xlim(0,10); axD.set_ylim(0,20); axD.legend(title='$C$ [$10^{-31}$]',frameon=False,fontsize=7)
axD.set_title('(d) EQE vs voltage:  $C$ sweep')
# ---------- (e,f) M sweep ----------
Ms=[2,3,4]
grM=plt.cm.Greys(np.linspace(0.45,0.92,len(Ms)))
axE=axes[0,2]; axEr=axE.twinx()
for M,g in zip(Ms,grM):
    V,j,e=load('/tmp/f3M_%d.npz'%M); axE.plot(V,j,'-',color=g,lw=1.8,label='%d'%M)
    axEr.semilogy(V,np.clip(lum(j,e),1e-1,None),'-',color='#d62728',lw=1.4)
axE.set_xlabel('Voltage [V]'); axE.set_ylabel('$J$ [A cm$^{-2}$]'); axE.set_xlim(0,10); axE.set_ylim(0,1.0)
axEr.set_ylabel('$L$ [cd m$^{-2}$]'); axEr.set_ylim(1e0,1e6); axE.legend(title='$M$',frameon=False,fontsize=8,loc='upper left')
axE.set_title('(e) J–L vs voltage:  $M$ sweep ($C$=10, $\\tau$=1.5)')
axF=axes[1,2]
for M,g in zip(Ms,grM):
    V,j,e=load('/tmp/f3M_%d.npz'%M); axF.plot(V,np.where(np.abs(j)>1e-6,e*100,0),'-',color=g,lw=1.8,label='%d'%M)
axF.set_xlabel('Voltage [V]'); axF.set_ylabel('EQE [%]'); axF.set_xlim(0,10); axF.set_ylim(0,7); axF.legend(title='$M$',frameon=False,fontsize=8)
axF.set_title('(f) EQE vs voltage:  $M$ sweep')
fig.suptitle('Recreation of Jung et al. (2021) Figure 3 with the offset-removed model (green base device, LEE=20%)',fontsize=13)
fig.savefig(OUT+'/recreate_fig3.png',dpi=150); print('saved recreate_fig3')
