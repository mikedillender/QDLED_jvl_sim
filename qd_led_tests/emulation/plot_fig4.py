import matplotlib; matplotlib.use('Agg', force=True)
import sys; sys.path.insert(0,'/tmp')
import numpy as np, sesame
import sesame.utils as u
import matplotlib; matplotlib.use('Agg',force=True)
import thesis_style as ts; from thesis_style import CB
import matplotlib.pyplot as plt; plt.show=lambda *a,**k:None
OUT='/sessions/sleepy-adoring-pasteur/mnt/Numerical_QDLED/thesis_figures'
V=np.linspace(0,10,71); B=0.58e-12; C=6.3e-31; tau=1.5e-6
def extract(name):
    nq=np.full(len(V),np.nan);pq=np.full(len(V),np.nan);Ur=np.full(len(V),np.nan);Us=np.full(len(V),np.nan);Ua=np.full(len(V),np.nan)
    for i in range(len(V)):
        try:
            s,r=u.load_sim('/tmp/f4_%s/V_%d.gzip'%(name,i))
        except: continue
        Nd=s.scaling.density; vt=s.scaling.energy; qd=s.qd_sites
        n=s.Nc*np.exp(s.bl+r['efn']+r['v'])*Nd; p=s.Nv*np.exp(-s.Eg-s.bl-r['efp']-r['v'])*Nd
        nq[i]=n[qd].mean(); pq[i]=p[qd].mean()
        Nc=s.Nc[qd]*Nd; Nv=s.Nv[qd]*Nd; Eg=s.Eg[qd]*vt; ni=np.sqrt(Nc*Nv)*np.exp(-Eg/(2*0.02585))
        npm=n[qd]*p[qd]-ni**2
        Ur[i]=np.mean(np.maximum(B*npm,0));Us[i]=np.mean(np.maximum(npm/(tau*(n[qd]+p[qd]+2*ni)),0));Ua[i]=np.mean(np.maximum(C*(n[qd]+p[qd])*npm,0))
    return nq,pq,Ur,Us,Ua
D={n:extract(n) for n in ['balanced','erich','hrich','dEvp0.4','dEvm0.4']}
EQE={n:np.load('/tmp/f4_%s.npz'%n)['eqe']*100 for n in D}
J={n:np.abs(np.load('/tmp/f4_%s.npz'%n)['j']) for n in D}
def egate(n): return np.where(J[n]>1e-6,EQE[n],np.nan)
fig,ax=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
cols=[('balanced',['balanced'],['balanced']),
      ('alpha_p:alpha_n',['erich','hrich'],['0.5:1.5 (e-rich)','1.5:0.5 (h-rich)']),
      ('dEv-dEc',['dEvp0.4','dEvm0.4'],['+0.4 eV','-0.4 eV'])]
palette=[CB['black'],CB['orange']]
for c,(title,names,labs) in enumerate(cols):
    for name,lab,pc in zip(names,labs,palette):
        nq,pq,Ur,Us,Ua=D[name]
        ax[0,c].semilogy(V,nq,'-',color=pc,lw=1.8,label='$n_{QD}$ '+lab); ax[0,c].semilogy(V,pq,'--',color=pc,lw=1.8,label='$p_{QD}$ '+lab)
        ax[1,c].semilogy(V,Ur,'-',color=CB['green'],lw=1.6); ax[1,c].semilogy(V,Us,'-',color=CB['blue'],lw=1.6); ax[1,c].semilogy(V,Ua,'-',color=CB['red'],lw=1.6)
        ax[2,c].plot(V,egate(name),'-',color=pc,lw=2,label=lab)
    ax[0,c].set_ylim(1e13,1e19); ax[0,c].set_title('(row1) QD densities: %s'%title); ax[0,c].legend(fontsize=7,frameon=False)
    ax[1,c].set_ylim(1e18,1e24); ax[1,c].set_title('(row2) Recomb rates'); 
    ax[2,c].set_ylim(0,8); ax[2,c].set_title('(row3) EQE'); ax[2,c].legend(fontsize=8,frameon=False)
    for rr in range(3): ax[rr,c].set_xlim(0,10); ax[rr,c].set_xlabel('Voltage [V]')
ax[1,0].plot([],[],'-',color=CB['green'],label='Radiative');ax[1,0].plot([],[],'-',color=CB['blue'],label='SRH');ax[1,0].plot([],[],'-',color=CB['red'],label='Auger');ax[1,0].legend(fontsize=8,frameon=False)
ax[0,0].set_ylabel('density [cm$^{-3}$]'); ax[1,0].set_ylabel('R [cm$^{-3}$s$^{-1}$]'); ax[2,0].set_ylabel('EQE [%]')
fig.suptitle('Recreation of Jung et al. (2021) Fig 4: charge-balance evolution (our offset-removed model)',fontsize=13)
fig.savefig(OUT+'/recreate_fig4.png',dpi=140); print('saved recreate_fig4')
# print n/p at 6V for each
for n in D:
    nq,pq,_,_,_=D[n]; i=int(round(6/10*70)); print("%-9s n/p@6V=%.3f"%(n,nq[i]/pq[i]))
