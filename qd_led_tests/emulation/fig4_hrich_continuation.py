import matplotlib; matplotlib.use('Agg', force=True)
import numpy as np, sesame
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt; plt.show=lambda *a,**k:None
import sys,os; sys.path.insert(0,'/tmp')
from fig4_build import build
from sesame.solvers import Solver
from sesame.analyzer import Analyzer
a=0.9e-9; V=np.linspace(0,10,71)
s,dEc,dEv=build(alpha_p=1.5*a,alpha_n=0.5*a,qd_shift=0.0)
apt,ant=s.qd_alpha_p,s.qd_alpha_n
nx=s.nx; rc=nx-1; q=1 if s.rho[nx-1]<0 else -1
solver=Solver(use_mumps=False); solver.solve(s,compute='Poisson',tol=1e-8,maxiter=2000,verbose=False,htp=1)
eq=solver.equilibrium.copy()
res={'efn':np.zeros(nx),'efp':np.zeros(nx),'v':eq.copy()}
for f in np.geomspace(1e-3,1.0,10):
    s.qd_alpha_p=apt*f; s.qd_alpha_n=ant*f; solver.equilibrium=eq.copy()
    g={k:res[k].copy() for k in res}; g['v'][rc]=eq[rc]
    out=solver.solve(s,guess=g,compute='all',tol=1e-6,maxiter=400,verbose=False,htp=1)
    if out is not None: res=out
s.qd_alpha_p=apt; s.qd_alpha_n=ant
J=np.full(len(V),np.nan); E=np.full(len(V),np.nan); prev=None; last=res
d='/tmp/f4_hrich'; os.makedirs(d,exist_ok=True)
import pickle,gzip
for i,vapp in enumerate(V):
    vd=vapp/s.scaling.energy; g={k:last[k].copy() for k in last}
    if prev is not None and vapp>1:
        for k in g: g[k]=last[k]+(last[k]-prev[k])
    g['v'][rc]=eq[rc]+q*vd; solver.equilibrium=eq.copy()
    out=solver.solve(s,guess=g,compute='all',tol=1e-6,maxiter=300,verbose=False,htp=1)
    if out is None: continue
    az=Analyzer(s,out); J[i]=az.full_current()*s.scaling.current; E[i]=az.full_emission()
    f=gzip.GzipFile(d+'/V_%d.gzip'%i,'wb'); f.write(pickle.dumps((s,out))); f.close()
    prev=last; last=out
np.savez('/tmp/f4_hrich.npz',V=V,j=np.abs(J),eqe=E)
print("hrich (continuation): conv=%d/%d Jmax=%.3f EQEpk=%.1f%%"%(np.sum(np.isfinite(J)),len(V),np.nanmax(np.abs(J)),100*np.nanmax(E)))
