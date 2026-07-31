import matplotlib; matplotlib.use('Agg', force=True)
import numpy as np, sesame
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt; plt.show=lambda *a,**k:None
import sys,os; sys.path.insert(0,'/tmp')
from fig4_build import build
a=0.9e-9; V=np.linspace(0,10,71)
conds={'balanced':dict(alpha_p=a,alpha_n=a,qd_shift=0.0),
       'erich':dict(alpha_p=0.5*a,alpha_n=1.5*a,qd_shift=0.0),
       'hrich':dict(alpha_p=1.5*a,alpha_n=0.5*a,qd_shift=0.0),
       'dEvp0.4':dict(alpha_p=a,alpha_n=a,qd_shift=0.2),
       'dEvm0.4':dict(alpha_p=a,alpha_n=a,qd_shift=-0.2)}
for name,cfg in conds.items():
    s,dEc,dEv=build(**cfg)
    d='/tmp/f4_%s'%name; os.makedirs(d,exist_ok=True)
    try:
        j,eqe,jem=sesame.IVcurve(s,V,d+'/V',tol=1e-6,htp=1,maxiter=400,verbose=False)
        j=np.abs(j*s.scaling.current)
        np.savez('/tmp/f4_%s.npz'%name,V=V,j=j,eqe=eqe)
        print("%-9s dEc=%.2f dEv=%.2f conv=%d Jmax=%.3f EQEpk=%.1f%%"%(name,dEc,dEv,np.sum(np.isfinite(j)),np.nanmax(j),100*np.nanmax(eqe)))
    except Exception as e:
        print("%-9s FAILED %s"%(name,type(e).__name__))
