import numpy as np, sesame
# Green base (offset-removed), with separately tunable hole/electron injection (alpha_p, alpha_n)
# and a QD level shift qd_shift that makes dEv-dEc = 2*qd_shift (for the offset-imbalance panels).
def build(alpha_p=0.9e-9, alpha_n=0.9e-9, qd_shift=0.0, M=2, tau=1.5e-6, Cval=6.3e-31):
    aff=3.66+qd_shift; Eg=2.28
    t_hil=t_htl=20e-7; t_etl=40e-7; t_bqd=t_hil+t_htl
    r_qd=2.5e-7; r_qds=0.5e-7; t_qdl=2*M*r_qd; t_aqd=t_bqd+t_qdl; t_total=t_aqd+t_etl
    dd,dd2=4e-7,1.2e-7; qc=t_bqd+(2*np.arange(M)+1)*r_qd
    x=np.concatenate((np.linspace(0,dd,22,endpoint=False),np.linspace(dd,t_hil-dd2,28,endpoint=False),
        np.linspace(t_hil-dd2,t_hil+dd2,18,endpoint=False),np.linspace(t_hil+dd2,t_bqd-dd,28,endpoint=False),
        np.linspace(t_bqd-dd,t_bqd,16,endpoint=False),qc,np.linspace(t_aqd,t_aqd+dd,16,endpoint=False),
        np.linspace(t_aqd+dd,t_total-dd,28,endpoint=False),np.linspace(t_total-dd,t_total,22)))
    s=sesame.Builder(x)
    hil={'Nc':2.5e19,'Nv':2.5e19,'Eg':2.0,'epsilon':3.0,'Et':0,'mu_e':0.322e-3,'mu_h':0.322e-3,'tau_e':1.2e-6,'tau_h':1.2e-6,'affinity':3.6}
    htl={'Nc':2.5e19,'Nv':2.5e19,'Eg':3.0,'epsilon':3.5,'Et':0,'mu_e':2e-3,'mu_h':2e-3,'tau_e':1.2e-6,'tau_h':1.2e-6,'affinity':2.6}
    qdc={'Nc':2.5e19*0.13**1.5,'Nv':2.5e19*0.45**1.5,'Eg':Eg,'epsilon':9.4,'Et':0,'mu_e':2e-6,'mu_h':1e-6,'tau_e':tau,'tau_h':tau,'affinity':aff,'Cn':Cval,'Cp':Cval,'B':0.58e-12}
    etl={'Nc':2.5e19*0.24**1.5,'Nv':2.5e19*0.59**1.5,'Eg':3.4,'epsilon':8.5,'Et':0,'mu_e':2e-3,'mu_h':2e-3,'tau_e':1.2e-6,'tau_h':1.2e-6,'affinity':4.0}
    hr=lambda x:x<=t_hil;tr=lambda x:np.logical_and(x>t_hil,x<=t_bqd);qr=lambda x:np.logical_and(t_bqd<x,x<t_aqd);er=lambda x:x>=t_aqd
    s.add_material(etl,er);s.add_material(htl,tr);s.add_material(hil,hr);s.add_material(qdc,qr)
    dEc=etl['affinity']-qdc['affinity'];dEv=qdc['affinity']+qdc['Eg']-(htl['affinity']+htl['Eg'])
    s.add_qd(r_qds*1e7,qd_mns=0.19,qd_mps=0.60,dEc=dEc,dEv=dEv,location=qr,r_qd=r_qd)
    s.qd_alpha_p=alpha_p; s.qd_alpha_n=alpha_n; s.eqe_outcoupling=0.20
    s.add_donor(1e17,er);s.add_acceptor(1e17,tr);s.add_acceptor(2.81e19,hr)
    s.contact_type('Ohmic','Ohmic');s.contact_S(1e7,1e7,1e7,1e7)
    return s,dEc,dEv
