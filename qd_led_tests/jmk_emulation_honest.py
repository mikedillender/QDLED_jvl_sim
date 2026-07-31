import numpy as np, sesame
# JMK-honest emulation: reported doping (HTL 1e17, HIL 2.8e19), but REMOVE the abrupt
# HIL/HTL valence step by aligning the HIL valence band to the HTL (HIL Eg 1.57->2.0 so
# EV_HIL = EV_HTL = -5.6, EC_HIL kept at -3.6 to still block electrons).  This lets holes
# injected from the degenerate PEDOT flood the HTL (no exp(-dEv/kT) suppression) while the
# QD injection offset dEv (HTL->QD) is unchanged.
COL={'Red':dict(aff=3.82,Eg=1.97,d=7.0,M=2,tau=4.0e-6,C=2.8e-31,alpha=1.15e-8),
     'Green':dict(aff=3.66,Eg=2.28,d=5.0,M=2,tau=3.0e-6,C=6.3e-31,alpha=2.26e-9),
     'Blue':dict(aff=3.47,Eg=2.67,d=10.8,M=3,tau=1.1e-6,C=130e-31,alpha=1.45e-7)}
def build(color, remove_offset=True, alpha_scale=1.0):
    p=COL[color]; M=p['M']
    t_hil=20e-7; t_htl=20e-7; t_etl=40e-7; t_bqd=t_hil+t_htl
    r_qd=(p['d']/2)*1e-7; r_qds=0.5e-7
    t_qdl=2*M*r_qd; t_aqd=t_bqd+t_qdl; t_total=t_aqd+t_etl
    dd=4e-7; dd2=1.2e-7; qc=t_bqd+(2*np.arange(M)+1)*r_qd
    x=np.concatenate((np.linspace(0,dd,22,endpoint=False),np.linspace(dd,t_hil-dd2,30,endpoint=False),
        np.linspace(t_hil-dd2,t_hil+dd2,20,endpoint=False),np.linspace(t_hil+dd2,t_bqd-dd,30,endpoint=False),
        np.linspace(t_bqd-dd,t_bqd,16,endpoint=False),qc,np.linspace(t_aqd,t_aqd+dd,16,endpoint=False),
        np.linspace(t_aqd+dd,t_total-dd,30,endpoint=False),np.linspace(t_total-dd,t_total,22)))
    s=sesame.Builder(x)
    hil_Eg = 2.0 if remove_offset else 1.57   # 2.0 -> EV_HIL=-5.6 (aligned, no offset); 1.57 -> EV=-5.17 (real)
    hil={'Nc':2.5e19,'Nv':2.5e19,'Eg':hil_Eg,'epsilon':3.0,'Et':0,'mu_e':0.322e-3,'mu_h':0.322e-3,'tau_e':1.2e-6,'tau_h':1.2e-6,'affinity':3.6}
    htl={'Nc':2.5e19,'Nv':2.5e19,'Eg':3.0,'epsilon':3.5,'Et':0,'mu_e':2.0e-3,'mu_h':2.0e-3,'tau_e':1.2e-6,'tau_h':1.2e-6,'affinity':2.6}
    qdc={'Nc':2.5e19*0.13**1.5,'Nv':2.5e19*0.45**1.5,'Eg':p['Eg'],'epsilon':9.4,'Et':0,'mu_e':2e-6,'mu_h':1e-6,'tau_e':p['tau'],'tau_h':p['tau'],'affinity':p['aff'],'Cn':p['C'],'Cp':p['C'],'B':0.58e-12}
    etl={'Nc':2.5e19*0.24**1.5,'Nv':2.5e19*0.59**1.5,'Eg':3.4,'epsilon':8.5,'Et':0,'mu_e':2.0e-3,'mu_h':2.0e-3,'tau_e':1.2e-6,'tau_h':1.2e-6,'affinity':4.0}
    hr=lambda x:x<=t_hil; tr=lambda x:np.logical_and(x>t_hil,x<=t_bqd); qr=lambda x:np.logical_and(t_bqd<x,x<t_aqd); er=lambda x:x>=t_aqd
    s.add_material(etl,er); s.add_material(htl,tr); s.add_material(hil,hr); s.add_material(qdc,qr)
    dEc=etl['affinity']-qdc['affinity']; dEv=qdc['affinity']+qdc['Eg']-(htl['affinity']+htl['Eg'])
    s.add_qd(r_qds*1e7,qd_mns=0.19,qd_mps=0.60,dEc=dEc,dEv=dEv,location=qr,r_qd=r_qd)
    s.qd_alpha_n=p['alpha']*alpha_scale; s.qd_alpha_p=p['alpha']*alpha_scale
    s.add_donor(1e17,er); s.add_acceptor(1e17,tr); s.add_acceptor(2.81e19,hr)
    s.contact_type('Ohmic','Ohmic'); s.contact_S(1e7,1e7,1e7,1e7)
    return s,dEv


def run_iv(color='Green', voltages=None, alpha_scale=0.4, export_root='jmk_honest_out'):
    import os
    if voltages is None:
        voltages = np.linspace(0, 8, 81)
    s, dEv = build(color, remove_offset=True, alpha_scale=alpha_scale)
    folder = os.path.join(export_root, color); os.makedirs(folder, exist_ok=True)
    j, l, jem = sesame.IVcurve(s, voltages, os.path.join(folder, 'V'), maxiter=400, htp=1)
    j = j * s.scaling.current; jem = jem * s.scaling.current
    return s, {'v': voltages, 'j': j, 'jem': jem, 'eqe': l, 'color': color}


if __name__ == '__main__':
    for c in ['Red', 'Green', 'Blue']:
        run_iv(c)
