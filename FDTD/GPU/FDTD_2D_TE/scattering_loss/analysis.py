"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import fdtd_config as cfg
import aux_funcs as aux


sigma_corners = [15]
lc_corners = [700]
fdtd = np.load('./{}/results_hmode.npy'.format(cfg.output_dir))


def grab_sim_results(A, sig, lc):
    B = np.zeros(A.shape[0])
    for n in range(A.shape[1]):
        if A[0, n] == sig and A[1, n] == lc:
            B = np.vstack([B, A[:, n]])
    if len(B.shape) < 2:
        print('\n\n\n!v!v!THIS SET INVLAID!v!v!')
        return np.zeros([A.shape[0], 1])
    return B.T[:, 1:]

def grab_some_results(A, sig, lc, max_res):
    B = np.zeros(A.shape[0])
    i = 0
    for n in range(A.shape[1]):
        if A[0, n] == sig and A[1, n] == lc and i < max_res:
            B = np.vstack([B, A[:, n]])
            i += 1
    return B.T[:, 1:]


def find_alpha(sigma, Lc):
    Be = 1
    eta0 = 120*np.pi
    n1 = np.sqrt(cfg.eps_rel_fg * cfg.eps_rel_bg)
    n2 = np.sqrt(cfg.eps_rel_bg)
    d = 100e-9
    k0 = 2 * np.pi * cfg.f0 / 3e8

    neff = aux.find_neff(n1, n2, d, k0, mode='tm')
    beta = 1.0*neff * k0
    kappa = np.sqrt( ( n1 * k0 )**2 - beta**2 )
    gamma = np.sqrt( beta**2 - ( n2 * k0 )**2 )
    MW = ( n1**2 - n2**2 )**2 * k0**3 / ( 4 * np.pi * n2 )
    theta = np.linspace(0, np.pi, num=3142)
    k = beta - (n2 * k0 * np.cos( theta ))
    RXX = 2 * Lc * sigma**2 / ( 1 + Lc**2 * k**2 )
    SW = 0
    for t in range(len(theta)):
        SW += RXX[t] * theta[1]

    nrad = n2
    na1 = n1
    na2 = n1
    na3 = n1**2/neff
    na4 = neff
    nb1 = n1
    nb2 = n2
    nb3 = n2**2/neff
    nb4 = neff

    F = n2**2 / (n1**4 * gamma**2 + n2**4 * kappa**2)
    deff_lp90 = n1*((d + n1**2 * F * gamma)/na1 + (n2**2 * kappa**2 * F/gamma)/nb1)
    deff_dp = n1*((d + n1**2 * F * gamma)/na2 + (n2**2 * kappa**2 * F/gamma)/nb2)
    deff_access = (n1**2 / neff)*((d + n1**2 * F * gamma)/na3 + (n2**2 * kappa**2 * F/gamma)/nb3)
    deff_ei = neff*((d + n1**2 * F * gamma)/na4 + (n2**2 * kappa**2 * F/gamma)/nb4)

    Prad = 0.5 * eta0/nrad * Be**2 * np.cos(kappa*d)**2 * MW * SW
    Pg_lp90 = 0.5 * eta0/n1 * Be**2 * deff_lp90
    Pg_dp = 0.5 * eta0/n1 * Be**2 * deff_dp
    Pg_access = 0.5 * (neff*eta0/n1**2) * Be**2 * deff_access
    Pg_ei = 0.5 * eta0/neff * Be**2 * deff_ei
    alpha_lp90 = Prad/Pg_lp90
    alpha_dp = Prad/Pg_dp
    alpha_access = Prad/Pg_access
    alpha_ei = Prad/Pg_ei
    return alpha_lp90, alpha_dp, alpha_access, alpha_ei


for kind in range(4):
    if kind == 0:
        print('Lacey/Payne 1990 Style (n1, n1)')
    if kind == 1:
        print('Piecewise Style (n1, n2)')
    if kind == 2:
        print('Access Style (n1_eff, n2_eff)')
    if kind == 3:
        print('Effective Impdedance (n_eff, n_eff)')
    for s in sigma_corners:
        print('sigma = {:} nm'.format(s))
        sigma_offsets = np.array([])
        overall_errors = np.array([])
        for l in lc_corners:
            this_set = grab_sim_results(fdtd, s, l)
            set_alpha = this_set[4]
            set_dbcm = set_alpha * 8.686/100
            set_sigma = np.mean(this_set[2])
            set_lc = np.mean(this_set[3])
            m = 0
            b = 0
            cfa = (1 - m*l - b)
            analytic = find_alpha(s*1e-9, l*1e-9)[kind] * cfa
            error = 1 * 100 * (analytic - np.mean(set_alpha))/analytic
            eplus = 100 * (analytic - np.max(set_alpha))/analytic
            eminus = 100 * (analytic - np.min(set_alpha))/analytic
            sigma_offsets = np.append(sigma_offsets, analytic/np.mean(set_alpha))
            overall_errors = np.append(overall_errors, error)
            print('Lc: {:} nm,\t sims: {:},\tset cf: {:.2f},\tset sd: {:.0f},\tset alpha: {:.0f},\tBW: +/-{:.2f}%,\tmin: {:.2f}%\tmax: {:.2f}%,\terr: {:.2f}%'.format(l, this_set.shape[1], analytic/np.mean(set_alpha), np.std(set_alpha), np.mean(set_alpha), 100*np.std(set_alpha)/np.mean(set_alpha), eplus, eminus, error))
        print('Error mean across LCs for this sigma is:\t{:.3f}'.format(np.mean(overall_errors)))
        print('Error dev. across LCs for this sigma is:\t{:.3f}\n\n'.format(np.std(overall_errors)))
