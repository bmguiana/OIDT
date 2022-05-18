"""
Created on Tue Dec 28 10:16:59 2021

@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import fdtd_config as cfg
import fdtd_funcs as funcs
import aux_funcs as aux


sigma_corners = [15]
lc_corners = [700]
num_profs = 1

results = np.zeros([5, len(sigma_corners)*len(lc_corners)*num_profs])

r = 0
for s in sigma_corners:
    for l in lc_corners:
        prof_path = './rough_profiles/'
        print('processing sigma={}nm, Lc={}nm'.format(s, l))
        FG_REG = funcs.gen_fg_mask_sgl(cfg.nx, cfg.ny, cfg.by, cfg.ny_swg, cfg.p1x, cfg.nx-cfg.p2x, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth', correlation=cfg.ctype, upper_path=cfg.up, lower_path=cfg.lp, atol=0.10, stol=0.10, mtol=0.01)
        EPS_MASK = cfg.eps * ( cfg.eps_rel_fg * FG_REG + 1 * ~FG_REG )
        EPS_EY = EPS_MASK.copy()

        for p in range(num_profs):
            res_path = './Results/'
            hz_file = 'hz_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            ey_file = 'ey_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            up_name = 'profile_upper_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            lp_name = 'profile_lower_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            try:
                H = np.load(res_path+hz_file)
                E = np.load(res_path+ey_file)
                up = np.load(prof_path+up_name)
                lp = np.load(prof_path+lp_name)
                E1 = E[cfg.p1x, :]
                E2 = E[cfg.p2x, :]
                H1 = H[cfg.p1x, :]
                H2 = H[cfg.p2x, :]

                offset = cfg.by-cfg.num_cpml + 0
                alpha = aux.get_scattering_loss_h2p_fd(H1, H2, EPS_EY[cfg.cx, :], 3.5, 1.5, 194.8e12, cfg.delta_t, cfg.delta_x, 100e-9, (cfg.p2x-cfg.p1x)*cfg.delta_x, cfg.num_cpml+offset)
                su, lcu, h = aux.check_discretization(up, cfg.delta_x)
                sl, lcl, h = aux.check_discretization(lp, cfg.delta_x)
                smeas = 0.5 * (su + sl)
                lcmeas = 0.5 * (lcu + lcl)
                results[0, r] = s
                results[1, r] = l
                results[2, r] = smeas
                results[3, r] = lcmeas
                results[4, r] = alpha
                r += 1
            except:
                print('unable to process profile', s, l, p)
                pass

np.save('./{}/results_hmode'.format(cfg.output_dir), results[:, :r])
