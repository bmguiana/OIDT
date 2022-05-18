"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import fdtd_config as cfg
import aux_funcs as aux
import importlib

sigma_corners = [15]
lc_corners = [700]
num_profs = 1

results = np.zeros([5, len(sigma_corners)*len(lc_corners)*num_profs])
i = 0

for s in sigma_corners:
    for l in lc_corners:
        print('processing sigma={}nm, Lc={}nm, '.format(s, l))
        prof_path = './rough_profiles/'
        for p in range(num_profs):
            res_path = './Results/'
            ez_file = 'ez_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            up_name = 'profile_upper_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            lp_name = 'profile_lower_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            try:
                ez = np.load(res_path+ez_file)
                up = np.load(prof_path+up_name)
                lp = np.load(prof_path+lp_name)
                A = ez[cfg.p1x, :]
                B = ez[cfg.p2x, :]
                alpha = aux.get_scattering_loss_e2p_fd(A, B, 3.5, 1.5, 194.8e12, cfg.delta_t, cfg.delta_x, 100e-9, (cfg.p2x-cfg.p1x)*cfg.delta_x, cfg.num_cpml+10)
                su, lcu, h = aux.check_discretization(up, cfg.delta_x)
                sl, lcl, h = aux.check_discretization(lp, cfg.delta_x)
                smeas = 0.5 * (su + sl)
                lcmeas = 0.5 * (lcu + lcl)
                if su == 0 and sl == 0:
                    print('both bad at: {}, {}, {}'.format(s, l, p))
                elif su == 0:
                    print('only upper bad at: {}, {}, {}'.format(s, l, p))
                elif sl == 0:
                    print('only lower bad at: {}, {}, {}'.format(s, l, p))
                results[0, i] = s
                results[1, i] = l
                results[2, i] = smeas
                results[3, i] = lcmeas
                results[4, i] = alpha
                i += 1
            except:
                print('unable to process profile', s, l, p)
                pass

np.save('./{}/results_emode'.format(cfg.output_dir), results[:, :i])
