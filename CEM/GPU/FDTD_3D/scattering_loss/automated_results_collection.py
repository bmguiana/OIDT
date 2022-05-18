"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import fdtd_config as cfg
import aux_funcs as aux


sigmas = [15]
lcs = [700]
num_profs = 1

results = np.zeros([5, len(sigmas)*len(lcs)*num_profs])

i = 0
ell = (cfg.p2z-cfg.p1z)*cfg.delta_x

for s in sigmas:
    for l in lcs:
        print('processing sigma={}nm, Lc={}nm'.format(s, l))
        for p in range(num_profs):
            res_path = './Results/'
            prof_path = './rough_profiles/'
            ex1_file = 'ex1_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            ey1_file = 'ey1_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            hx1_file = 'hx1_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            hy1_file = 'hy1_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)

            ex2_file = 'ex2_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            ey2_file = 'ey2_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            hx2_file = 'hx2_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            hy2_file = 'hy2_fft_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)

            up_name = 'profile_upper_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            lp_name = 'profile_lower_s{}nm_lc{}nm_r{}.npy'.format(s, l, p)
            try:
                Ex1 = np.load(res_path+ex1_file)
                Ey1 = np.load(res_path+ey1_file)
                Hx1 = np.load(res_path+hx1_file)
                Hy1 = np.load(res_path+hy1_file)

                Ex2 = np.load(res_path+ex2_file)
                Ey2 = np.load(res_path+ey2_file)
                Hx2 = np.load(res_path+hx2_file)
                Hy2 = np.load(res_path+hy2_file)

                up = np.load(prof_path+up_name)
                lp = np.load(prof_path+lp_name)
                su, lcu, h = aux.check_discretization(up, cfg.delta_x)
                sl, lcl, h = aux.check_discretization(lp, cfg.delta_x)
                smeas = 0.5 * (su + sl)
                lcmeas = 0.5 * (lcu + lcl)

                S1 = 0.5 * (Ex1 * np.conjugate(Hy1) - Ey1 * np.conjugate(Hx1)).real
                S2 = 0.5 * (Ex2 * np.conjugate(Hy2) - Ey2 * np.conjugate(Hx2)).real

                P1 = S1.sum() * cfg.delta_x*cfg.delta_x
                P2 = S2.sum() * cfg.delta_x*cfg.delta_x
                alpha = np.log(P1/P2)/ell

                results[0, i] = s
                results[1, i] = l
                results[2, i] = alpha
                results[3, i] = smeas
                results[4, i] = lcmeas
                i += 1
            except:
                print('unable to process profile', s, l, p)
                pass

np.save('./{}/results_3D_{}'.format(cfg.output_dir, cfg.source_condition), results[:, :i])
