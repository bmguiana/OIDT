"""
Author: B. Guiana

Description:


Acknowledgement: This project was completed as part of research conducted with
                 my major professor and advisor, Prof. Ata Zadehgol, at the
                 Applied and Computational Electromagnetics Signal and Power
                 Integrity (ACEM-SPI) Lab while working toward the Ph.D. in
                 Electrical Engineering at the University of Idaho, Moscow,
                 Idaho, USA. This project was funded, in part, by the National
                 Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import matplotlib.pyplot as plt
import fdtd_auto_setup as cfg
import user_config as usr
from scipy import fft
import skrf as rf
import aux_funcs as aux

if usr.precision == np.float32:
    cprec = np.complex64
elif usr.precision == np.float64:
    cprec = np.complex128

num_ports = 2 * usr.num_lines

S_mat = np.zeros([cfg.nt, num_ports, num_ports], dtype=cprec)
for i in range(num_ports):
    for j in range(num_ports):
        iline = i//2 + 1
        jline = j//2 + 1
        if j % 2 == 0:
            print('fi{}'.format(jline))
            ex_inc_td = np.load('./{}/ex_fi_l{}.npy'.format(usr.output_dir, jline))
            ey_inc_td = np.load('./{}/ey_fi_l{}.npy'.format(usr.output_dir, jline))
            hx_inc_td = np.load('./{}/hx_fi_l{}.npy'.format(usr.output_dir, jline))
            hy_inc_td = np.load('./{}/hy_fi_l{}.npy'.format(usr.output_dir, jline))
        elif j % 2 == 1:
            print('bi{}'.format(jline))
            ex_inc_td = np.load('./{}/ex_bi_l{}.npy'.format(usr.output_dir, jline))
            ey_inc_td = np.load('./{}/ey_bi_l{}.npy'.format(usr.output_dir, jline))
            hx_inc_td = np.load('./{}/hx_bi_l{}.npy'.format(usr.output_dir, jline))
            hy_inc_td = np.load('./{}/hy_bi_l{}.npy'.format(usr.output_dir, jline))

        if i % 2 == j % 2:
            if i % 2 == 0:
                print('ft{}'.format(jline))
                ex_ref_td = np.load('./{}/ex_ft_l{}.npy'.format(usr.output_dir, jline))
                ey_ref_td = np.load('./{}/ey_ft_l{}.npy'.format(usr.output_dir, jline))
                hx_ref_td = np.load('./{}/hx_ft_l{}.npy'.format(usr.output_dir, jline))
                hy_ref_td = np.load('./{}/hy_ft_l{}.npy'.format(usr.output_dir, jline))
            elif i % 2 == 1:
                print('bt{}'.format(jline))
                ex_ref_td = np.load('./{}/ex_bt_l{}.npy'.format(usr.output_dir, jline))
                ey_ref_td = np.load('./{}/ey_bt_l{}.npy'.format(usr.output_dir, jline))
                hx_ref_td = np.load('./{}/hx_bt_l{}.npy'.format(usr.output_dir, jline))
                hy_ref_td = np.load('./{}/hy_bt_l{}.npy'.format(usr.output_dir, jline))
        elif i % 2 != j % 2:
            if i % 2 == 0:
                print('fr{}'.format(jline))
                ex_ref_td = np.load('./{}/ex_fr_l{}.npy'.format(usr.output_dir, jline))
                ey_ref_td = np.load('./{}/ey_fr_l{}.npy'.format(usr.output_dir, jline))
                hx_ref_td = np.load('./{}/hx_fr_l{}.npy'.format(usr.output_dir, jline))
                hy_ref_td = np.load('./{}/hy_fr_l{}.npy'.format(usr.output_dir, jline))
            elif i % 2 == 1:
                print('br{}'.format(jline))
                ex_ref_td = np.load('./{}/ex_br_l{}.npy'.format(usr.output_dir, jline))
                ey_ref_td = np.load('./{}/ey_br_l{}.npy'.format(usr.output_dir, jline))
                hx_ref_td = np.load('./{}/hx_br_l{}.npy'.format(usr.output_dir, jline))
                hy_ref_td = np.load('./{}/hy_br_l{}.npy'.format(usr.output_dir, jline))

        if i == j:
            if i % 2 == 0:
                print('cor fi{}'.format(jline))
                ex_inc_cor = np.load('./{}/ex_fi_l{}.npy'.format(usr.output_dir, iline))
                ey_inc_cor = np.load('./{}/ey_fi_l{}.npy'.format(usr.output_dir, iline))
                hx_inc_cor = np.load('./{}/hx_fi_l{}.npy'.format(usr.output_dir, iline))
                hy_inc_cor = np.load('./{}/hy_fi_l{}.npy'.format(usr.output_dir, iline))
            elif i % 2 == 1:
                print('cor bi{}'.format(jline))
                ex_inc_cor = np.load('./{}/ex_bi_l{}.npy'.format(usr.output_dir, iline))
                ey_inc_cor = np.load('./{}/ey_bi_l{}.npy'.format(usr.output_dir, iline))
                hx_inc_cor = np.load('./{}/hx_bi_l{}.npy'.format(usr.output_dir, iline))
                hy_inc_cor = np.load('./{}/hy_bi_l{}.npy'.format(usr.output_dir, iline))
            ex_ref_td -= ex_inc_cor
            ey_ref_td -= ey_inc_cor
            hx_ref_td -= hx_inc_cor
            hy_ref_td -= hy_inc_cor

        ex_inc_fd = fft.fft(ex_inc_td, axis=0)
        ey_inc_fd = fft.fft(ey_inc_td, axis=0)
        hx_inc_fd = fft.fft(hx_inc_td, axis=0)
        hy_inc_fd = fft.fft(hy_inc_td, axis=0)
        
        ex_ref_fd = fft.fft(ex_ref_td, axis=0)
        ey_ref_fd = fft.fft(ey_ref_td, axis=0)
        hx_ref_fd = fft.fft(hx_ref_td, axis=0)
        hy_ref_fd = fft.fft(hy_ref_td, axis=0)
        
        sav_inc = 0.5 * (ex_inc_fd * np.conjugate(hy_inc_fd) - ey_inc_fd * np.conjugate(hx_inc_fd))
        sav_ref = 0.5 * (ex_ref_fd * np.conjugate(hy_ref_fd) - ey_ref_fd * np.conjugate(hx_ref_fd))
        start_inc = cfg.bx+cfg.cx_swg-cfg.ph + cfg.pp*(jline-1)
        end_inc = start_inc + cfg.pp
        start_ref = cfg.bx+cfg.cx_swg-cfg.ph + cfg.pp*(iline-1)
        end_ref = start_ref + cfg.pp

        Pinc = np.zeros(cfg.nt, dtype=cprec)
        Pref = np.zeros(cfg.nt, dtype=cprec)
        for y in range(usr.num_cpml, cfg.ny-usr.num_cpml):
            for p in range(start_inc, end_inc):
                Pinc += sav_inc[:, p, y]
            for p in range(start_ref, end_ref):
                Pref += sav_ref[:, p, y]

        S_mat[:, i, j] = Pref / Pinc
        print('S{}{}'.format(i+1, j+1))
        print(Pref[68], Pinc[68], '\n')

ff = np.arange(0, cfg.nt, 1) / (cfg.nt * cfg.delta_t)
fmin = int(100e12*cfg.nt*cfg.delta_t)+1
fmax = int(300e12*cfg.nt*cfg.delta_t)+2

freq = rf.Frequency.from_f(ff[fmin:fmax], unit='Hz')

net = rf.Network(frequency=freq, s=S_mat[fmin:fmax, :, :], z0=50, name='Interconnects')
net.write_touchstone('{}'.format(usr.sparam_file), usr.output_dir, write_z0=True, skrf_comment=True, form='ri')

file = open('./{}/{}.s{}p'.format(usr.output_dir, usr.sparam_file, num_ports), mode='r')
file_lines = file.readlines()
file_ack = '!This S-parameter file was generated as part of the OIDT. This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, at the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].\n'
file_lines.insert(0, file_ack)

file = open('./{}/{}.s{}p'.format(usr.output_dir, usr.sparam_file, num_ports), mode='w')
file.writelines(file_lines)
file.close()

net.plot_s_db()
plt.grid(True)
plt.show()
