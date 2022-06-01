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
            ey_inc_td = np.load('./{}/ey_fi_l{}.npy'.format(usr.output_dir, jline))
            hz_inc_td = np.load('./{}/hz_fi_l{}.npy'.format(usr.output_dir, jline))
        elif j % 2 == 1:
            ey_inc_td = np.load('./{}/ey_bi_l{}.npy'.format(usr.output_dir, jline))
            hz_inc_td = np.load('./{}/hz_bi_l{}.npy'.format(usr.output_dir, jline))

        if i % 2 == j % 2:
            if i % 2 == 0:
                ey_ref_td = np.load('./{}/ey_ft_l{}.npy'.format(usr.output_dir, jline))
                hz_ref_td = np.load('./{}/hz_ft_l{}.npy'.format(usr.output_dir, jline))
            elif i % 2 == 1:
                ey_ref_td = np.load('./{}/ey_bt_l{}.npy'.format(usr.output_dir, jline))
                hz_ref_td = np.load('./{}/hz_bt_l{}.npy'.format(usr.output_dir, jline))
        elif i % 2 != j % 2:
            if i % 2 == 0:
                ey_ref_td = np.load('./{}/ey_fr_l{}.npy'.format(usr.output_dir, jline))
                hz_ref_td = np.load('./{}/hz_fr_l{}.npy'.format(usr.output_dir, jline))
            elif i % 2 == 1:
                ey_ref_td = np.load('./{}/ey_br_l{}.npy'.format(usr.output_dir, jline))
                hz_ref_td = np.load('./{}/hz_br_l{}.npy'.format(usr.output_dir, jline))

        if i == j:
            if i % 2 == 0:
                ey_inc_cor = np.load('./{}/ey_fi_l{}.npy'.format(usr.output_dir, iline))
                hz_inc_cor = np.load('./{}/hz_fi_l{}.npy'.format(usr.output_dir, iline))
            elif i % 2 == 1:
                ey_inc_cor = np.load('./{}/ey_bi_l{}.npy'.format(usr.output_dir, iline))
                hz_inc_cor = np.load('./{}/hz_bi_l{}.npy'.format(usr.output_dir, iline))
            ey_ref_td -= ey_inc_cor
            hz_ref_td -= hz_inc_cor

        ey_inc_fd = fft.fft(ey_inc_td, axis=0)
        hz_inc_fd = fft.fft(hz_inc_td, axis=0)
        ey_ref_fd = fft.fft(ey_ref_td, axis=0)
        hz_ref_fd = fft.fft(hz_ref_td, axis=0)
        sav_inc = 0.5*(ey_inc_fd * np.conjugate(hz_inc_fd))
        sav_ref = 0.5*(ey_ref_fd * np.conjugate(hz_ref_fd))
        start_inc = cfg.by+cfg.cy_swg-cfg.ph + cfg.pp*(jline-1)
        end_inc = start_inc + cfg.pp
        start_ref = cfg.by+cfg.cy_swg-cfg.ph + cfg.pp*(iline-1)
        end_ref = start_ref + cfg.pp

        Pinc = np.zeros(cfg.nt, dtype=cprec)
        Pref = np.zeros(cfg.nt, dtype=cprec)
        for p in range(start_inc, end_inc):
            Pinc += sav_inc[:, p]
        for p in range(start_ref, end_ref):
            Pref += sav_ref[:, p]

        S_mat[:, i, j] = Pref / Pinc

ff = np.arange(0, cfg.nt, 1) / (cfg.nt * cfg.delta_t)
fmin = int(100e12*cfg.nt*cfg.delta_t)+1
fmax = int(300e12*cfg.nt*cfg.delta_t)+2

freq = rf.Frequency.from_f(ff[fmin:fmax], unit='Hz')

net = rf.Network(frequency=freq, s=S_mat[fmin:fmax, :, :], z0=50, name='Interconnects')
net.write_touchstone('example', usr.output_dir, write_z0=True, skrf_comment=True, form='ri')

file = open('./{}/example.s{}p'.format(usr.output_dir, num_ports), mode='r')
file_lines = file.readlines()
file_ack = '!This S-parameter file was generated as part of the OIDT. This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, at the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].\n'
file_lines.insert(0, file_ack)

file = open('./{}/example.s{}p'.format(usr.output_dir, num_ports), mode='w')
file.writelines(file_lines)
file.close()

net.plot_s_db()
plt.grid(True)
plt.show()
