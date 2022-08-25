"""
Author: B. Guiana

Description:


Acknowledgement: This project was completed as part of research conducted with
                 my major professor and advisor, Prof. Ata Zadehgol, as part of
                 the Applied and Computational Electromagnetics Signal and
                 Power Integrity (ACEM-SPI) Lab while working toward the Ph.D.
                 in Electrical Engineering at the University of Idaho, Moscow,
                 Idaho, USA. This project was funded, in part, by the National
                 Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import fdtd_auto_setup as cfg
#import user_config as usr
from scipy import fft
import skrf as rf
import aux_funcs as aux

args = sys.argv
try:
    usr_name = 'user_dict_{}.pkl'.format(args[1])
    uf = open(usr_name, 'rb')
except:
    print('loading default configuration')
    import user_config as user_module
    user_dict = aux.dict_from_module(user_module)
    usr_name = 'user_dict.pkl'
    dict_file = open(usr_name, 'wb')
    pickle.dump(user_dict, dict_file)
    dict_file.close()
    uf = open(usr_name, 'rb')

usr = pickle.load(uf)

if usr['precision'] == np.float32:
    cprec = np.complex64
elif usr['precision'] == np.float64:
    cprec = np.complex128

num_ports = 2 * usr['num_lines']

S_mat = np.zeros([usr['sparam_num_freqs'], num_ports, num_ports], dtype=cprec)

for i in range(num_ports):
    for j in range(i, num_ports):
        iline = i//2 + 1
        jline = j//2 + 1
        if j % 2 == 0:
            ez_inc = np.load('./{}/ez_fi_l{}.npy'.format(usr['output_dir'], jline))
            hy_inc = np.load('./{}/hy_fi_l{}.npy'.format(usr['output_dir'], jline))
        elif j % 2 == 1:
            ez_inc = np.load('./{}/ez_bi_l{}.npy'.format(usr['output_dir'], jline))
            hy_inc = np.load('./{}/hy_bi_l{}.npy'.format(usr['output_dir'], jline))

        if i % 2 == j % 2:
            if i % 2 == 0:
                ez_ref = np.load('./{}/ez_ft_l{}.npy'.format(usr['output_dir'], jline))
                hy_ref = np.load('./{}/hy_ft_l{}.npy'.format(usr['output_dir'], jline))
            elif i % 2 == 1:
                ez_ref = np.load('./{}/ez_bt_l{}.npy'.format(usr['output_dir'], jline))
                hy_ref = np.load('./{}/hy_bt_l{}.npy'.format(usr['output_dir'], jline))
        elif i % 2 != j % 2:
            if i % 2 == 0:
                ez_ref = np.load('./{}/ez_fr_l{}.npy'.format(usr['output_dir'], jline))
                hy_ref = np.load('./{}/hy_fr_l{}.npy'.format(usr['output_dir'], jline))
            elif i % 2 == 1:
                ez_ref = np.load('./{}/ez_br_l{}.npy'.format(usr['output_dir'], jline))
                hy_ref = np.load('./{}/hy_br_l{}.npy'.format(usr['output_dir'], jline))

        if i == j:
            if i % 2 == 0:
                ez_inc_cor = np.load('./{}/ez_fi_l{}.npy'.format(usr['output_dir'], iline))
                hy_inc_cor = np.load('./{}/hy_fi_l{}.npy'.format(usr['output_dir'], iline))
            elif i % 2 == 1:
                ez_inc_cor = np.load('./{}/ez_bi_l{}.npy'.format(usr['output_dir'], iline))
                hy_inc_cor = np.load('./{}/hy_bi_l{}.npy'.format(usr['output_dir'], iline))
            ez_ref -= ez_inc_cor
            hy_ref -= hy_inc_cor

        sav_inc = -0.5*(ez_inc * np.conjugate(hy_inc))
        sav_ref = -0.5*(ez_ref * np.conjugate(hy_ref))
        start_inc = cfg.by+cfg.cy_swg-cfg.ph + cfg.pp*(jline-1)
        end_inc = start_inc + cfg.pp
        start_ref = cfg.by+cfg.cy_swg-cfg.ph + cfg.pp*(iline-1)
        end_ref = start_ref + cfg.pp

        Pinc = np.zeros(usr['sparam_num_freqs'], dtype=cprec)
        Pref = np.zeros(usr['sparam_num_freqs'], dtype=cprec)

        for p in range(start_inc, end_inc):
            Pinc += sav_inc[:, p]
        for p in range(start_ref, end_ref):
            Pref += sav_ref[:, p]

        S_mat[:, i, j] = np.sqrt(Pref / Pinc)
        S_mat[:, j, i] = np.sqrt(Pref / Pinc)

freq = rf.Frequency.from_f(cfg.sparam_freqs, unit='Hz')
net = rf.Network(frequency=freq, s=S_mat, z0=50, name='Interconnects')

net.write_touchstone(usr['sparam_file'], usr['output_dir'], write_z0=True, skrf_comment=True, form='ri')
file = open('./{}/{}.s{}p'.format(usr['output_dir'], usr['sparam_file'], num_ports), mode='r')
file_lines = file.readlines()
file_ack = '!This S-parameter file was generated as part of the OIDT. This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, at the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].\n'
file_lines.insert(0, file_ack)

file = open('./{}/{}.s{}p'.format(usr['output_dir'], usr['sparam_file'], num_ports), mode='w')
file.writelines(file_lines)
file.close()

print('Saving S-parmeter dB plot')
plt.figure()
net.plot_s_db()
plt.savefig('./{}/{}_db_plot.pdf'.format(usr['output_dir'], usr['sparam_file']))
print('Saving S-parameter angle (degrees) plot')
plt.figure()
net.plot_s_deg()
plt.savefig('./{}/{}_angle_plot.pdf'.format(usr['output_dir'], usr['sparam_file']))
plt.close()
