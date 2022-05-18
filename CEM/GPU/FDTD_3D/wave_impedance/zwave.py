"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import matplotlib.pyplot as plt
import fdtd_config as cfg
import aux_funcs as aux

mode = 'TM'
cell = 3

xref = cfg.bx - cell
if mode == 'TE':
    ey_td = np.load('./Results/ey_te_zwave.npy')[:, xref]
    hz_td = np.load('./Results/hz_te_zwave.npy')[:, xref]
    ey_fd = np.fft.fft(ey_td)
    hz_fd = np.fft.fft(hz_td)
    zw_sim = ey_fd / hz_fd
elif mode == 'TM':
    ez_td = np.load('./Results/ez_tm_zwave.npy')[:, xref]
    hy_td = np.load('./Results/hy_tm_zwave.npy')[:, xref]
    ez_fd = np.fft.fft(ez_td)
    hy_fd = np.fft.fft(hy_td)
    zw_sim = -1*ez_fd / hy_fd

ff = np.arange(0, cfg.nt, 1) / (cfg.delta_t * cfg.nt)
fa = np.linspace(100e12, 300e12, num=100)

eps0 = 1e-9 / (36 * np.pi)
k0 = 2 * np.pi * fa / 3e8
n1 = 3.5
n2 = 1.5
d = 100e-9
neff = np.zeros(len(fa))
for n in range(len(fa)):
    if mode == 'TE':
        neff[n] = aux.find_neff(n1, n2, d, k0[n], mode='te')
    elif mode == 'TM':
        neff[n] = aux.find_neff(n1, n2, d, k0[n], mode='tm')

beta = k0*neff
gamma = np.sqrt(beta**2 - n2**2 * k0**2)
if mode == 'TE':
    zw_ana = -1j * 2*np.pi*fa * cfg.mu / gamma
elif mode == 'TM':
    zw_ana = -1*gamma / (1j * 2*np.pi*fa*eps0*n2**2)

fig, ax = plt.subplots()
ax.plot(fa/1e12, zw_ana.imag, ls='-', c='k', label='Analytical')
ax.plot(ff/1e12, zw_sim.imag, ls='', c='b', marker='o', mfc='none', mec='b', label='3D FDTD')

ax.legend()
if mode == 'TE':
    ax.axis([100, 300, -300, 0])
    ax.set_ylabel(r'$\Im\{ Z_{w,TE}^{-d}\}\ (\Omega)$')
elif mode == 'TM':
    ax.axis([100, 300, 0, 400])
    ax.set_ylabel(r'$\Im\{ Z_{w,TM}^{-d}\}\ (\Omega)$')
ax.set_xlabel('Frequency (THz)')
ax.grid(True)

fig.tight_layout()
fig.savefig('./{}/zwave_te.pdf'.format(cfg.output_dir))
