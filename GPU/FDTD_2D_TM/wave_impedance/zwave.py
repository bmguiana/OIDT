"""
Created on Wed Apr  6 10:48:30 2022

@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import matplotlib.pyplot as plt
import fdtd_config as cfg
import aux_funcs as aux

cell = 3
yref = cfg.by - cell
ez_td = np.load('./Results/ez_zwave.npy')[:, yref]
hx_td = np.load('./Results/hx_zwave.npy')[:, yref]
ez_fd = np.fft.fft(ez_td)
hx_fd = np.fft.fft(hx_td)
zw_sim = ez_fd / hx_fd

ff = np.arange(0, cfg.nt, 1) / (cfg.delta_t * cfg.nt)
fa = np.linspace(100e12, 300e12, num=100)

eps0 = 1e-9 / (36 * np.pi)
k0 = 2 * np.pi * fa / 3e8
n1 = 3.5
n2 = 1.5
d = 100e-9
neff = np.zeros(len(fa))
for n in range(len(fa)):
    neff[n] = aux.find_neff(n1, n2, d, k0[n], mode='te')

beta = k0*neff
gamma = np.sqrt(beta**2 - n2**2 * k0**2)
zw_ana = -1j * 2*np.pi*fa * cfg.mu / gamma

fig, ax = plt.subplots()
ax.plot(fa/1e12, zw_ana.imag, ls='-', c='k', label='Analytical')
ax.plot(ff/1e12, zw_sim.imag, ls='', c='r', marker='s', mfc='none', mec='r', label='2D FDTD')

ax.legend()
ax.axis([100, 300, -300, 0])
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel(r'$\Im\{ Z_{w,TE}^{-d}\}\ (\Omega)$')
ax.grid(True)

fig.tight_layout()
fig.savefig('./{}/zwave_te.pdf'.format(cfg.output_dir))
