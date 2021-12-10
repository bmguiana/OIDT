#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import matplotlib.pyplot as plt
import fdtd_config as cfg
import aux_funcs as aux

# =============================================================================
# Import and interpret FDTD data
# =============================================================================

cell = 2

ez_td = np.load('./Results/ez_zwave.npy')
hx_td = np.load('./Results/hx_zwave.npy')

ez_td = ez_td[:, cfg.by-cell]
hx_td = 0.5*(hx_td[:, cfg.by-cell-1] + hx_td[:, cfg.by-cell])

ez_fd = np.fft.fft(ez_td)
hx_fd = np.fft.fft(hx_td)

zw_sim = ez_fd/hx_fd

# =============================================================================
# Generate analytical solution data
# =============================================================================

ff = np.arange(0, cfg.nt, 1) / (cfg.nt * cfg.delta_t)
eps0 = cfg.eps/cfg.eps_rel_bg

k0 = 2 * np.pi * ff / 3e8
n1 = np.sqrt(cfg.eps_rel_fg*cfg.eps_rel_bg)
n2 = np.sqrt(cfg.eps_rel_bg)
d = (cfg.ny_swg/2)*cfg.delta_x
neff = np.zeros(len(ff))
for n in range(len(ff)//4):
    neff[n] = aux.find_neff(n1, n2, d, k0[n], mode='te')

beta = k0*neff
gamma = np.sqrt(beta**2 - n2**2 * k0**2)
zw_ana = -1j * 2*np.pi*ff * cfg.mu / gamma

# =============================================================================
# Plot results
# =============================================================================

# Direct comparison
fig, ax = plt.subplots()
ax.plot(ff/1e12, zw_ana.imag, c='b', ls='-', label='Analytical')
ax.plot(ff/1e12, zw_sim.imag, c='r', ls='--', label='FDTD')
ax.axvline(cfg.f0/1e12, c='k', ls=':', label=r'$f_0$')
ax.set_xlim([100, 300])
ax.set_ylim([-250, -100])
ax.grid()
ax.legend()
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel(r'Impedance ($\Omega$)')
fig.tight_layout()

# Percentage error comparison
fig, ax = plt.subplots()
ax.plot(ff/1e12, 100 - 100*zw_ana.imag/zw_sim.imag, c='g', ls='-', label='')
ax.axvline(cfg.f0/1e12, c='k', ls=':', label=r'$f_0$')
ax.set_xlim([100, 300])
ax.set_ylim([-2, 2])
ax.grid()
ax.legend()
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Error (%)')
ax.set_title('Analytical vs. FDTD Error')
fig.tight_layout()
