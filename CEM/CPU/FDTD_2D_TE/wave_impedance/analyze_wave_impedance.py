#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:35:10 2021

@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import matplotlib.pyplot as plt
import aux_funcs as aux
import fdtd_config as cfg

# =============================================================================
# Import and interpret FDTD data
# =============================================================================

cell = 2                                        # Set cell offset for analysis location

ex_td = np.load('./Results/ex_zwave.npy')       # Load E-field data
hz_td = np.load('./Results/hz_zwave.npy')       # Load H-field data

ex_td = 0.5*(ex_td[:, cfg.by-cell] + ex_td[:, cfg.by-cell+1])   # Interpolate E-field data
hz_td = hz_td[:, cfg.by-cell]                                   # Interpolate H-field data

ex_fd = np.fft.fft(ex_td)                       # Apply FFT to E-field
hz_fd = np.fft.fft(hz_td)                       # Apply FFT to H-field

zw_sim = -1*ex_fd/hz_fd                         # Calculate Zwave

# =============================================================================
# Generate analytical solution data
# =============================================================================

ff = np.arange(0, cfg.nt, 1) / (cfg.nt*cfg.delta_t)     # Aux. array for frequency samples
eps0 = cfg.eps/cfg.eps_rel_bg                           # Free-space permittivity

k0 = 2*np.pi*ff/3e8                                     # Free-space wave number across samples
n1 = np.sqrt(cfg.eps_rel_fg*cfg.eps_rel_bg)             # Core region refractive index
n2 = np.sqrt(cfg.eps_rel_bg)                            # Cladding region refractive index
d = (cfg.ny_swg/2)*cfg.delta_x                          # Waveguide half-width
neff = np.zeros(cfg.nt)                                 # Setup for effective index method
for n in range(cfg.nt//2):
    neff[n] = aux.find_neff(n1, n2, d, k0[n], mode='tm')
beta = k0*neff                                          # Waveguide wave number
gamma = np.sqrt(beta**2 - n2**2 * k0**2)                # Half-width modification across samples
zw_ana = -1*gamma / (1j * 2*np.pi*ff*cfg.eps)           # Calculate Zwave

# =============================================================================
# Plot results
# =============================================================================

# Direct comparison
fig, ax = plt.subplots()
ax.plot(ff/1e12, zw_ana.imag, c='b', ls='-', label='Analytical')
ax.plot(ff/1e12, zw_sim.imag, c='r', ls='--', label='FDTD')
ax.axvline(cfg.f0/1e12, c='k', ls=':', label=r'$f_0$')
ax.set_xlim([100, 300])
ax.set_ylim([0, 400])
ax.grid()
ax.legend()
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel(r'Impedance ($\Omega$)')
ax.set_title(r'FDTD 2D $TE^z$ Wave Impedance')
fig.tight_layout()

# Percentage error comparison
fig, ax = plt.subplots()
ax.plot(ff/1e12, 100 - 100*zw_ana.imag/zw_sim.imag, c='g', ls='-', label='')
ax.axvline(cfg.f0/1e12, c='k', ls=':', label=r'$f_0$')
ax.set_xlim([100, 300])
ax.set_ylim([-5, 5])
ax.grid()
ax.legend()
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Error (%)')
ax.set_title('Analytical vs. FDTD Error')
fig.tight_layout()
