#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:03:10 2021

@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import fdtd_config as cfg
from matplotlib import pyplot as plt
from aux_funcs import find_neff

# =============================================================================
# Import FDTD data
# =============================================================================

hp1i_p2i0_td = np.load('./Results/hz_p1i_p2i0_s0_lc0_r0.npy')
hp2i_p1i0_td = np.load('./Results/hz_p2i_p1i0_s0_lc0_r0.npy')
hp1t_p2i0_td = np.load('./Results/hz_p1t_p2i0_s0_lc0_r0.npy')
hp2t_p1i0_td = np.load('./Results/hz_p2t_p1i0_s0_lc0_r0.npy')
hp1r_p1i0_td = np.load('./Results/hz_p1r_p1i0_s0_lc0_r0.npy')
hp2r_p2i0_td = np.load('./Results/hz_p2r_p2i0_s0_lc0_r0.npy')
hp1r_p2i0_td = hp1t_p2i0_td - hp1i_p2i0_td              # Calculate implicit port 1 reflected field
hp2r_p1i0_td = hp2t_p1i0_td - hp2i_p1i0_td              # Calculate implicit port 2 reflected field

# =============================================================================
# Convert field data (A/m) to port current (A)
# =============================================================================

I1i_p2i0_td = np.zeros(cfg.nt)
I2i_p1i0_td = np.zeros(cfg.nt)
I1r_p1i0_td = np.zeros(cfg.nt)
I2r_p2i0_td = np.zeros(cfg.nt)
I1r_p2i0_td = np.zeros(cfg.nt)
I2r_p1i0_td = np.zeros(cfg.nt)
I1t_p2i0_td = np.zeros(cfg.nt)

gamma_len = int(cfg.buffer_yhat/5)+1                    # Additional portion of effective half-width
start = cfg.num_cpml + cfg.buffer_yhat - gamma_len      # Effective width lower bound
stop = start + cfg.ny_swg + 2*gamma_len                 # Effective width upper bound
for j in range(start, stop):
    I1i_p2i0_td += hp1i_p2i0_td[:, j] * cfg.delta_x
    I2i_p1i0_td += hp2i_p1i0_td[:, j] * cfg.delta_x
    I1r_p1i0_td += hp1r_p1i0_td[:, j] * cfg.delta_x
    I2r_p2i0_td += hp2r_p2i0_td[:, j] * cfg.delta_x
    I1r_p2i0_td += hp1r_p2i0_td[:, j] * cfg.delta_x
    I2r_p1i0_td += hp2r_p1i0_td[:, j] * cfg.delta_x
    I1t_p2i0_td += hp1t_p2i0_td[:, j] * cfg.delta_x

I1i_p2i0_fft = np.fft.fft(I1i_p2i0_td)                  # Apply FFT to port data
I2i_p1i0_fft = np.fft.fft(I2i_p1i0_td)
I1r_p1i0_fft = np.fft.fft(I1r_p1i0_td)
I2r_p2i0_fft = np.fft.fft(I2r_p2i0_td)
I1r_p2i0_fft = np.fft.fft(I1r_p2i0_td)
I2r_p1i0_fft = np.fft.fft(I2r_p1i0_td)
I1t_p2i0_fft = np.fft.fft(I1t_p2i0_td)

# =============================================================================
# Calculate S-parameters
# =============================================================================

S11 = I1r_p2i0_fft / I1i_p2i0_fft
S12 = I1r_p1i0_fft / I2i_p1i0_fft
S21 = I2r_p2i0_fft / I1i_p2i0_fft
S22 = I2r_p1i0_fft / I2i_p1i0_fft

# =============================================================================
# Plot S-parameter magnitudes
# =============================================================================

ff = np.arange(0, cfg.nt) / (cfg.nt * cfg.delta_t)

fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, sharex=True)
ax1.plot(ff/1e12, 20*np.log10(abs(S12)), c='b', ls='-', label='|S12|')
ax1.plot(ff/1e12, 20*np.log10(abs(S21)), c='r', ls='--', label='|S21|')
ax1.set_xlim([100, 300])
ax1.set_ylim([-0.25, 1])
ax1.grid(which='both')
ax1.legend()
ax1.set_ylabel('Magnitude (dB)')
ax2.plot(ff/1e12, 20*np.log10(abs(S11)), c='b', ls='-', label='|S11|')
ax2.plot(ff/1e12, 20*np.log10(abs(S22)), c='r', ls='--', label='|S22|')
ax2.set_ylim([-170, -140])
ax2.grid(which='both')
ax2.legend()
ax2.set_xlabel('Frequency (THz)')
ax2.set_ylabel('Magnitude (dB)')
fig.tight_layout()