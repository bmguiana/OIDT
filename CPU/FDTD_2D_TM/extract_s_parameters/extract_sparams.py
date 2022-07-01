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
from aux_funcs import find_neff

# =============================================================================
# Load Data
# =============================================================================

ep1i_p2i0_td = np.load('./Results/ez_p1i_p2i0.npy')
ep2i_p1i0_td = np.load('./Results/ez_p2i_p1i0.npy')
ep1t_p2i0_td = np.load('./Results/ez_p1t_p2i0.npy')
ep2t_p1i0_td = np.load('./Results/ez_p2t_p1i0.npy')
ep1r_p1i0_td = np.load('./Results/ez_p1r_p1i0.npy')
ep2r_p2i0_td = np.load('./Results/ez_p2r_p2i0.npy')
ep1r_p2i0_td = ep1t_p2i0_td - ep1i_p2i0_td
ep2r_p1i0_td = ep2t_p1i0_td - ep2i_p1i0_td

# =============================================================================
# Convert E to V
# =============================================================================

V1i_p2i0_td = np.zeros(cfg.nt)
V2i_p1i0_td = np.zeros(cfg.nt)
V1r_p1i0_td = np.zeros(cfg.nt)
V2r_p2i0_td = np.zeros(cfg.nt)
V1r_p2i0_td = np.zeros(cfg.nt)
V2r_p1i0_td = np.zeros(cfg.nt)
V1t_p2i0_td = np.zeros(cfg.nt)

gamma_len = int(cfg.buffer_yhat/5)+1                    # Additional portion of effective half-width
start = cfg.num_cpml + cfg.buffer_yhat - gamma_len      # Effective width lower bound
stop = start + cfg.ny_swg + 2*gamma_len                 # Effective width upper bound
for j in range(start, stop):
    V1i_p2i0_td += ep1i_p2i0_td[:, j] * cfg.delta_x
    V2i_p1i0_td += ep2i_p1i0_td[:, j] * cfg.delta_x
    V1r_p1i0_td += ep1r_p1i0_td[:, j] * cfg.delta_x
    V2r_p2i0_td += ep2r_p2i0_td[:, j] * cfg.delta_x
    V1r_p2i0_td += ep1r_p2i0_td[:, j] * cfg.delta_x
    V2r_p1i0_td += ep2r_p1i0_td[:, j] * cfg.delta_x
    V1t_p2i0_td += ep1t_p2i0_td[:, j] * cfg.delta_x

V1i_p2i0_fd = np.fft.fft(V1i_p2i0_td)
V2i_p1i0_fd = np.fft.fft(V2i_p1i0_td)
V1r_p1i0_fd = np.fft.fft(V1r_p1i0_td)
V2r_p2i0_fd = np.fft.fft(V2r_p2i0_td)
V1r_p2i0_fd = np.fft.fft(V1r_p2i0_td)
V2r_p1i0_fd = np.fft.fft(V2r_p1i0_td)
V1t_p2i0_fd = np.fft.fft(V1t_p2i0_td)

# =============================================================================
# Calculate S-Parameters
# =============================================================================

S11 = V1r_p2i0_fd / V1i_p2i0_fd
S12 = V1r_p1i0_fd / V2i_p1i0_fd
S21 = V2r_p2i0_fd / V1i_p2i0_fd
S22 = V2r_p1i0_fd / V2i_p1i0_fd

# =============================================================================
# Plot results
# =============================================================================

ff = np.arange(0, cfg.nt) / (cfg.nt * cfg.delta_t)

fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, sharex=True)
ax1.plot(ff/1e12, 20*np.log10(abs(S12)), c='b', ls='-', label='|S12|')
ax1.plot(ff/1e12, 20*np.log10(abs(S21)), c='r', ls='--', label='|S21|')
ax1.set_xlim([100, 300])
ax1.set_ylim([-0.5, 0.5])
ax1.grid(which='both')
ax1.legend()
ax1.set_ylabel('Magnitude (dB)')
ax2.plot(ff/1e12, 20*np.log10(abs(S11)), c='b', ls='-', label='|S11|')
ax2.plot(ff/1e12, 20*np.log10(abs(S22)), c='r', ls='--', label='|S22|')
ax2.set_ylim([-155, -135])
ax2.grid(which='both')
ax2.legend()
ax2.set_xlabel('Frequency (THz)')
ax2.set_ylabel('Magnitude (dB)')
fig.tight_layout()
