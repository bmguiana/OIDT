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
rprof = '_s15_lc300_r0'

hp1i_p2i0_td = np.load('./Results/hz_p1i_p2i0'+rprof+'.npy')
hp2i_p1i0_td = np.load('./Results/hz_p2i_p1i0'+rprof+'.npy')
hp1t_p2i0_td = np.load('./Results/hz_p1t_p2i0'+rprof+'.npy')
hp2t_p1i0_td = np.load('./Results/hz_p2t_p1i0'+rprof+'.npy')
hp1r_p1i0_td = np.load('./Results/hz_p1r_p1i0'+rprof+'.npy')
hp2r_p2i0_td = np.load('./Results/hz_p2r_p2i0'+rprof+'.npy')
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
# Calculate alpha
# =============================================================================

ell_cm = (cfg.sgl_wg_length - 2*cfg.port_length)*100
ff = np.arange(0, cfg.nt) / (cfg.nt * cfg.delta_t)
alpha_s = 8.686 * np.log( np.abs( 1 + S11) * np.abs(S12) / np.abs(S11 + S12*S21) ) / ell_cm
gamma = -1 * np.log( I2r_p2i0_fft /  I1t_p2i0_fft )
alpha_h = 8.686 * gamma.real / ell_cm

# =============================================================================
# Filter the scattering loss
# =============================================================================

alpha_hf = np.zeros_like(alpha_h)
alpha_sf = np.zeros_like(alpha_s)
samples = 10
for i in range(cfg.nt-samples//2):
    alpha_hf[i] = np.average(alpha_h[i-samples//2:i+samples//2+1])
    alpha_sf[i] = np.average(alpha_s[i-samples//2:i+samples//2+1])
    if np.isnan(alpha_hf[i]):
        alpha_hf[i] = alpha_h[i]
    if np.isnan(alpha_sf[i]):
        alpha_sf[i] = alpha_s[i]

# =============================================================================
# Plot Results
#   alpha_s is calculated from scattering parameters
#   alpha_dir is calculated from direct measurement comparison
# =============================================================================
fig, ax = plt.subplots()
ax.plot(ff/1e12, alpha_s, c='r', ls=':', label=r'$\alpha_{s}$')
ax.plot(ff/1e12, alpha_sf, c='r', ls='-', label=r'$\alpha_{s}$ filtered')
ax.plot(ff/1e12, alpha_h, c='b', ls=':', label=r'$\alpha_{dir}$')
ax.plot(ff/1e12, alpha_hf, c='b', ls='-', label=r'$\alpha_{dir}$ filtered')
ax.set_xlim([100, 300])
ax.set_ylim([-1000, 4000])
ax.grid()
ax.legend()
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Scattering Loss (dB/cm)')
fig.tight_layout()
