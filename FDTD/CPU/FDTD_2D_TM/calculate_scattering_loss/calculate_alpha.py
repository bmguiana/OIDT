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
rprof = '_s15_lc300_r0'

ep1i_p2i0_td = np.load('./Results/ez_p1i_p2i0'+rprof+'.npy')
ep2i_p1i0_td = np.load('./Results/ez_p2i_p1i0'+rprof+'.npy')
ep1t_p2i0_td = np.load('./Results/ez_p1t_p2i0'+rprof+'.npy')
ep2t_p1i0_td = np.load('./Results/ez_p2t_p1i0'+rprof+'.npy')
ep1r_p1i0_td = np.load('./Results/ez_p1r_p1i0'+rprof+'.npy')
ep2r_p2i0_td = np.load('./Results/ez_p2r_p2i0'+rprof+'.npy')
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
# Calculate alpha
# =============================================================================

ell_cm = (cfg.sgl_wg_length - 2*cfg.port_length)*100
ff = np.arange(0, cfg.nt) / (cfg.nt * cfg.delta_t)
alpha_s = 8.686 * np.log( np.abs( 1 + S11) * np.abs(S12) / np.abs(S11 + S12*S21) ) / ell_cm
gamma = -1 * np.log( V2r_p2i0_fd /  V1t_p2i0_fd )
alpha_e = 8.686 * gamma.real / ell_cm

# =============================================================================
# Filter the scattering loss
# =============================================================================

alpha_ef = np.zeros_like(alpha_e)
alpha_sf = np.zeros_like(alpha_s)
samples = 10
for i in range(cfg.nt-samples//2):
    alpha_ef[i] = np.average(alpha_e[i-samples//2:i+samples//2+1])
    alpha_sf[i] = np.average(alpha_s[i-samples//2:i+samples//2+1])
    if np.isnan(alpha_ef[i]):
        alpha_ef[i] = alpha_e[i]
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
ax.plot(ff/1e12, alpha_e, c='b', ls=':', label=r'$\alpha_{dir}$')
ax.plot(ff/1e12, alpha_ef, c='b', ls='-', label=r'$\alpha_{dir}$ filtered')
ax.set_xlim([100, 300])
ax.set_ylim([-1000, 5000])
ax.grid()
ax.legend()
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Scattering Loss (dB/cm)')
fig.tight_layout()
