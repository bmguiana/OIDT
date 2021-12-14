# -*- coding: utf-8 -*-
"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import os
import psutil
from time import time
from fdtd_config import *
from fdtd_funcs import *

# =============================================================================
# Main array definition
# =============================================================================

EZ = np.zeros([nx, ny])         # Electric field at current time step, z-component
HX = np.zeros([nx, ny])         # Magnetic field at current time step, x-component
HY = np.zeros([nx, ny])         # Magnetic field at current time step, y-component

# Create computational domain mask
FG_REG = gen_fg_mask_sgl(nx, ny, by, ny_swg, p1x, rstd, racl, delta_x, roughness_profile, mode='smooth', atol=0.10, stol=0.10, mtol=0.01)
SIGMA_MASK = np.zeros([nx, ny])
EPS_MASK = eps * ( eps_rel_fg * FG_REG + 1 * ~FG_REG )

# =============================================================================
# Auxiliary array definition
# =============================================================================

# Curl arrays
CURL_E = delta_t / ( mu * delta_x )
CURL_H = 2 * delta_t / ( delta_x * ( 2 * EPS_MASK + SIGMA_MASK * delta_t ) )
MOD_E = ( 2 * EPS_MASK - SIGMA_MASK * delta_t ) / ( 2 * EPS_MASK + SIGMA_MASK * delta_t )

# =============================================================================
# Output definition
# =============================================================================

# Define data storage arrays
EZ_ZWAVE = np.zeros([nt, ny])
HX_ZWAVE = np.zeros([nt, ny])

# =============================================================================
# CPML setup
# =============================================================================

# Constants
m_grade = 4                                         # Grading maximum exponent for sigma and kappa
m_alpha = 1                                         # Grading maximum exponent for alpha
cpml_sigma_optimal = 0.8*(m_grade+1)/(eta*delta_x)  # Optimal sigma value

cpml_sigma_max = 1.2*cpml_sigma_optimal             # Maximum sigma value
cpml_alpha_max = 0.05                               # Maximum alpha value
if True:
    cpml_kappa_max = 5                              # Maximum kappa value if CPML is on
else:
    cpml_kappa_max = 1                              # Maximum kappa value if CPML is off

# Define arrays
PSI_EZ_XLO = np.zeros([cpml_range, ny])             # Ez correction field, low-side x-boundary
PSI_EZ_XHI = np.zeros([cpml_range, ny])             # Ez correction field, high-side x-boundary
PSI_EZ_YLO = np.zeros([nx, cpml_range])             # Ez correction field, low-side y-boundary
PSI_EZ_YHI = np.zeros([nx, cpml_range])             # Ez correction field, high-side y-boundary
PSI_HX_YLO = np.zeros([nx, cpml_range])             # Hx correction field, low-side y-boundary
PSI_HX_YHI = np.zeros([nx, cpml_range])             # Hx correction field, high-side y-boundary
PSI_HY_XLO = np.zeros([cpml_range, ny])             # Hy correction field, low-side x-boundary
PSI_HY_XHI = np.zeros([cpml_range, ny])             # Hy correction field, high-side x-boundary

AE_CPML = np.zeros(cpml_range)                      # Electric field alpha grading
KE_CPML = np.zeros(cpml_range)                      # Electric field kappa grading
SE_CPML = np.zeros(cpml_range)                      # Electric field sigma grading

AH_CPML = np.zeros(cpml_range)                      # Magnetic field alpha grading
KH_CPML = np.zeros(cpml_range)                      # Magnetic field kappa grading
SH_CPML = np.zeros(cpml_range)                      # Magnetic field sigma grading

BE = np.zeros(cpml_range)                           # Electric field auxiliary variable b
CE = np.zeros(cpml_range)                           # Electric field auxiliary variable c
BH = np.zeros(cpml_range)                           # Magnetic field auxiliary variable b
CH = np.zeros(cpml_range)                           # Magnetic field auxiliary variable c

DEN_EX = np.ones(nx)                                # Electric field x-direction kappa division
DEN_EY = np.ones(ny)                                # Electric field y-direction kappa division
DEN_HX = np.ones(nx)                                # Magnetic field x-direction kappa division
DEN_HY = np.ones(ny)                                # Magnetic field y-direction kappa division

# Assign array values
for q in range(num_cpml):
    AE_CPML[q] = cpml_alpha_max * ( q / num_cpml )**m_alpha
    KE_CPML[q] = 1 + ( cpml_kappa_max - 1 ) * ( ( cpml_range - q - 1 ) / num_cpml )**m_grade
    SE_CPML[q] = cpml_sigma_max * ( ( cpml_range - q - 1 ) / num_cpml )**m_grade
    BE[q] = np.exp( -1 * ( SE_CPML[q] / KE_CPML[q] + AE_CPML[q] ) * delta_t / eps )
    CE[q] = SE_CPML[q] / ( SE_CPML[q] + KE_CPML[q] * AE_CPML[q] ) / KE_CPML[q] * ( BE[q] - 1 )

    AH_CPML[q] = cpml_alpha_max * ( ( q + 0.5 ) / num_cpml )**m_alpha
    KH_CPML[q] = 1 + ( cpml_kappa_max - 1 ) * ( ( cpml_range - q - 1.5 ) / num_cpml )**m_grade
    SH_CPML[q] = cpml_sigma_max * ( ( cpml_range - q - 1.5 ) / num_cpml )**m_grade
    BH[q] = np.exp( -1 * ( SH_CPML[q] / KH_CPML[q] + AH_CPML[q] ) * delta_t / eps )
    CH[q] = SH_CPML[q] / ( SH_CPML[q] + KH_CPML[q] * AH_CPML[q] ) / KH_CPML[q] * ( BH[q] - 1 )

    DEN_EX[q] = KE_CPML[q]
    DEN_EX[nx-1-q] = KE_CPML[q]
    DEN_EY[q] = KE_CPML[q]
    DEN_EY[ny-1-q] = KE_CPML[q]
    DEN_HX[q] = KH_CPML[q]
    DEN_HX[nx-2-q] = KH_CPML[q]
    DEN_HY[q] = KH_CPML[q]
    DEN_HY[ny-2-q] = KH_CPML[q]

# =============================================================================
# Time stepping loop
# =============================================================================

process = psutil.Process(os.getpid())   # Get current process ID
loop_start_time = time()                # Start loop timing
last40 = np.zeros(40)
cu_time = time() - loop_start_time

for n in range(nt):
    EZ = update_ez(EZ, HX, HY, MOD_E, CURL_H, DEN_EX, DEN_EY, nx, ny)
    EZ[sx, by:(ny-by)] += J_SRC[n]
    PSI_EZ_XLO, PSI_EZ_XHI, EZ = update_ez_cpml_x(PSI_EZ_XLO, PSI_EZ_XHI, EZ, HX, HY, CURL_H, BE, CE, delta_x, delta_t, eps, nx, ny, cpml_range)
    PSI_EZ_YLO, PSI_EZ_YHI, EZ = update_ez_cpml_y(PSI_EZ_YLO, PSI_EZ_YHI, EZ, HX, HY, CURL_H, BE, CE, delta_x, delta_t, eps, nx, ny, cpml_range)
    
    HX, HY = update_hx_hy(HX, HY, EZ, CURL_E, DEN_HX, DEN_HY, nx, ny)
    PSI_HY_XLO, PSI_HY_XHI, HY = update_hy_cpml_x(PSI_HY_XLO, PSI_HY_XHI, HY, EZ, BH, CH, delta_x, delta_t, mu, nx, ny, cpml_range)
    PSI_HX_YLO, PSI_HX_YHI, HX = update_hx_cpml_y(PSI_HX_YLO, PSI_HX_YHI, HX, EZ, BH, CH, delta_x, delta_t, mu, nx, ny, cpml_range)

    # Progress feedback
    last40 = np.roll(last40, 1)
    iter_time = time() - cu_time - loop_start_time
    cu_time = time() - loop_start_time
    last40[0] = iter_time
    if n % status_readout_interval == 0:
        avg_iter_time = np.average(last40)
        cu_time = time() - loop_start_time
        time_rem = avg_iter_time * ( nt - n - 1 )
        print('\nStep {} of {} done, {:.3f} % complete'.format(n+1, nt, n/(nt-2)*100))
        print('Loop time elapsed:         {} (hr) {} (min) {:.1f} (s)'.format(int(cu_time/3600), int((cu_time - 3600*(cu_time//3600))//60), cu_time - 60*((cu_time - 3600*(cu_time//3600))//60) - 3600*(cu_time//3600)))
        print('Avg. loop period:          {:.2f} (ms)'.format(avg_iter_time*1000))
        print('Estimated time remaining:  {} (hr) {} (min) {:.1f} (s)'.format(int(time_rem/3600), int((time_rem - 3600*(time_rem//3600))//60), time_rem - 60*((time_rem - 3600*(time_rem//3600))//60) - 3600*(time_rem//3600)))
        print('Memory used:              {:6.3f} (GB)'.format(process.memory_info().rss/1024/1024/1024))

    # Record Data
    EZ_ZWAVE[n, :] = EZ[cx, :]
    HX_ZWAVE[n, :] = HX[cx, :]

# =============================================================================
# Save storage arrays to files
# =============================================================================

np.save('./'+output_dir+'/ez_zwave', EZ_ZWAVE)
np.save('./'+output_dir+'/hx_zwave', HX_ZWAVE)
