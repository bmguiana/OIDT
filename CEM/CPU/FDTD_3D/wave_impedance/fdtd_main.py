# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:31:09 2021

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

EX = np.zeros([nx, ny, nz])     # Electric field at current time step, x-component
EY = np.zeros([nx, ny, nz])     # Electric field at current time step, y-component
EZ = np.zeros([nx, ny, nz])     # Electric field at current time step, z-component
HX = np.zeros([nx, ny, nz])     # Magnetic field at current time step, x-component
HY = np.zeros([nx, ny, nz])     # Magnetic field at current time step, y-component
HZ = np.zeros([nx, ny, nz])     # Magnetic field at current time step, z-component

# Create a computational domain mask
FG_REG = generate_fg_mask(nx, ny, nz, bx, by, nx_swg, ny_swg, p1z, rstd, racl, delta_x, roughness_profile, mode='smooth')
SIGMA_MASK = np.zeros([nx, ny, nz])
EPS_MASK = np.zeros([nx, ny, nz])

# =============================================================================
# Auxiliary array definition
# =============================================================================

# tf/sf arrays
E_INC = np.zeros(nz)            # Incident electric field at current time step
H_INC = np.zeros(nz)            # Incident magnetic field at current time step

# Curl arrays
EPS_MASK = eps * ( eps_rel_fg * FG_REG + 1 * ~FG_REG )
MOD_E = ( 2 * EPS_MASK - SIGMA_MASK * delta_t ) / ( 2 * EPS_MASK + SIGMA_MASK * delta_t )
CURL_H = 2 * delta_t / ( delta_x * ( 2 * EPS_MASK + SIGMA_MASK * delta_t ) )
CURL_E = delta_t / ( mu * delta_x )

# =============================================================================
# Output definition
# =============================================================================

# Define data storage arrays
EX_ZW = np.zeros([nt, nx, ny])
EY_ZW = np.zeros([nt, nx, ny])
EZ_ZW = np.zeros([nt, nx, ny])
HX_ZW = np.zeros([nt, nx, ny])
HY_ZW = np.zeros([nt, nx, ny])
HZ_ZW = np.zeros([nt, nx, ny])

# =============================================================================
# Begin CPML
# =============================================================================

# Constants
m_grade = 4                                     # Grading maximum exponent for sigma and kappa
m_alpha = 1                                     # Grading maximum exponent for alpha
cpml_sigma_optimal = 0.8 * ( m_grade + 1 ) / ( eta * delta_x )  # Optimal sigma value
cpml_sigma_max = 1.2 * cpml_sigma_optimal       # Maximum sigma value
cpml_alpha_max = 0.05                           # Maximum alpha value
if cpml_toggle:
    cpml_kappa_max = 5                          # Maximum kappa value if CPML is on
else:
    cpml_kappa_max = 1                          # Maximum kappa value if CPML is off

# Define arrays
PSI_EX_YLO = np.zeros([nx, cpml_range, nz])     # Ex correction field, low-side y-boundary
PSI_EX_YHI = np.zeros([nx, cpml_range, nz])     # Ex correction field, high-side y-boundary
PSI_EX_ZLO = np.zeros([nx, ny, cpml_range])     # Ex correction field, low-side z-boundary
PSI_EX_ZHI = np.zeros([nx, ny, cpml_range])     # Ex correction field, high-side z-boundary

PSI_EY_XLO = np.zeros([cpml_range, ny, nz])     # Ey correction field, low-side x-boundary
PSI_EY_XHI = np.zeros([cpml_range, ny, nz])     # Ey correction field, high-side x-boundary
PSI_EY_ZLO = np.zeros([nx, ny, cpml_range])     # Ey correction field, low-side z-boundary
PSI_EY_ZHI = np.zeros([nx, ny, cpml_range])     # Ey correction field, high-side z-boundary

PSI_EZ_XLO = np.zeros([cpml_range, ny, nz])     # Ez correction field, low-side x-boundary
PSI_EZ_XHI = np.zeros([cpml_range, ny, nz])     # Ez correction field, high-side x-boundary
PSI_EZ_YLO = np.zeros([nx, cpml_range, nz])     # Ez correction field, low-side y-boundary
PSI_EZ_YHI = np.zeros([nx, cpml_range, nz])     # Ez correction field, high-side y-boundary

PSI_HX_YLO = np.zeros([nx, cpml_range, nz])     # Hx correction field, low-side y-boundary
PSI_HX_YHI = np.zeros([nx, cpml_range, nz])     # Hx correction field, high-side y-boundary
PSI_HX_ZLO = np.zeros([nx, ny, cpml_range])     # Hx correction field, low-side z-boundary
PSI_HX_ZHI = np.zeros([nx, ny, cpml_range])     # Hx correction field, high-side z-boundary

PSI_HY_XLO = np.zeros([cpml_range, ny, nz])     # Hy correction field, low-side x-boundary
PSI_HY_XHI = np.zeros([cpml_range, ny, nz])     # Hy correction field, high-side x-boundary
PSI_HY_ZLO = np.zeros([nx, ny, cpml_range])     # Hy correction field, low-side z-boundary
PSI_HY_ZHI = np.zeros([nx, ny, cpml_range])     # Hy correction field, high-side z-boundary

PSI_HZ_XLO = np.zeros([cpml_range, ny, nz])     # Hz correction field, low-side x-boundary
PSI_HZ_XHI = np.zeros([cpml_range, ny, nz])     # Hz correction field, high-side x-boundary
PSI_HZ_YLO = np.zeros([nx, cpml_range, nz])     # Hz correction field, low-side y-boundary
PSI_HZ_YHI = np.zeros([nx, cpml_range, nz])     # Hz correction field, high-side y-boundary

PSI_E_INC_Z1 = np.zeros(cpml_range)             # Incident E correction field, low-side boundary
PSI_E_INC_Z2 = np.zeros(cpml_range)             # Incident E correction field, high-side boundary
PSI_H_INC_Z1 = np.zeros(cpml_range)             # Incident H correction field, low-side boundary
PSI_H_INC_Z2 = np.zeros(cpml_range)             # Incident H correction field, high-side boundary

AE_CPML = np.zeros(cpml_range)                  # Electric field alpha grading
KE_CPML = np.zeros(cpml_range)                  # Electric field kappa grading
SE_CPML = np.zeros(cpml_range)                  # Electric field sigma grading

AH_CPML = np.zeros(cpml_range)                  # Magnetic field alpha grading
KH_CPML = np.zeros(cpml_range)                  # Magnetic field kappa grading
SH_CPML = np.zeros(cpml_range)                  # Magnetic field sigma grading

BE = np.zeros(cpml_range)                       # Electric field auxiliary variable b
CE = np.zeros(cpml_range)                       # Electric field auxiliary variable c
BH = np.zeros(cpml_range)                       # Magnetic field auxiliary variable b
CH = np.zeros(cpml_range)                       # Magnetic field auxiliary variable c

DEN_EX = np.ones(nx)                            # Electric field x-direction kappa division
DEN_EY = np.ones(ny)                            # Electric field y-direction kappa division
DEN_EZ = np.ones(nz)                            # Electric field z-direction kappa division
DEN_HX = np.ones(nx)                            # Magnetic field x-direction kappa division
DEN_HY = np.ones(ny)                            # Magnetic field y-direction kappa division
DEN_HZ = np.ones(nz)                            # Magnetic field z-direction kappa division

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
    DEN_EZ[q] = KE_CPML[q]
    DEN_EZ[nz-1-q] = KE_CPML[q]
    DEN_HX[q] = KH_CPML[q]
    DEN_HX[nx-2-q] = KH_CPML[q]
    DEN_HY[q] = KH_CPML[q]
    DEN_HY[ny-2-q] = KH_CPML[q]
    DEN_HZ[q] = KH_CPML[q]
    DEN_HZ[nz-2-q] = KH_CPML[q]


# =============================================================================
# Time stepping loop
# =============================================================================

process = psutil.Process(os.getpid())   # Get current process ID
loop_start_time = time()                # Start loop timing

for n in range(nt):
    EX, EY, EZ = update_e(EX, EY, EZ, HX, HY, HZ, MOD_E, CURL_H, DEN_EX, DEN_EY, DEN_EZ, nx, ny, nz)
    EX[(bx+1):(bx+nx_swg-1), (by+1):(by+ny_swg-1), sz] += J_SRC[n]
    PSI_EX_YLO, PSI_EX_YHI, PSI_EX_ZLO, PSI_EX_ZHI, EX = update_ex_cpml(PSI_EX_YLO, PSI_EX_YHI, PSI_EX_ZLO, PSI_EX_ZHI, EX, HY, HZ, CURL_H, BE, CE, delta_x, delta_t, eps, nx, ny, nz, cpml_range)
    PSI_EY_XLO, PSI_EY_XHI, PSI_EY_ZLO, PSI_EY_ZHI, EY = update_ey_cpml(PSI_EY_XLO, PSI_EY_XHI, PSI_EY_ZLO, PSI_EY_ZHI, EY, HX, HZ, CURL_H, BE, CE, delta_x, delta_t, eps, nx, ny, nz, cpml_range)
    PSI_EZ_XLO, PSI_EZ_XHI, PSI_EZ_YLO, PSI_EZ_YHI, EZ = update_ez_cpml(PSI_EZ_XLO, PSI_EZ_XHI, PSI_EZ_YLO, PSI_EZ_YHI, EZ, HX, HY, CURL_H, BE, CE, delta_x, delta_t, eps, nx, ny, nz, cpml_range)

    HX, HY, HZ = update_h(EX, EY, EZ, HX, HY, HZ, CURL_E, DEN_HX, DEN_HY, DEN_HZ, nx, ny, nz)
    PSI_HX_YLO, PSI_HX_YHI, PSI_HX_ZLO, PSI_HX_ZHI, HX = update_hx_cpml(PSI_HX_YLO, PSI_HX_YHI, PSI_HX_ZLO, PSI_HX_ZHI, HX, EY, EZ, BH, CH, delta_x, delta_t, mu, nx, ny, nz, cpml_range)
    PSI_HY_XLO, PSI_HY_XHI, PSI_HY_ZLO, PSI_HY_ZHI, HY = update_hy_cpml(PSI_HY_XLO, PSI_HY_XHI, PSI_HY_ZLO, PSI_HY_ZHI, HY, EX, EZ, BH, CH, delta_x, delta_t, mu, nx, ny, nz, cpml_range)
    PSI_HZ_XLO, PSI_HZ_XHI, PSI_HZ_YLO, PSI_HZ_YHI, HZ = update_hz_cpml(PSI_HZ_XLO, PSI_HZ_XHI, PSI_HZ_YLO, PSI_HZ_YHI, HZ, EX, EY, BH, CH, delta_x, delta_t, mu, nx, ny, nz, cpml_range)

    # Progress feedback
    if n % status_readout_interval == 0:
        cu_time = time() - loop_start_time
        avg_iter_time = cu_time / ( n + 1 )
        time_rem = avg_iter_time * ( nt - n - 1 )
        print('\nStep {} of {} done, {:.3f} % complete'.format(n+1, nt, n/(nt-2)*100))
        print('Loop time elapsed:         {} (hr) {} (min) {:.1f} (s)'.format(int(cu_time/3600), int((cu_time - 3600*(cu_time//3600))//60), cu_time - 60*((cu_time - 3600*(cu_time//3600))//60) - 3600*(cu_time//3600)))
        print('Avg. loop period:          {:.2f} (ms)'.format(avg_iter_time*1000))
        print('Estimated time remaining:  {} (hr) {} (min) {:.1f} (s)'.format(int(time_rem/3600), int((time_rem - 3600*(time_rem//3600))//60), time_rem - 60*((time_rem - 3600*(time_rem//3600))//60) - 3600*(time_rem//3600)))
        print('Memory used:              {:6.3f} (GB)'.format(process.memory_info().rss/1024/1024/1024))

    # Record Data
    EX_ZW[n, :, :] = EX[:, :, cz]
    EY_ZW[n, :, :] = EY[:, :, cz]
    EZ_ZW[n, :, :] = EZ[:, :, cz]
    HX_ZW[n, :, :] = HX[:, :, cz]
    HY_ZW[n, :, :] = HY[:, :, cz]
    HZ_ZW[n, :, :] = HZ[:, :, cz]


# =============================================================================
# Save storage arrays to files
# =============================================================================

np.save('./'+output_dir+'/ex_zwave', EX_ZW)
np.save('./'+output_dir+'/ey_zwave', EY_ZW)
np.save('./'+output_dir+'/ez_zwave', EZ_ZW)
np.save('./'+output_dir+'/hx_zwave', HX_ZW)
np.save('./'+output_dir+'/hy_zwave', HY_ZW)
np.save('./'+output_dir+'/hz_zwave', HZ_ZW)
