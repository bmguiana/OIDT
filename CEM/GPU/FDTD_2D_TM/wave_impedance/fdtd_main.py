"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import os
import psutil
from time import time
from numba import cuda

import fdtd_config as cfg
import fdtd_funcs as funcs
import aux_funcs as aux

print('The simulation size will be: {:} x {:} cubic cells over {:} time steps\n\n\n'.format(cfg.nx, cfg.ny, cfg.nt))
aux.mkdir(cfg.output_dir)
cuda.select_device(0)

# =============================================================================
# Main array definition
# =============================================================================

EZ = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)         # Electric field at current time step, z-component
HX = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)         # Magnetic field at current time step, x-component
HY = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)         # Magnetic field at current time step, y-component

# Assign Materials
if cfg.rough_toggle:
    FG_REG = funcs.gen_fg_mask_sgl(cfg.nx, cfg.ny, cfg.by, cfg.ny_swg, cfg.p1x, cfg.rstd, cfg.racl, cfg.delta_x, mode='gen', correlation=cfg.ctype, upper_path=cfg.up, lower_path=cfg.lp, atol=cfg.tol_acl, stol=cfg.tol_std, mtol=0.01)
else:
    FG_REG = funcs.gen_fg_mask_sgl(cfg.nx, cfg.ny, cfg.by, cfg.ny_swg, cfg.p1x, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth')

SIGMA_MASK = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)
EPS_MASK = cfg.eps * ( cfg.eps_rel_fg * FG_REG + 1 * ~FG_REG )
EPS_MASK = EPS_MASK.astype(np.float32)

# Curl arrays
CURL_E = cfg.delta_t / ( cfg.mu * cfg.delta_x )
CURL_H = 2 * cfg.delta_t / ( cfg.delta_x * ( 2 * EPS_MASK + SIGMA_MASK * cfg.delta_t ) )
MOD_E = ( 2 * EPS_MASK - SIGMA_MASK * cfg.delta_t ) / ( 2 * EPS_MASK + SIGMA_MASK * cfg.delta_t )

CURL_E = CURL_E.astype(np.float32)
CURL_H = CURL_H.astype(np.float32)
MOD_E = MOD_E.astype(np.float32)

# Define data storage arrays
EZT = np.zeros([cfg.nt, cfg.ny], dtype=np.float32)
HXT = np.zeros([cfg.nt, cfg.ny], dtype=np.float32)

# =============================================================================
# CPML setup
# =============================================================================

# Constants
m_grade = 4                                                        # Grading maximum exponent for sigma and kappa
m_alpha = 1                                                        # Grading maximum exponent for alpha
cpml_sigma_optimal = 0.8*(m_grade+1)/(cfg.eta*cfg.delta_x)         # Optimal sigma value

cpml_sigma_max = 1.2*cpml_sigma_optimal                            # Maximum sigma value
cpml_alpha_max = 0.05                                              # Maximum alpha value
cpml_kappa_max = 5                                                 # Maximum kappa value if CPML is on

# Define arrays
PSI_EZ_XLO = np.zeros([cfg.cpml_range, cfg.ny], dtype=np.float32)  # Ez correction field, low-side x-boundary
PSI_EZ_XHI = np.zeros([cfg.cpml_range, cfg.ny], dtype=np.float32)  # Ez correction field, high-side x-boundary
PSI_EZ_YLO = np.zeros([cfg.nx, cfg.cpml_range], dtype=np.float32)  # Ez correction field, low-side y-boundary
PSI_EZ_YHI = np.zeros([cfg.nx, cfg.cpml_range], dtype=np.float32)  # Ez correction field, high-side y-boundary
PSI_HX_YLO = np.zeros([cfg.nx, cfg.cpml_range], dtype=np.float32)  # Hx correction field, low-side y-boundary
PSI_HX_YHI = np.zeros([cfg.nx, cfg.cpml_range], dtype=np.float32)  # Hx correction field, high-side y-boundary
PSI_HY_XLO = np.zeros([cfg.cpml_range, cfg.ny], dtype=np.float32)  # Hy correction field, low-side x-boundary
PSI_HY_XHI = np.zeros([cfg.cpml_range, cfg.ny], dtype=np.float32)  # Hy correction field, high-side x-boundary

AE_CPML = np.zeros(cfg.cpml_range, dtype=np.float32)               # Electric field alpha grading
KE_CPML = np.zeros(cfg.cpml_range, dtype=np.float32)               # Electric field kappa grading
SE_CPML = np.zeros(cfg.cpml_range, dtype=np.float32)               # Electric field sigma grading

AH_CPML = np.zeros(cfg.cpml_range, dtype=np.float32)               # Magnetic field alpha grading
KH_CPML = np.zeros(cfg.cpml_range, dtype=np.float32)               # Magnetic field kappa grading
SH_CPML = np.zeros(cfg.cpml_range, dtype=np.float32)               # Magnetic field sigma grading

BE = np.zeros(cfg.cpml_range, dtype=np.float32)                    # Electric field auxiliary variable b
CE = np.zeros(cfg.cpml_range, dtype=np.float32)                    # Electric field auxiliary variable c
BH = np.zeros(cfg.cpml_range, dtype=np.float32)                    # Magnetic field auxiliary variable b
CH = np.zeros(cfg.cpml_range, dtype=np.float32)                    # Magnetic field auxiliary variable c

DEN_EX = np.ones(cfg.nx, dtype=np.float32)                         # Electric field x-direction kappa division
DEN_EY = np.ones(cfg.ny, dtype=np.float32)                         # Electric field y-direction kappa division
DEN_HX = np.ones(cfg.nx, dtype=np.float32)                         # Magnetic field x-direction kappa division
DEN_HY = np.ones(cfg.ny, dtype=np.float32)                         # Magnetic field y-direction kappa division

# Assign array values
for q in range(cfg.num_cpml):
    AE_CPML[q] = cpml_alpha_max * ( q / cfg.num_cpml )**m_alpha
    KE_CPML[q] = 1 + ( cpml_kappa_max - 1 ) * ( ( cfg.cpml_range - q - 1 ) / cfg.num_cpml )**m_grade
    SE_CPML[q] = cpml_sigma_max * ( ( cfg.cpml_range - q - 1 ) / cfg.num_cpml )**m_grade
    BE[q] = np.exp( -1 * ( SE_CPML[q] / KE_CPML[q] + AE_CPML[q] ) * cfg.delta_t / cfg.eps )
    CE[q] = SE_CPML[q] / ( SE_CPML[q] + KE_CPML[q] * AE_CPML[q] ) / KE_CPML[q] * ( BE[q] - 1 )

    AH_CPML[q] = cpml_alpha_max * ( ( q + 0.5 ) / cfg.num_cpml )**m_alpha
    KH_CPML[q] = 1 + ( cpml_kappa_max - 1 ) * ( ( cfg.cpml_range - q - 1.5 ) / cfg.num_cpml )**m_grade
    SH_CPML[q] = cpml_sigma_max * ( ( cfg.cpml_range - q - 1.5 ) / cfg.num_cpml )**m_grade
    BH[q] = np.exp( -1 * ( SH_CPML[q] / KH_CPML[q] + AH_CPML[q] ) * cfg.delta_t / cfg.eps )
    CH[q] = SH_CPML[q] / ( SH_CPML[q] + KH_CPML[q] * AH_CPML[q] ) / KH_CPML[q] * ( BH[q] - 1 )

    DEN_EX[q] = KE_CPML[q]
    DEN_EX[cfg.nx-1-q] = KE_CPML[q]
    DEN_EY[q] = KE_CPML[q]
    DEN_EY[cfg.ny-1-q] = KE_CPML[q]
    DEN_HX[q] = KH_CPML[q]
    DEN_HX[cfg.nx-2-q] = KH_CPML[q]
    DEN_HY[q] = KH_CPML[q]
    DEN_HY[cfg.ny-2-q] = KH_CPML[q]

# =============================================================================
# GPU Processing Setup
# =============================================================================

# Declare thread sizes
tpbx = 1
tpby = 128

# Declare block sizes
bpgx = int(np.ceil(cfg.nx/tpbx))
bpgy = int(np.ceil(cfg.ny/tpby))
bpgx_pml = int(np.ceil(cfg.cpml_range/tpbx))
bpgy_pml = int(np.ceil(cfg.cpml_range/tpby))

# Combine into tuples
tpb = (tpbx, tpby)
bpg = (bpgx, bpgy)
bpg_pmlx = (bpgx_pml, bpgy)
bpg_pmly = (bpgx, bpgy_pml)

# GPU Process feedback
cells_on_device = 5*(cfg.nx*cfg.ny) + 4*(cfg.cpml_range + cfg.cpml_range*cfg.nx + cfg.cpml_range*cfg.ny) + 2*(cfg.nx + cfg.ny + cfg.nt*cfg.ny)
device_req_mem = 4*cells_on_device / 1024 / 1024
print('Transferring {:.0f} Mcells onto GPU, requiring {:} MB'.format(cells_on_device/1e6, device_req_mem))

# Device arrays of size (nx*ny)
dEZ = cuda.to_device(EZ)
dHX = cuda.to_device(HX)
dHY = cuda.to_device(HY)
dMOD_E = cuda.to_device(MOD_E)
dCURL_H = cuda.to_device(CURL_H)

# Device arrays of size (cpml_range)
dBE = cuda.to_device(BE)
dCE = cuda.to_device(CE)
dBH = cuda.to_device(BH)
dCH = cuda.to_device(CH)

# Device arrays of size (nx)
dDEN_EX = cuda.to_device(DEN_EX)
dDEN_HX = cuda.to_device(DEN_HX)

# Device arrays of size (ny)
dDEN_EY = cuda.to_device(DEN_EY)
dDEN_HY = cuda.to_device(DEN_HY)

# Device arrays of size (cpml_range*nx)
dPSI_EZ_YLO = cuda.to_device(PSI_EZ_YLO)
dPSI_EZ_YHI = cuda.to_device(PSI_EZ_YHI)
dPSI_HX_YLO = cuda.to_device(PSI_HX_YLO)
dPSI_HX_YHI = cuda.to_device(PSI_HX_YHI)

# Device arrays of size (cpml_range*ny)
dPSI_EZ_XLO = cuda.to_device(PSI_EZ_XLO)
dPSI_EZ_XHI = cuda.to_device(PSI_EZ_XHI)
dPSI_HY_XLO = cuda.to_device(PSI_HY_XLO)
dPSI_HY_XHI = cuda.to_device(PSI_HY_XHI)

# Device arrays of size (nt*ny)
dEZT = cuda.to_device(EZT)
dHXT = cuda.to_device(HXT)

# =============================================================================
# Time stepping loop
# =============================================================================

feedback_interval = np.round(np.linspace(0, cfg.nt, num=101))
process = psutil.Process(os.getpid())
last40 = np.zeros(40)
loop_start_time = time()
cu_time = time() - loop_start_time

print('loop started')
for n in range(cfg.nt):
    # Update E-field component
    funcs.update_ez[bpg, tpb](dEZ, dHX, dHY, dMOD_E, dCURL_H, dDEN_EX, dDEN_EY, cfg.nx, cfg.ny)

    # Update source condition
    funcs.update_source[bpg, tpb](dEZ, cfg.J_SRC[n], cfg.sx, cfg.sx+1, cfg.by, cfg.ny-cfg.by)

    # Update E-field component in PML
    funcs.update_ez_cpml_x[bpg_pmlx, tpb](dPSI_EZ_XLO, dPSI_EZ_XHI, dEZ, dHX, dHY, dCURL_H, dBE, dCE, cfg.delta_x, cfg.delta_t, cfg.eps, cfg.nx, cfg.ny, cfg.cpml_range)
    funcs.update_ez_cpml_y[bpg_pmly, tpb](dPSI_EZ_YLO, dPSI_EZ_YHI, dEZ, dHX, dHY, dCURL_H, dBE, dCE, cfg.delta_x, cfg.delta_t, cfg.eps, cfg.nx, cfg.ny, cfg.cpml_range)

    # Update H-field components
    funcs.update_hx_hy[bpg, tpb](dHX, dHY, dEZ, CURL_E, dDEN_HX, dDEN_HY, cfg.nx, cfg.ny)

    # Update H-field components in PML
    funcs.update_hy_cpml_x[bpg_pmlx, tpb](dPSI_HY_XLO, dPSI_HY_XHI, dHY, dEZ, dBH, dCH, cfg.delta_x, cfg.delta_t, cfg.mu, cfg.nx, cfg.ny, cfg.cpml_range)
    funcs.update_hx_cpml_y[bpg_pmly, tpb](dPSI_HX_YLO, dPSI_HX_YHI, dHX, dEZ, dBH, dCH, cfg.delta_x, cfg.delta_t, cfg.mu, cfg.nx, cfg.ny, cfg.cpml_range)

    # Record Data
    funcs.map_efield_zwave[bpgy, tpby](dEZT, dEZ, n, cfg.cx, cfg.ny)
    funcs.map_hfield_zwave[bpgy, tpby](dHXT, dHX, n, cfg.cx, cfg.ny)

    # Progress feedback
    last40 = np.roll(last40, 1)
    iter_time = time() - cu_time - loop_start_time
    cu_time = time() - loop_start_time
    last40[0] = iter_time
    if (n == feedback_interval).any():
        avg_iter_time = np.average(last40)
        cu_time = time() - loop_start_time
        time_rem = avg_iter_time * ( cfg.nt - n - 1 )
        print('\nStep {} of {} done, {:.1f} % complete'.format(n+1, cfg.nt, n/(cfg.nt-2)*100))
        print('Loop time elapsed:         {} (hr) {} (min) {:.1f} (s)'.format(int(cu_time/3600), int((cu_time - 3600*(cu_time//3600))//60), cu_time - 60*((cu_time - 3600*(cu_time//3600))//60) - 3600*(cu_time//3600)))
        print('Avg. loop period:          {:.2f} (ms)'.format(avg_iter_time*1000))
        print('Estimated time remaining:  {} (hr) {} (min) {:.1f} (s)'.format(int(time_rem/3600), int((time_rem - 3600*(time_rem//3600))//60), time_rem - 60*((time_rem - 3600*(time_rem//3600))//60) - 3600*(time_rem//3600)))
        print('Memory used:              {:6.3f} (GB)'.format(process.memory_info().rss/1024/1024/1024))
        print('MC/sec:                    {}'.format((cfg.nx * cfg.ny) / (1e6 * avg_iter_time)))

# =============================================================================
# Post-simulation processes and analytics
# =============================================================================

looping_time = time() - loop_start_time
SPEED_MCells_Sec = (cfg.nx * cfg.ny * cfg.nt) / (1e6 * looping_time)
lth = int(np.floor(looping_time / 3600))
ltm = int(np.floor((looping_time - lth*3600) / 60))
lts = int(np.ceil(looping_time - lth*3600 - ltm*60))
if lts == 60:
    lts -= 60
    ltm += 1
if ltm < 10:
    ltm = '0'+str(ltm)
if lts < 10:
    lts = '0'+str(lts)
print('FDTD loop time was: {:}:{:}:{:}'.format(lth, ltm, lts))
print('Speed in MCells/sec: {:}'.format(SPEED_MCells_Sec))

# =============================================================================
# Save storage arrays to files
# =============================================================================

print('saving results')
np.save('./{}/ez_zwave'.format(cfg.output_dir), dEZT.copy_to_host(EZT))
np.save('./{}/hx_zwave'.format(cfg.output_dir), dHXT.copy_to_host(HXT))
print('results saved!')
cuda.close()
