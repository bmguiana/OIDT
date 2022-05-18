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

EX = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)     # Electric field at current time step, x-component
EY = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)     # Electric field at current time step, y-component
HZ = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)     # Magnetic field at current time step, z-component

# Assign Materials
if cfg.rough_toggle:
    FG_REG = funcs.gen_fg_mask_sgl(cfg.nx, cfg.ny, cfg.by, cfg.ny_swg, cfg.p1x, cfg.nx-cfg.p2x, cfg.rstd, cfg.racl, cfg.delta_x, mode='gen', correlation=cfg.ctype, upper_path=cfg.up, lower_path=cfg.lp, atol=cfg.tol_acl, stol=cfg.tol_std, mtol=0.01)
else:
    FG_REG = funcs.gen_fg_mask_sgl(cfg.nx, cfg.ny, cfg.by, cfg.ny_swg, cfg.p1x, cfg.nx-cfg.p2x, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth')

EPS_MASK = (cfg.eps * ( cfg.eps_rel_fg * FG_REG + 1 * ~FG_REG )).astype(np.float32)
EPS_EX = EPS_MASK.copy().astype(np.float32)
EPS_EY = EPS_MASK.copy().astype(np.float32)
for j in range(1, cfg.ny):
    for i in range(1, cfg.nx):
        EPS_EX[i, j] = 0.5 * (EPS_MASK[i, j] + EPS_MASK[i-1, j])
        EPS_EY[i, j] = 0.5 * (EPS_MASK[i, j] + EPS_MASK[i, j-1])

CBX = cfg.delta_t / EPS_EX
CBY = cfg.delta_t / EPS_EY
DB = (cfg.delta_t / cfg.mu).astype(np.float32)


# Define data storage arrays
HZfr = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)
HZfi = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)
EYfr = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)
EYfi = np.zeros([cfg.nx, cfg.ny], dtype=np.float32)

# =============================================================================
# Begin CPML
# =============================================================================

# Computational domain limits
offset = 2
ll = 0+offset
lb = 0+offset
lr = cfg.nx - ll
lt = cfg.ny - lb

# Constants
m_grade = 4                                     # Grading maximum exponent for sigma and kappa
m_alpha = 1                                     # Grading maximum exponent for alpha
cpml_sigma_optimal = 0.8 * ( m_grade + 1 ) / ( cfg.eta * cfg.delta_x )  # Optimal sigma value
cpml_sigma_max = 1.2 * cpml_sigma_optimal       # Maximum sigma value
cpml_alpha_max = 0.05                           # Maximum alpha value
cpml_kappa_max = 5                          # Maximum kappa value if CPML is on

# # Define arrays
PSI_EX_YLO = np.zeros([cfg.nx, cfg.num_cpml], dtype=np.float32)     # Ex correction field, low-side y-boundary
PSI_EX_YHI = np.zeros([cfg.nx, cfg.num_cpml], dtype=np.float32)     # Ex correction field, high-side y-boundary
PSI_EY_XLO = np.zeros([cfg.num_cpml, cfg.ny], dtype=np.float32)     # Ey correction field, low-side x-boundary
PSI_EY_XHI = np.zeros([cfg.num_cpml, cfg.ny], dtype=np.float32)     # Ey correction field, high-side x-boundary

PSI_HZ_XLO = np.zeros([cfg.num_cpml, cfg.ny], dtype=np.float32)     # Hz correction field, low-side x-boundary
PSI_HZ_XHI = np.zeros([cfg.num_cpml, cfg.ny], dtype=np.float32)     # Hz correction field, high-side x-boundary
PSI_HZ_YLO = np.zeros([cfg.nx, cfg.num_cpml], dtype=np.float32)     # Hz correction field, low-side y-boundary
PSI_HZ_YHI = np.zeros([cfg.nx, cfg.num_cpml], dtype=np.float32)     # Hz correction field, high-side y-boundary

SE = np.zeros(cfg.num_cpml, dtype=np.float32)
KE = np.zeros(cfg.num_cpml, dtype=np.float32)
AE = np.zeros(cfg.num_cpml, dtype=np.float32)
BE = np.zeros(cfg.num_cpml, dtype=np.float32)
CE = np.zeros(cfg.num_cpml, dtype=np.float32)

SH = np.zeros(cfg.num_cpml, dtype=np.float32)
KH = np.zeros(cfg.num_cpml, dtype=np.float32)
AH = np.zeros(cfg.num_cpml, dtype=np.float32)
BH = np.zeros(cfg.num_cpml, dtype=np.float32)
CH = np.zeros(cfg.num_cpml, dtype=np.float32)

KXE = np.ones(cfg.nx, dtype=np.float32) * cfg.delta_x
KYE = np.ones(cfg.ny, dtype=np.float32) * cfg.delta_x
KXH = np.ones(cfg.nx, dtype=np.float32) * cfg.delta_x
KYH = np.ones(cfg.ny, dtype=np.float32) * cfg.delta_x

for d in range(cfg.num_cpml):
    SE[d] = cpml_sigma_max * ( (d+0.5)/cfg.num_cpml )**m_grade
    KE[d] = 1 + (cpml_kappa_max - 1)*( (d+0.5)/cfg.num_cpml )**m_grade
    AE[d] = cpml_alpha_max * ((cfg.num_cpml - (d+0.5))/cfg.num_cpml)**m_alpha
    BE[d] = np.exp( -1 * ( SE[d]/KE[d] + AE[d] ) * ( cfg.delta_t / cfg.eps ) )
    CE[d] = SE[d] / (SE[d] * KE[d] + KE[d]**2 * AE[d]) * (BE[d] - 1)

    SH[d] = cpml_sigma_max * ( d/cfg.num_cpml )**m_grade
    KH[d] = 1 + (cpml_kappa_max - 1)*( d/cfg.num_cpml )**m_grade
    AH[d] = cpml_alpha_max * ((cfg.num_cpml - d)/cfg.num_cpml)**m_alpha
    BH[d] = np.exp( -1 * ( SH[d]/KH[d] + AH[d] ) * ( cfg.delta_t / cfg.eps ) )
    CH[d] = SH[d] / (SH[d] * KH[d] + KH[d]**2 * AH[d]) * (BH[d] - 1)

for d in range(cfg.num_cpml):
    KXE[ll+cfg.num_cpml-(d+1)] = KE[d] * cfg.delta_x  # dx
    KYE[lb+cfg.num_cpml-(d+1)] = KE[d] * cfg.delta_x  # dy
    KXH[ll+cfg.num_cpml-d] = KH[d] * cfg.delta_x      # dx
    KYH[lb+cfg.num_cpml-d] = KH[d] * cfg.delta_x      # dy
    KXE[lr-(d+1)] = KE[-(d+1)] * cfg.delta_x          # dx
    KYE[lt-(d+1)] = KE[-(d+1)] * cfg.delta_x          # dy
    KXH[lr-(d+1)] = KH[-(d+1)] * cfg.delta_x          # dx
    KYH[lt-(d+1)] = KH[-(d+1)] * cfg.delta_x          # dy

# =============================================================================
# GPU Processing Setup
# =============================================================================

# Declare thread sizes
tpbx = 1
tpby = 128

# Declare block sizes
bpgx = int(np.ceil(cfg.nx / tpbx))
bpgy = int(np.ceil(cfg.ny / tpby))
bpgx_pml = int(np.ceil(cfg.num_cpml / tpbx))
bpgy_pml = int(np.ceil(cfg.num_cpml / tpby))

# Combine into tuples
tpb = (tpbx, tpby)
bpg_xy = (bpgx, bpgy)
bpg_xp = (bpgx, bpgy_pml)
bpg_py = (bpgx_pml, bpgy)

# GPU process feedback
cells_on_device = 5*(cfg.nx*cfg.ny) + 4*(cfg.num_cpml*cfg.nx + cfg.num_cpml*cfg.ny + cfg.num_cpml) + 2*(cfg.nx + cfg.ny) + 1*(cfg.nt)
device_req_mem = 4*cells_on_device / 1024 / 1024
print('Transferring {:.0f} Mcells onto GPU, requiring {:} MB'.format(cells_on_device/1e6, device_req_mem))

# Device arrays of size (nx*ny)
dEX = cuda.to_device(EX)
dEY = cuda.to_device(EY)
dHZ = cuda.to_device(HZ)
dCBX = cuda.to_device(CBX)
dCBY = cuda.to_device(CBY)
dHZfr = cuda.to_device(HZfr)
dHZfi = cuda.to_device(HZfi)
dEYfr = cuda.to_device(EYfr)
dEYfi = cuda.to_device(EYfi)

# Device arrays of size (num_cpml*nx)
dPSI_EX_YLO = cuda.to_device(PSI_EX_YLO)
dPSI_EX_YHI = cuda.to_device(PSI_EX_YHI)
dPSI_HZ_YLO = cuda.to_device(PSI_HZ_YLO)
dPSI_HZ_YHI = cuda.to_device(PSI_HZ_YHI)

# Device arrays of size (num_cpml*ny)
dPSI_EY_XLO = cuda.to_device(PSI_EY_XLO)
dPSI_EY_XHI = cuda.to_device(PSI_EY_XHI)
dPSI_HZ_XLO = cuda.to_device(PSI_HZ_XLO)
dPSI_HZ_XHI = cuda.to_device(PSI_HZ_XHI)

# Device arrays of size (num_cpml)
dBE = cuda.to_device(BE)
dBH = cuda.to_device(BH)
dCE = cuda.to_device(CE)
dCH = cuda.to_device(CH)

# Device arrays of size (nx)
dKXE = cuda.to_device(KXE)
dKXH = cuda.to_device(KXH)

# Device arrays of size (ny)
dKYE = cuda.to_device(KYE)
dKYH = cuda.to_device(KYH)

# Device arrays of size (nt)
dJ_SRC = cuda.to_device(cfg.J_SRC)

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
    # Update H-field component
    funcs.update_hz[bpg_xy, tpb](dEX, dEY, dHZ, dKXH, dKYH, DB, ll, lb, lr, lt)

    # Update H-field component in PML
    funcs.update_pml_hz_yinc[bpg_xp, tpb](dPSI_HZ_YLO, dPSI_HZ_YHI, dEX, dHZ, dBH, dCH, DB, cfg.delta_x, cfg.num_cpml, ll, lb, lr, lt)
    funcs.update_pml_hz_xinc[bpg_py, tpb](dPSI_HZ_XLO, dPSI_HZ_XHI, dEY, dHZ, dBH, dCH, DB, cfg.delta_x, cfg.num_cpml, ll, lb, lr, lt)

    # Update source conditions
    funcs.update_source[bpg_xy, tpb](dHZ, dJ_SRC, cfg.sx, cfg.by+2, cfg.by+cfg.ny_swg-2, n)

    # Update E-field components
    funcs.update_ex[bpg_xy, tpb](dEX, dHZ, dKYE, dCBX, ll, lb, lr, lt)
    funcs.update_ey[bpg_xy, tpb](dEY, dHZ, dKXE, dCBY, ll, lb, lr, lt)

    # Update E-field components in PML
    funcs.update_pml_ex_yinc[bpg_xp, tpb](dPSI_EX_YLO, dPSI_EX_YHI, dEX, dHZ, dBE, dCE, dCBX, cfg.delta_x, cfg.num_cpml, ll, lb, lr, lt)
    funcs.update_pml_ey_xinc[bpg_py, tpb](dPSI_EY_XLO, dPSI_EY_XHI, dEY, dHZ, dBE, dCE, dCBY, cfg.delta_x, cfg.num_cpml, ll, lb, lr, lt)

    # Record Data
    funcs.simul_fft[bpg_xy, tpb](dHZfr, dHZfi, dHZ, cfg.f0, n+0.0, cfg.delta_t, cfg.nx, cfg.ny)
    funcs.simul_fft[bpg_xy, tpb](dEYfr, dEYfi, dEY, cfg.f0, n+0.5, cfg.delta_t, cfg.nx, cfg.ny)

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
np.save('./'+cfg.output_dir+'/'+cfg.hz_fft_name+cfg.roughness_profile, dHZfr.copy_to_host() - 1j*dHZfi.copy_to_host())
np.save('./'+cfg.output_dir+'/'+cfg.ey_fft_name+cfg.roughness_profile, dEYfr.copy_to_host() - 1j*dEYfi.copy_to_host())
print('results saved!')
cuda.close()
