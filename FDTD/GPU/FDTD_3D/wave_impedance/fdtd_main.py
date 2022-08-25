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

print('The simulation size will be: {:} x {:} x {:} cubic cells over {:} time steps\n\n\n'.format(cfg.nx, cfg.ny, cfg.nz, cfg.nt))
aux.mkdir(cfg.output_dir)
cuda.select_device(0)

# =============================================================================
# Main array definition
# =============================================================================

EX = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=np.float32)     # Electric field at current time step, x-component
EY = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=np.float32)     # Electric field at current time step, y-component
EZ = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=np.float32)     # Electric field at current time step, z-component
HX = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=np.float32)     # Magnetic field at current time step, x-component
HY = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=np.float32)     # Magnetic field at current time step, y-component
HZ = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=np.float32)     # Magnetic field at current time step, z-component

# Assign Materials
if cfg.rough_toggle:
    FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.nz, cfg.bx, cfg.by, cfg.nx_swg, cfg.ny_swg, cfg.p1z, cfg.nz-cfg.p2z, cfg.rstd, cfg.racl, cfg.delta_x, upper_name=cfg.up, lower_name=cfg.lp, atol=cfg.tol_acl, stol=cfg.tol_std, mtol=0.01)
else:
    FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.nz, cfg.bx, cfg.by, cfg.nx_swg, cfg.ny_swg, cfg.p1z, cfg.nz-cfg.p2z, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth')
EPS_MASK = cfg.eps * ( cfg.eps_rel_fg * FG_REG + 1 * ~FG_REG )
EPS_EX = EPS_MASK.copy().astype(np.float32)
EPS_EY = EPS_MASK.copy().astype(np.float32)
EPS_EZ = EPS_MASK.copy().astype(np.float32)
funcs.average_materials(EPS_MASK, EPS_EX, EPS_EY, EPS_EZ, cfg.nx, cfg.ny, cfg.nz)
CBX = cfg.delta_t / EPS_EX
CBY = cfg.delta_t / EPS_EY
CBZ = cfg.delta_t / EPS_EZ
DB = cfg.delta_t / cfg.mu

# Define data storage arrays
EY_ZW = np.zeros([cfg.nt, cfg.nx], dtype=np.float32)
EZ_ZW = np.zeros([cfg.nt, cfg.nx], dtype=np.float32)
HY_ZW = np.zeros([cfg.nt, cfg.nx], dtype=np.float32)
HZ_ZW = np.zeros([cfg.nt, cfg.nx], dtype=np.float32)

# =============================================================================
# Begin CPML
# =============================================================================

# Computational Domain Limits
offset = 2
ib = 0+offset
jb = 0+offset
kb = 0+offset
ie = cfg.nx - ib
je = cfg.ny - jb
ke = cfg.nz - kb

# Constants
m_grade = 4                                                              # Grading maximum exponent for sigma and kappa
m_alpha = 1                                                              # Grading maximum exponent for alpha
cpml_sigma_optimal = 0.8 * ( m_grade + 1 ) / ( cfg.eta * cfg.delta_x )   # Optimal sigma value
cpml_sigma_max = 1.2 * cpml_sigma_optimal                                # Maximum sigma value
cpml_alpha_max = 0.05                                                    # Maximum alpha value
cpml_kappa_max = 5                                                       # Maximum kappa value

# Define arrays
PSI_EX_YLO = np.zeros([cfg.nx, cfg.num_cpml, cfg.nz], dtype=np.float32)  # Ex correction field, low-side y-boundary
PSI_EX_YHI = np.zeros([cfg.nx, cfg.num_cpml, cfg.nz], dtype=np.float32)  # Ex correction field, high-side y-boundary
PSI_EX_ZLO = np.zeros([cfg.nx, cfg.ny, cfg.num_cpml], dtype=np.float32)  # Ex correction field, low-side z-boundary
PSI_EX_ZHI = np.zeros([cfg.nx, cfg.ny, cfg.num_cpml], dtype=np.float32)  # Ex correction field, high-side z-boundary

PSI_EY_XLO = np.zeros([cfg.num_cpml, cfg.ny, cfg.nz], dtype=np.float32)  # Ey correction field, low-side x-boundary
PSI_EY_XHI = np.zeros([cfg.num_cpml, cfg.ny, cfg.nz], dtype=np.float32)  # Ey correction field, high-side x-boundary
PSI_EY_ZLO = np.zeros([cfg.nx, cfg.ny, cfg.num_cpml], dtype=np.float32)  # Ey correction field, low-side z-boundary
PSI_EY_ZHI = np.zeros([cfg.nx, cfg.ny, cfg.num_cpml], dtype=np.float32)  # Ey correction field, high-side z-boundary

PSI_EZ_XLO = np.zeros([cfg.num_cpml, cfg.ny, cfg.nz], dtype=np.float32)  # Ez correction field, low-side x-boundary
PSI_EZ_XHI = np.zeros([cfg.num_cpml, cfg.ny, cfg.nz], dtype=np.float32)  # Ez correction field, high-side x-boundary
PSI_EZ_YLO = np.zeros([cfg.nx, cfg.num_cpml, cfg.nz], dtype=np.float32)  # Ez correction field, low-side y-boundary
PSI_EZ_YHI = np.zeros([cfg.nx, cfg.num_cpml, cfg.nz], dtype=np.float32)  # Ez correction field, high-side y-boundary

PSI_HX_YLO = np.zeros([cfg.nx, cfg.num_cpml, cfg.nz], dtype=np.float32)  # Hx correction field, low-side y-boundary
PSI_HX_YHI = np.zeros([cfg.nx, cfg.num_cpml, cfg.nz], dtype=np.float32)  # Hx correction field, high-side y-boundary
PSI_HX_ZLO = np.zeros([cfg.nx, cfg.ny, cfg.num_cpml], dtype=np.float32)  # Hx correction field, low-side z-boundary
PSI_HX_ZHI = np.zeros([cfg.nx, cfg.ny, cfg.num_cpml], dtype=np.float32)  # Hx correction field, high-side z-boundary

PSI_HY_XLO = np.zeros([cfg.num_cpml, cfg.ny, cfg.nz], dtype=np.float32)  # Hy correction field, low-side x-boundary
PSI_HY_XHI = np.zeros([cfg.num_cpml, cfg.ny, cfg.nz], dtype=np.float32)  # Hy correction field, high-side x-boundary
PSI_HY_ZLO = np.zeros([cfg.nx, cfg.ny, cfg.num_cpml], dtype=np.float32)  # Hy correction field, low-side z-boundary
PSI_HY_ZHI = np.zeros([cfg.nx, cfg.ny, cfg.num_cpml], dtype=np.float32)  # Hy correction field, high-side z-boundary

PSI_HZ_XLO = np.zeros([cfg.num_cpml, cfg.ny, cfg.nz], dtype=np.float32)  # Hz correction field, low-side x-boundary
PSI_HZ_XHI = np.zeros([cfg.num_cpml, cfg.ny, cfg.nz], dtype=np.float32)  # Hz correction field, high-side x-boundary
PSI_HZ_YLO = np.zeros([cfg.nx, cfg.num_cpml, cfg.nz], dtype=np.float32)  # Hz correction field, low-side y-boundary
PSI_HZ_YHI = np.zeros([cfg.nx, cfg.num_cpml, cfg.nz], dtype=np.float32)  # Hz correction field, high-side y-boundary

SE = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Electric field sigma grading
KE = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Electric field kappa grading
AE = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Electric field alpha grading
BE = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Electric field auxiliary variable b
CE = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Electric field auxiliary variable c

SH = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Magnetic field sigma grading
KH = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Magnetic field kappa grading
AH = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Magnetic field alpha grading
BH = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Magnetic field auxiliary variable b
CH = np.zeros(cfg.num_cpml, dtype=np.float32)                            # Magnetic field auxiliary variable c

KXE = np.ones(cfg.nx, dtype=np.float32) * cfg.delta_x                    # Electric field x-direction kappa division
KYE = np.ones(cfg.ny, dtype=np.float32) * cfg.delta_x                    # Electric field y-direction kappa division
KZE = np.ones(cfg.nz, dtype=np.float32) * cfg.delta_x                    # Electric field z-direction kappa division
KXH = np.ones(cfg.nx, dtype=np.float32) * cfg.delta_x                    # Magnetic field x-direction kappa division
KYH = np.ones(cfg.ny, dtype=np.float32) * cfg.delta_x                    # Magnetic field y-direction kappa division
KZH = np.ones(cfg.nz, dtype=np.float32) * cfg.delta_x                    # Magnetic field z-direction kappa division

# Assign array values
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
    KXE[ib+cfg.num_cpml-(d+1)] = KE[d] * cfg.delta_x  # dx
    KYE[jb+cfg.num_cpml-(d+1)] = KE[d] * cfg.delta_x  # dy
    KZE[kb+cfg.num_cpml-(d+1)] = KE[d] * cfg.delta_x  # dz
    KXH[ib+cfg.num_cpml-d] = KH[d] * cfg.delta_x      # dx
    KYH[jb+cfg.num_cpml-d] = KH[d] * cfg.delta_x      # dy
    KZH[kb+cfg.num_cpml-d] = KH[d] * cfg.delta_x      # dz
    KXE[ie-(d+1)] = KE[-(d+1)] * cfg.delta_x          # dx
    KYE[je-(d+1)] = KE[-(d+1)] * cfg.delta_x          # dy
    KZE[ke-(d+1)] = KE[-(d+1)] * cfg.delta_x          # dz
    KXH[ie-(d+1)] = KH[-(d+1)] * cfg.delta_x          # dx
    KYH[je-(d+1)] = KH[-(d+1)] * cfg.delta_x          # dy
    KZH[ke-(d+1)] = KH[-(d+1)] * cfg.delta_x          # dz

# =============================================================================
# GPU Processing Setup
# =============================================================================

# Declare thread sizes
tpbx = 1
tpby = 4
tpbz = 64

# Declare block sizes
bpgx = int(np.ceil(cfg.nx / tpbx))
bpgy = int(np.ceil(cfg.ny / tpby))
bpgz = int(np.ceil(cfg.nz / tpbz))
bpgx_cpml = int(np.ceil(cfg.num_cpml / tpbx))
bpgy_cpml = int(np.ceil(cfg.num_cpml / tpby))
bpgz_cpml = int(np.ceil(cfg.num_cpml / tpbz))

# Combine into tuples
tpb = (tpbx, tpby, tpbz)
bpg_xyz = (bpgx, bpgy, bpgz)
bpg_xyp = (bpgx, bpgy, bpgz_cpml)
bpg_xpz = (bpgx, bpgy_cpml, bpgz)
bpg_pyz = (bpgx_cpml, bpgy, bpgz)
tpb_fft = (1, 128)
bpg_fft = (int(np.ceil(cfg.nx / tpb_fft[0])), int(np.ceil(cfg.ny / tpb_fft[1])))

# GPU process feedback
cells_on_device = 9*(cfg.nx*cfg.ny*cfg.nz) + 8*(cfg.num_cpml*cfg.ny*cfg.nz + cfg.nx*cfg.num_cpml*cfg.nz + cfg.nx*cfg.ny*cfg.num_cpml) + 4*(cfg.num_cpml) + 2*(cfg.nx + cfg.ny + cfg.nz + cfg.nt) + 4*(cfg.nx*cfg.nt)
device_req_mem = 4*cells_on_device / 1024 / 1024
print('Transferring {:.0f} Mcells onto GPU, requiring {:} MB'.format(cells_on_device/1e6, device_req_mem))

# Device arrays of size (nx*ny*nz)
dEX = cuda.to_device(EX)
dEY = cuda.to_device(EY)
dEZ = cuda.to_device(EZ)
dHX = cuda.to_device(HX)
dHY = cuda.to_device(HY)
dHZ = cuda.to_device(HZ)
dCBX = cuda.to_device(CBX)
dCBY = cuda.to_device(CBY)
dCBZ = cuda.to_device(CBZ)

# Device arrays of size (num_cpml*ny*nz)
dPSI_EY_XLO = cuda.to_device(PSI_EY_XLO)
dPSI_EY_XHI = cuda.to_device(PSI_EY_XHI)
dPSI_EZ_XLO = cuda.to_device(PSI_EZ_XLO)
dPSI_EZ_XHI = cuda.to_device(PSI_EZ_XHI)
dPSI_HY_XLO = cuda.to_device(PSI_HY_XLO)
dPSI_HY_XHI = cuda.to_device(PSI_HY_XHI)
dPSI_HZ_XLO = cuda.to_device(PSI_HZ_XLO)
dPSI_HZ_XHI = cuda.to_device(PSI_HZ_XHI)

# Device arrays of size (nx*num_cpml*nz)
dPSI_EX_YLO = cuda.to_device(PSI_EX_YLO)
dPSI_EX_YHI = cuda.to_device(PSI_EX_YHI)
dPSI_EZ_YLO = cuda.to_device(PSI_EZ_YLO)
dPSI_EZ_YHI = cuda.to_device(PSI_EZ_YHI)
dPSI_HX_YLO = cuda.to_device(PSI_HX_YLO)
dPSI_HX_YHI = cuda.to_device(PSI_HX_YHI)
dPSI_HZ_YLO = cuda.to_device(PSI_HZ_YLO)
dPSI_HZ_YHI = cuda.to_device(PSI_HZ_YHI)

# Device arrays of size (nx*ny*num_cpml)
dPSI_EX_ZLO = cuda.to_device(PSI_EX_ZLO)
dPSI_EX_ZHI = cuda.to_device(PSI_EX_ZHI)
dPSI_EY_ZLO = cuda.to_device(PSI_EY_ZLO)
dPSI_EY_ZHI = cuda.to_device(PSI_EY_ZHI)
dPSI_HX_ZLO = cuda.to_device(PSI_HX_ZLO)
dPSI_HX_ZHI = cuda.to_device(PSI_HX_ZHI)
dPSI_HY_ZLO = cuda.to_device(PSI_HY_ZLO)
dPSI_HY_ZHI = cuda.to_device(PSI_HY_ZHI)

# Device arrays of size (num_cpml)
dBE = cuda.to_device(BE)
dCE = cuda.to_device(CE)
dBH = cuda.to_device(BH)
dCH = cuda.to_device(CH)

# Device arrays of size (nx)
dKXE = cuda.to_device(KXE)
dKXH = cuda.to_device(KXH)

# Device arrays of size (ny)
dKYE = cuda.to_device(KYE)
dKYH = cuda.to_device(KYH)

# Device arrays of size (nz)
dKZE = cuda.to_device(KZE)
dKZH = cuda.to_device(KZH)

# Device arrays of size (nt)
dJ_SRC = cuda.to_device(cfg.J_SRC)

# Device arrays of size(nt*nx)
dEY_ZW = cuda.to_device(EY_ZW)
dEZ_ZW = cuda.to_device(EZ_ZW)
dHY_ZW = cuda.to_device(HY_ZW)
dHZ_ZW = cuda.to_device(HZ_ZW)

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
    # Update E-field components
    funcs.update_ex[bpg_xyz, tpb](dEX, dHY, dHZ, dKYE, dKZE, dCBX, ib, jb, kb, ie, je, ke)
    funcs.update_ey[bpg_xyz, tpb](dEY, dHX, dHZ, dKXE, dKZE, dCBY, ib, jb, kb, ie, je, ke)
    funcs.update_ez[bpg_xyz, tpb](dEZ, dHX, dHY, dKXE, dKYE, dCBZ, ib, jb, kb, ie, je, ke)

    # Update source conditions
    if cfg.source_condition == 'TE':
        funcs.update_source[bpg_xyz, tpb](dEY, dJ_SRC, cfg.bx+1, cfg.by+1, cfg.sz, cfg.bx+cfg.nx_swg-1, cfg.by+cfg.ny_swg-1, cfg.sz+2, n)
    elif cfg.source_condition == 'TM':
        funcs.update_source[bpg_xyz, tpb](dHY, dJ_SRC, cfg.bx+2, cfg.by+2, cfg.sz, cfg.bx+cfg.nx_swg-2, cfg.by+cfg.ny_swg-2, cfg.sz+2, n)

    # Update E-field components in PML
    funcs.update_pml_ex_yinc[bpg_xpz, tpb](dPSI_EX_YLO, dPSI_EX_YHI, dEX, dHZ, dBE, dCE, dCBX, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_ex_zinc[bpg_xyp, tpb](dPSI_EX_ZLO, dPSI_EX_ZHI, dEX, dHY, dBE, dCE, dCBX, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_ey_xinc[bpg_pyz, tpb](dPSI_EY_XLO, dPSI_EY_XHI, dEY, dHZ, dBE, dCE, dCBY, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_ey_zinc[bpg_xyp, tpb](dPSI_EY_ZLO, dPSI_EY_ZHI, dEY, dHX, dBE, dCE, dCBY, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_ez_xinc[bpg_pyz, tpb](dPSI_EZ_XLO, dPSI_EZ_XHI, dEZ, dHY, dBE, dCE, dCBZ, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_ez_yinc[bpg_xpz, tpb](dPSI_EZ_YLO, dPSI_EZ_YHI, dEZ, dHX, dBE, dCE, dCBZ, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)

    # Update H-field components
    funcs.update_hx[bpg_xyz, tpb](dEY, dEZ, dHX, dKYH, dKZH, DB, ib, jb, kb, ie, je, ke)
    funcs.update_hy[bpg_xyz, tpb](dEX, dEZ, dHY, dKXH, dKZH, DB, ib, jb, kb, ie, je, ke)
    funcs.update_hz[bpg_xyz, tpb](dEX, dEY, dHZ, dKXH, dKYH, DB, ib, jb, kb, ie, je, ke)

    # Update H-field components in PML
    funcs.update_pml_hx_yinc[bpg_xpz, tpb](dPSI_HX_YLO, dPSI_HX_YHI, dEZ, dHX, dBH, dCH, DB, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_hx_zinc[bpg_xyp, tpb](dPSI_HX_ZLO, dPSI_HX_ZHI, dEY, dHX, dBH, dCH, DB, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_hy_xinc[bpg_pyz, tpb](dPSI_HY_XLO, dPSI_HY_XHI, dEZ, dHY, dBH, dCH, DB, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_hy_zinc[bpg_xyp, tpb](dPSI_HY_ZLO, dPSI_HY_ZHI, dEX, dHY, dBH, dCH, DB, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_hz_xinc[bpg_pyz, tpb](dPSI_HZ_XLO, dPSI_HZ_XHI, dEY, dHZ, dBH, dCH, DB, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)
    funcs.update_pml_hz_yinc[bpg_xpz, tpb](dPSI_HZ_YLO, dPSI_HZ_YHI, dEX, dHZ, dBH, dCH, DB, cfg.delta_x, cfg.num_cpml, ib, jb, kb, ie, je, ke)

    # Record Data
    funcs.map_efield_zwave[bpgx, tpbx](dEY_ZW, dEZ_ZW, dEY, dEZ, n, cfg.cy, cfg.cz, ie)
    funcs.map_hfield_zwave[bpgx, tpbx](dHY_ZW, dHZ_ZW, dHY, dHZ, n, cfg.cy, cfg.cz, ie)

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
        print('MC/sec:                    {}'.format((cfg.nx * cfg.ny * cfg.nz) / (1e6 * avg_iter_time)))

# =============================================================================
# Post-simulation processes and analytics
# =============================================================================

looping_time = time() - loop_start_time
SPEED_MCells_Sec = (cfg.nx * cfg.ny * cfg.nz * cfg.nt) / (1e6 * looping_time)
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
if cfg.source_condition == 'TE':
    np.save('./{}/{}'.format(cfg.output_dir, cfg.ey_zwave_name), dEY_ZW.copy_to_host(EY_ZW))
    np.save('./{}/{}'.format(cfg.output_dir, cfg.hz_zwave_name), dHZ_ZW.copy_to_host(HZ_ZW))
elif cfg.source_condition == 'TM':
    np.save('./{}/{}'.format(cfg.output_dir, cfg.ez_zwave_name), dEZ_ZW.copy_to_host(EZ_ZW))
    np.save('./{}/{}'.format(cfg.output_dir, cfg.hy_zwave_name), dHY_ZW.copy_to_host(HY_ZW))
print('results saved!')
cuda.close()
