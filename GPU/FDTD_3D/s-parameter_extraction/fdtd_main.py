"""
Author: B. Guiana

Description:


Acknowledgement: This project was completed as part of research conducted with
                 my major professor and advisor, Prof. Ata Zadehgol, at the
                 Applied and Computational Electromagnetics Signal and Power
                 Integrity (ACEM-SPI) Lab while working toward the Ph.D. in
                 Electrical Engineering at the University of Idaho, Moscow,
                 Idaho, USA. This project was funded, in part, by the National
                 Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import os
import psutil
from time import time
from numba import cuda
import sys
import pickle

import fdtd_auto_setup as cfg
import fdtd_funcs as funcs
import aux_funcs as aux

args = sys.argv
try:
    uf = open('user_dict_{}.pkl'.format(args[1]), 'rb')
except:
    print('loading default configuration')
    import user_config as user_module
    user_dict = aux.dict_from_module(user_module)
    dict_file = open('user_dict.pkl', 'wb')
    pickle.dump(user_dict, dict_file)
    dict_file.close()
    uf = open('user_dict.pkl', 'rb')

usr = pickle.load(uf)

print('The simulation size will be: {:} x {:} x {:} cubic cells over {:} time steps\n\n\n'.format(cfg.nx, cfg.ny, cfg.nz, cfg.nt))
aux.mkdir(usr['output_dir'])
cuda.select_device(1)

# =============================================================================
# Main array definition
# =============================================================================

EX = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=usr['precision'])     # Electric field at current time step, x-component
EY = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=usr['precision'])     # Electric field at current time step, y-component
EZ = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=usr['precision'])     # Electric field at current time step, z-component
HX = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=usr['precision'])     # Magnetic field at current time step, x-component
HY = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=usr['precision'])     # Magnetic field at current time step, y-component
HZ = np.zeros([cfg.nx, cfg.ny, cfg.nz], dtype=usr['precision'])     # Magnetic field at current time step, z-component

# Assign Materials
if usr['sim_type'] == 's-param':
    # Get sim input
    args = sys.argv
    if len(args) == 1:  # if no inputs given, stop simulation.
        raise Exception('No S-parameter sim type selected. Type 1, 2, 3, or 4 after the python launch command')
    else:
        # Get sim info from inputs
        line = int(args[1])  # Which line is being simulated?
        direction = args[2]  # Which direction does the wave travel? (forward or backward)
        kind = args[3]  # Is this incident or reflected?
        source_shift = line * cfg.pp
    if usr['rough_toggle']:
        if kind == 'i':
            # Generate smooth mask for all incident sims
            FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.nz, cfg.bx, cfg.by, cfg.nx_swg, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1z, cfg.nz-cfg.p2z, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth', out_dir=usr['prof_dir'])
        elif line == 0 and direction == 'f' and kind == 'r':
            # Make new roughness profiles during first line, first reflected sim
            FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.nz, cfg.bx, cfg.by, cfg.nx_swg, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1z, cfg.nz-cfg.p2z, cfg.rstd, cfg.racl, cfg.delta_x, out_dir=usr['prof_dir'], upper_name=cfg.up, lower_name=cfg.lp, atol=cfg.tol_acl, stol=cfg.tol_std, mtol=0.01)
        else:
            # Load the roughness mask for all other sim types. This conditional hierarchy should make these only during reflected sims
            FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.nz, cfg.bx, cfg.by, cfg.nx_swg, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1z, cfg.nz-cfg.p2z, cfg.rstd, cfg.racl, cfg.delta_x, mode='load', out_dir=usr['prof_dir'], upper_name=cfg.up+'.npy', lower_name=cfg.lp+'.npy')
    else:
        # Generate only smooth mask if no roughness
        FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.nz, cfg.bx, cfg.by, cfg.nx_swg, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1z, cfg.nz-cfg.p2z, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth', out_dir=usr['prof_dir'])
else:
    FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.nz, cfg.bx, cfg.by, cfg.nx_swg, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1z, cfg.nz-cfg.p2z, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth', out_dir=usr['prof_dir'])

EPS_MASK = cfg.eps * ( cfg.eps_rel_fg * FG_REG + 1 * ~FG_REG )
EPS_EX = EPS_MASK.copy().astype(usr['precision'])
EPS_EY = EPS_MASK.copy().astype(usr['precision'])
EPS_EZ = EPS_MASK.copy().astype(usr['precision'])
funcs.average_materials(EPS_MASK, EPS_EX, EPS_EY, EPS_EZ, cfg.nx, cfg.ny, cfg.nz)
CBX = cfg.delta_t / EPS_EX
CBY = cfg.delta_t / EPS_EY
CBZ = cfg.delta_t / EPS_EZ
DB = cfg.delta_t / cfg.mu

# Define data storage arrays
# EX1r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# EY1r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# EZ1r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HX1r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HY1r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HZ1r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])

# EX1i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# EY1i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# EZ1i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HX1i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HY1i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HZ1i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])

# EX2r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# EY2r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# EZ2r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HX2r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HY2r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HZ2r = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])

# EX2i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# EY2i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# EZ2i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HX2i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HY2i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
# HZ2i = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])

# EXZ = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
# EYZ = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
# EZZ = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
# HXZ = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
# HYZ = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
# HZZ = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])

EX1 = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
EY1 = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
HX1 = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
HY1 = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])

EX2 = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
EY2 = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
HX2 = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])
HY2 = np.zeros([cfg.nt, cfg.nx, cfg.ny], dtype=usr['precision'])


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
PSI_EX_YLO = np.zeros([cfg.nx, usr['num_cpml'], cfg.nz], dtype=usr['precision'])  # Ex correction field, low-side y-boundary
PSI_EX_YHI = np.zeros([cfg.nx, usr['num_cpml'], cfg.nz], dtype=usr['precision'])  # Ex correction field, high-side y-boundary
PSI_EX_ZLO = np.zeros([cfg.nx, cfg.ny, usr['num_cpml']], dtype=usr['precision'])  # Ex correction field, low-side z-boundary
PSI_EX_ZHI = np.zeros([cfg.nx, cfg.ny, usr['num_cpml']], dtype=usr['precision'])  # Ex correction field, high-side z-boundary

PSI_EY_XLO = np.zeros([usr['num_cpml'], cfg.ny, cfg.nz], dtype=usr['precision'])  # Ey correction field, low-side x-boundary
PSI_EY_XHI = np.zeros([usr['num_cpml'], cfg.ny, cfg.nz], dtype=usr['precision'])  # Ey correction field, high-side x-boundary
PSI_EY_ZLO = np.zeros([cfg.nx, cfg.ny, usr['num_cpml']], dtype=usr['precision'])  # Ey correction field, low-side z-boundary
PSI_EY_ZHI = np.zeros([cfg.nx, cfg.ny, usr['num_cpml']], dtype=usr['precision'])  # Ey correction field, high-side z-boundary

PSI_EZ_XLO = np.zeros([usr['num_cpml'], cfg.ny, cfg.nz], dtype=usr['precision'])  # Ez correction field, low-side x-boundary
PSI_EZ_XHI = np.zeros([usr['num_cpml'], cfg.ny, cfg.nz], dtype=usr['precision'])  # Ez correction field, high-side x-boundary
PSI_EZ_YLO = np.zeros([cfg.nx, usr['num_cpml'], cfg.nz], dtype=usr['precision'])  # Ez correction field, low-side y-boundary
PSI_EZ_YHI = np.zeros([cfg.nx, usr['num_cpml'], cfg.nz], dtype=usr['precision'])  # Ez correction field, high-side y-boundary

PSI_HX_YLO = np.zeros([cfg.nx, usr['num_cpml'], cfg.nz], dtype=usr['precision'])  # Hx correction field, low-side y-boundary
PSI_HX_YHI = np.zeros([cfg.nx, usr['num_cpml'], cfg.nz], dtype=usr['precision'])  # Hx correction field, high-side y-boundary
PSI_HX_ZLO = np.zeros([cfg.nx, cfg.ny, usr['num_cpml']], dtype=usr['precision'])  # Hx correction field, low-side z-boundary
PSI_HX_ZHI = np.zeros([cfg.nx, cfg.ny, usr['num_cpml']], dtype=usr['precision'])  # Hx correction field, high-side z-boundary

PSI_HY_XLO = np.zeros([usr['num_cpml'], cfg.ny, cfg.nz], dtype=usr['precision'])  # Hy correction field, low-side x-boundary
PSI_HY_XHI = np.zeros([usr['num_cpml'], cfg.ny, cfg.nz], dtype=usr['precision'])  # Hy correction field, high-side x-boundary
PSI_HY_ZLO = np.zeros([cfg.nx, cfg.ny, usr['num_cpml']], dtype=usr['precision'])  # Hy correction field, low-side z-boundary
PSI_HY_ZHI = np.zeros([cfg.nx, cfg.ny, usr['num_cpml']], dtype=usr['precision'])  # Hy correction field, high-side z-boundary

PSI_HZ_XLO = np.zeros([usr['num_cpml'], cfg.ny, cfg.nz], dtype=usr['precision'])  # Hz correction field, low-side x-boundary
PSI_HZ_XHI = np.zeros([usr['num_cpml'], cfg.ny, cfg.nz], dtype=usr['precision'])  # Hz correction field, high-side x-boundary
PSI_HZ_YLO = np.zeros([cfg.nx, usr['num_cpml'], cfg.nz], dtype=usr['precision'])  # Hz correction field, low-side y-boundary
PSI_HZ_YHI = np.zeros([cfg.nx, usr['num_cpml'], cfg.nz], dtype=usr['precision'])  # Hz correction field, high-side y-boundary

SE = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Electric field sigma grading
KE = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Electric field kappa grading
AE = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Electric field alpha grading
BE = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Electric field auxiliary variable b
CE = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Electric field auxiliary variable c

SH = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Magnetic field sigma grading
KH = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Magnetic field kappa grading
AH = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Magnetic field alpha grading
BH = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Magnetic field auxiliary variable b
CH = np.zeros(usr['num_cpml'], dtype=usr['precision'])                            # Magnetic field auxiliary variable c

KXE = np.ones(cfg.nx, dtype=usr['precision']) * cfg.delta_x                    # Electric field x-direction kappa division
KYE = np.ones(cfg.ny, dtype=usr['precision']) * cfg.delta_x                    # Electric field y-direction kappa division
KZE = np.ones(cfg.nz, dtype=usr['precision']) * cfg.delta_x                    # Electric field z-direction kappa division
KXH = np.ones(cfg.nx, dtype=usr['precision']) * cfg.delta_x                    # Magnetic field x-direction kappa division
KYH = np.ones(cfg.ny, dtype=usr['precision']) * cfg.delta_x                    # Magnetic field y-direction kappa division
KZH = np.ones(cfg.nz, dtype=usr['precision']) * cfg.delta_x                    # Magnetic field z-direction kappa division

# Assign array values
for d in range(usr['num_cpml']):
    SE[d] = cpml_sigma_max * ( (d+0.5)/usr['num_cpml'] )**m_grade
    KE[d] = 1 + (cpml_kappa_max - 1)*( (d+0.5)/usr['num_cpml'] )**m_grade
    AE[d] = cpml_alpha_max * ((usr['num_cpml'] - (d+0.5))/usr['num_cpml'])**m_alpha
    BE[d] = np.exp( -1 * ( SE[d]/KE[d] + AE[d] ) * ( cfg.delta_t / cfg.eps ) )
    CE[d] = SE[d] / (SE[d] * KE[d] + KE[d]**2 * AE[d]) * (BE[d] - 1)

    SH[d] = cpml_sigma_max * ( d/usr['num_cpml'] )**m_grade
    KH[d] = 1 + (cpml_kappa_max - 1)*( d/usr['num_cpml'] )**m_grade
    AH[d] = cpml_alpha_max * ((usr['num_cpml'] - d)/usr['num_cpml'])**m_alpha
    BH[d] = np.exp( -1 * ( SH[d]/KH[d] + AH[d] ) * ( cfg.delta_t / cfg.eps ) )
    CH[d] = SH[d] / (SH[d] * KH[d] + KH[d]**2 * AH[d]) * (BH[d] - 1)

for d in range(usr['num_cpml']):
    KXE[ib+usr['num_cpml']-(d+1)] = KE[d] * cfg.delta_x  # dx
    KYE[jb+usr['num_cpml']-(d+1)] = KE[d] * cfg.delta_x  # dy
    KZE[kb+usr['num_cpml']-(d+1)] = KE[d] * cfg.delta_x  # dz
    KXH[ib+usr['num_cpml']-d] = KH[d] * cfg.delta_x      # dx
    KYH[jb+usr['num_cpml']-d] = KH[d] * cfg.delta_x      # dy
    KZH[kb+usr['num_cpml']-d] = KH[d] * cfg.delta_x      # dz
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
bpgx_cpml = int(np.ceil(usr['num_cpml'] / tpbx))
bpgy_cpml = int(np.ceil(usr['num_cpml'] / tpby))
bpgz_cpml = int(np.ceil(usr['num_cpml'] / tpbz))

# Combine into tuples
tpb = (tpbx, tpby, tpbz)
bpg_xyz = (bpgx, bpgy, bpgz)
bpg_xyp = (bpgx, bpgy, bpgz_cpml)
bpg_xpz = (bpgx, bpgy_cpml, bpgz)
bpg_pyz = (bpgx_cpml, bpgy, bpgz)
tpb_fft = (1, 128)
bpg_fft = (int(np.ceil(cfg.nx / tpb_fft[0])), int(np.ceil(cfg.ny / tpb_fft[1])))

# GPU process feedback
cells_on_device = 9*(cfg.nx*cfg.ny*cfg.nz) + 8*(usr['num_cpml']*cfg.ny*cfg.nz + cfg.nx*usr['num_cpml']*cfg.nz + cfg.nx*cfg.ny*usr['num_cpml']) + 4*(usr['num_cpml']) + 2*(cfg.nx + cfg.ny + cfg.nz) + 1*(cfg.nt) + 9*(cfg.nt*cfg.nx*cfg.ny)
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

# Device arrays of size (nt*nx*ny)
# dEYZ = cuda.to_device(EYZ)
# dEZZ = cuda.to_device(EZZ)
# dHYZ = cuda.to_device(HYZ)
# dHZZ = cuda.to_device(HZZ)
dEX1 = cuda.to_device(EX1)
dEY1 = cuda.to_device(EY1)
dHX1 = cuda.to_device(HX1)
dHY1 = cuda.to_device(HY1)
dEX2 = cuda.to_device(EX2)
dEY2 = cuda.to_device(EY2)
dHX2 = cuda.to_device(HX2)
dHY2 = cuda.to_device(HY2)
dDDD = cuda.to_device(np.zeros_like(EX1))

# =============================================================================
# Time stepping loop
# =============================================================================

nfb = int( 100 / usr['feedback_at_n_percent']) + 1
feedback_interval = np.round(np.linspace(0, cfg.nt, num=nfb))
process = psutil.Process(os.getpid())   # Get current process ID
last40 = np.zeros(40)
loop_start_time = time()                # Start loop timing
cu_time = time() - loop_start_time

print('loop started')
for n in range(cfg.nt):
    # Update E-field components
    funcs.update_ex[bpg_xyz, tpb](dEX, dHY, dHZ, dKYE, dKZE, dCBX, ib, jb, kb, ie, je, ke)
    funcs.update_ey[bpg_xyz, tpb](dEY, dHX, dHZ, dKXE, dKZE, dCBY, ib, jb, kb, ie, je, ke)
    funcs.update_ez[bpg_xyz, tpb](dEZ, dHX, dHY, dKXE, dKYE, dCBZ, ib, jb, kb, ie, je, ke)

    # Update source conditions
    if usr['sim_type'] == 's-param':
        if direction == 'f':
            if usr['source_condition'] == 'TE':
                funcs.update_source[bpg_xyz, tpb](dEY, cfg.J_SRC[n], cfg.bx+source_shift+1, cfg.by+1, cfg.sz, cfg.bx+source_shift+cfg.nx_swg-1, cfg.by+cfg.ny_swg-1, cfg.sz+2)
            elif usr['source_condition'] == 'TM':
                funcs.update_source[bpg_xyz, tpb](dHY, cfg.J_SRC[n], cfg.bx+source_shift+2, cfg.by+2, cfg.sz, cfg.bx+source_shift+cfg.nx_swg-2, cfg.by+cfg.ny_swg-2, cfg.sz+2)
        elif direction == 'b':
            if usr['source_condition'] == 'TE':
                funcs.update_source[bpg_xyz, tpb](dEY, cfg.J_SRC[n], cfg.bx+source_shift+1, cfg.by+1, cfg.nz-cfg.sz-1, cfg.bx+source_shift+cfg.nx_swg-1, cfg.by+cfg.ny_swg-1, cfg.nz-cfg.sz)
            elif usr['source_condition'] == 'TM':
                funcs.update_source[bpg_xyz, tpb](dHY, cfg.J_SRC[n], cfg.bx+source_shift+2, cfg.by+2, cfg.nz-cfg.sz-1, cfg.bx+source_shift+cfg.nx_swg-2, cfg.by+cfg.ny_swg-2, cfg.nz-cfg.sz)
    else:
        if usr['source_condition'] == 'TE':
            funcs.update_source[bpg_xyz, tpb](dEY, cfg.J_SRC[n], cfg.bx+1, cfg.by+1, cfg.sz, cfg.bx+cfg.nx_swg-1, cfg.by+cfg.ny_swg-1, cfg.sz+1)
        elif usr['source_condition'] == 'TM':
            funcs.update_source[bpg_xyz, tpb](dHY, cfg.J_SRC[n], cfg.bx+2, cfg.by+2, cfg.sz, cfg.bx+cfg.nx_swg-2, cfg.by+cfg.ny_swg-2, cfg.sz+1)

    # Update E-field components in PML
    funcs.update_pml_ex_yinc[bpg_xpz, tpb](dPSI_EX_YLO, dPSI_EX_YHI, dEX, dHZ, dBE, dCE, dCBX, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_ex_zinc[bpg_xyp, tpb](dPSI_EX_ZLO, dPSI_EX_ZHI, dEX, dHY, dBE, dCE, dCBX, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_ey_xinc[bpg_pyz, tpb](dPSI_EY_XLO, dPSI_EY_XHI, dEY, dHZ, dBE, dCE, dCBY, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_ey_zinc[bpg_xyp, tpb](dPSI_EY_ZLO, dPSI_EY_ZHI, dEY, dHX, dBE, dCE, dCBY, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_ez_xinc[bpg_pyz, tpb](dPSI_EZ_XLO, dPSI_EZ_XHI, dEZ, dHY, dBE, dCE, dCBZ, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_ez_yinc[bpg_xpz, tpb](dPSI_EZ_YLO, dPSI_EZ_YHI, dEZ, dHX, dBE, dCE, dCBZ, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)

    # Update H-field components
    funcs.update_hx[bpg_xyz, tpb](dEY, dEZ, dHX, dKYH, dKZH, DB, ib, jb, kb, ie, je, ke)
    funcs.update_hy[bpg_xyz, tpb](dEX, dEZ, dHY, dKXH, dKZH, DB, ib, jb, kb, ie, je, ke)
    funcs.update_hz[bpg_xyz, tpb](dEX, dEY, dHZ, dKXH, dKYH, DB, ib, jb, kb, ie, je, ke)

    # Update H-field components in PML
    funcs.update_pml_hx_yinc[bpg_xpz, tpb](dPSI_HX_YLO, dPSI_HX_YHI, dEZ, dHX, dBH, dCH, DB, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_hx_zinc[bpg_xyp, tpb](dPSI_HX_ZLO, dPSI_HX_ZHI, dEY, dHX, dBH, dCH, DB, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_hy_xinc[bpg_pyz, tpb](dPSI_HY_XLO, dPSI_HY_XHI, dEZ, dHY, dBH, dCH, DB, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_hy_zinc[bpg_xyp, tpb](dPSI_HY_ZLO, dPSI_HY_ZHI, dEX, dHY, dBH, dCH, DB, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_hz_xinc[bpg_pyz, tpb](dPSI_HZ_XLO, dPSI_HZ_XHI, dEY, dHZ, dBH, dCH, DB, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)
    funcs.update_pml_hz_yinc[bpg_xpz, tpb](dPSI_HZ_YLO, dPSI_HZ_YHI, dEX, dHZ, dBH, dCH, DB, cfg.delta_x, usr['num_cpml'], ib, jb, kb, ie, je, ke)

    # Record Data
    # funcs.simul_fft_efield[bpg_fft, tpb_fft](dEX1r, dEY1r, dEZ1r, dEX1i, dEY1i, dEZ1i, dEX, dEY, dEZ, cfg.cosE[n], cfg.sinE[n], usr['num_cpml'], usr['num_cpml'], cfg.nx-usr['num_cpml'], cfg.ny-usr['num_cpml'], cfg.p1z)
    # funcs.simul_fft_efield[bpg_fft, tpb_fft](dEX2r, dEY2r, dEZ2r, dEX2i, dEY2i, dEZ2i, dEX, dEY, dEZ, cfg.cosE[n], cfg.sinE[n], usr['num_cpml'], usr['num_cpml'], cfg.nx-usr['num_cpml'], cfg.ny-usr['num_cpml'], cfg.p2z)
    # funcs.simul_fft_hfield[bpg_fft, tpb_fft](dHX1r, dHY1r, dHZ1r, dHX1i, dHY1i, dHZ1i, dHX, dHY, dHZ, cfg.cosH[n], cfg.sinH[n], usr['num_cpml'], usr['num_cpml'], cfg.nx-usr['num_cpml'], cfg.ny-usr['num_cpml'], cfg.p1z)
    # funcs.simul_fft_hfield[bpg_fft, tpb_fft](dHX2r, dHY2r, dHZ2r, dHX2i, dHY2i, dHZ2i, dHX, dHY, dHZ, cfg.cosH[n], cfg.sinH[n], usr['num_cpml'], usr['num_cpml'], cfg.nx-usr['num_cpml'], cfg.ny-usr['num_cpml'], cfg.p2z)
    # funcs.map_efield_plane_td[bpg_fft, tpb_fft](dDDD, dEYZ, dEZZ, dEX, dEY, dEZ, n, cfg.cz, cfg.nx, cfg.ny)
    # funcs.map_hfield_plane_td[bpg_fft, tpb_fft](dDDD, dHYZ, dHZZ, dHX, dHY, dHZ, n, cfg.cz, cfg.nx, cfg.ny)

    funcs.map_efield_plane_td[bpg_fft, tpb_fft](dEX1, dEY1, dDDD, dEX, dEY, dEZ, n, cfg.p1z, cfg.nx, cfg.ny)
    funcs.map_efield_plane_td[bpg_fft, tpb_fft](dEX2, dEY2, dDDD, dEX, dEY, dEZ, n, cfg.p2z, cfg.nx, cfg.ny)
    funcs.map_hfield_plane_td[bpg_fft, tpb_fft](dHX1, dHY1, dDDD, dHX, dHY, dHZ, n, cfg.p1z, cfg.nx, cfg.ny)
    funcs.map_hfield_plane_td[bpg_fft, tpb_fft](dHX2, dHY2, dDDD, dHX, dHY, dHZ, n, cfg.p2z, cfg.nx, cfg.ny)

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
if usr['sim_type'] == 'zwave':
    pass
    # np.save('./{}/ey_zwave'.format(usr['output_dir']), dEYZ.copy_to_host(EYZ))
    # np.save('./{}/ez_zwave'.format(usr['output_dir']), dEZZ.copy_to_host(EZZ))
    # np.save('./{}/hy_zwave'.format(usr['output_dir']), dHYZ.copy_to_host(HYZ))
    # np.save('./{}/hz_zwave'.format(usr['output_dir']), dHZZ.copy_to_host(HZZ))
elif usr['sim_type'] == 's-param':
    if kind == 'i':
        if direction == 'f':
            np.save('./{}/ex_fi_l{}'.format(usr['output_dir'], line+1), dEX1.copy_to_host(EX1))
            np.save('./{}/ey_fi_l{}'.format(usr['output_dir'], line+1), dEY1.copy_to_host(EY1))
            np.save('./{}/hx_fi_l{}'.format(usr['output_dir'], line+1), dHX1.copy_to_host(HX1))
            np.save('./{}/hy_fi_l{}'.format(usr['output_dir'], line+1), dHY1.copy_to_host(HY1))
        elif direction == 'b':
            np.save('./{}/ex_bi_l{}'.format(usr['output_dir'], line+1), dEX2.copy_to_host(EX2))
            np.save('./{}/ey_bi_l{}'.format(usr['output_dir'], line+1), dEY2.copy_to_host(EY2))
            np.save('./{}/hx_bi_l{}'.format(usr['output_dir'], line+1), dHX2.copy_to_host(HX2))
            np.save('./{}/hy_bi_l{}'.format(usr['output_dir'], line+1), dHY2.copy_to_host(HY2))
    elif kind == 'r':
        if direction == 'f':
            np.save('./{}/ex_ft_l{}'.format(usr['output_dir'], line+1), dEX1.copy_to_host(EX1))
            np.save('./{}/ey_ft_l{}'.format(usr['output_dir'], line+1), dEY1.copy_to_host(EY1))
            np.save('./{}/hx_ft_l{}'.format(usr['output_dir'], line+1), dHX1.copy_to_host(HX1))
            np.save('./{}/hy_ft_l{}'.format(usr['output_dir'], line+1), dHY1.copy_to_host(HY1))

            np.save('./{}/ex_br_l{}'.format(usr['output_dir'], line+1), dEX2.copy_to_host(EX2))
            np.save('./{}/ey_br_l{}'.format(usr['output_dir'], line+1), dEY2.copy_to_host(EY2))
            np.save('./{}/hx_br_l{}'.format(usr['output_dir'], line+1), dHX2.copy_to_host(HX2))
            np.save('./{}/hy_br_l{}'.format(usr['output_dir'], line+1), dHY2.copy_to_host(HY2))
        elif direction == 'b':
            np.save('./{}/ex_fr_l{}'.format(usr['output_dir'], line+1), dEX1.copy_to_host(EX1))
            np.save('./{}/ey_fr_l{}'.format(usr['output_dir'], line+1), dEY1.copy_to_host(EY1))
            np.save('./{}/hx_fr_l{}'.format(usr['output_dir'], line+1), dHX1.copy_to_host(HX1))
            np.save('./{}/hy_fr_l{}'.format(usr['output_dir'], line+1), dHY1.copy_to_host(HY1))

            np.save('./{}/ex_bt_l{}'.format(usr['output_dir'], line+1), dEX2.copy_to_host(EX2))
            np.save('./{}/ey_bt_l{}'.format(usr['output_dir'], line+1), dEY2.copy_to_host(EY2))
            np.save('./{}/hx_bt_l{}'.format(usr['output_dir'], line+1), dHX2.copy_to_host(HX2))
            np.save('./{}/hy_bt_l{}'.format(usr['output_dir'], line+1), dHY2.copy_to_host(HY2))
else:
    pass

print('results saved!')
cuda.close()
