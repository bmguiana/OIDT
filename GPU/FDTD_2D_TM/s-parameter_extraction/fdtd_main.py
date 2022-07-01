"""
Author: B. Guiana

Description:


Acknowledgement: This project was completed as part of research conducted with
                 my major professor and advisor, Prof. Ata Zadehgol, as part of
                 the Applied and Computational Electromagnetics Signal and
                 Power Integrity (ACEM-SPI) Lab while working toward the Ph.D.
                 in Electrical Engineering at the University of Idaho, Moscow,
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

print('The simulation size will be: {:} x {:} square cells over {:} time steps\n\n\n'.format(cfg.nx, cfg.ny, cfg.nt))
aux.mkdir(usr['output_dir'])
cuda.select_device(0)

# =============================================================================
# Main array definition
# =============================================================================

EZ = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])     # Electric field at current time step, z-component
HX = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])     # Magnetic field at current time step, x-component
HY = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])     # Magnetic field at current time step, y-component

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
            FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.bx, cfg.by, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1x, cfg.nx-cfg.p2x, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth', out_dir=usr['prof_dir'])
        elif line == 0 and direction == 'f' and kind == 'r':
            # Make new roughness profiles during first line, first reflected sim
            FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.bx, cfg.by, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1x, cfg.nx-cfg.p2x, cfg.rstd, cfg.racl, cfg.delta_x, out_dir=usr['prof_dir'], upper_name=cfg.up, lower_name=cfg.lp, atol=cfg.tol_acl, stol=cfg.tol_std, mtol=0.01)
        else:
            # Load the roughness mask for all other sim types. This conditional hierarchy should make these only during reflected sims
            FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.bx, cfg.by, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1x, cfg.nx-cfg.p2x, cfg.rstd, cfg.racl, cfg.delta_x, mode='load', out_dir=usr['prof_dir'], upper_name=cfg.up+'.npy', lower_name=cfg.lp+'.npy', atol=cfg.tol_acl, stol=cfg.tol_std, mtol=0.01)
    else:
        # Generate only smooth mask if no roughness
        FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.bx, cfg.by, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1x, cfg.nx-cfg.p2x, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth', out_dir=usr['prof_dir'])
else:
    FG_REG = funcs.generate_fg_mask(cfg.nx, cfg.ny, cfg.bx, cfg.by, cfg.ny_swg, cfg.pp, usr['num_lines'], cfg.p1x, cfg.nx-cfg.p2x, cfg.rstd, cfg.racl, cfg.delta_x, mode='smooth', out_dir=usr['prof_dir'])

EPS_MASK = cfg.eps * ( cfg.eps_rel_fg * FG_REG + 1 * ~FG_REG )
EPS_EZ = EPS_MASK.copy().astype(usr['precision'])
CBZ = cfg.delta_t / EPS_EZ
DB = cfg.delta_t / cfg.mu

# Define data storage arrays
EZr = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
HXr = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
HYr = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])

EZi = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
HXi = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])
HYi = np.zeros([cfg.nx, cfg.ny], dtype=usr['precision'])

EZ_P1 = np.zeros([cfg.nt, cfg.ny], dtype=usr['precision'])
HX_P1 = np.zeros([cfg.nt, cfg.ny], dtype=usr['precision'])
HY_P1 = np.zeros([cfg.nt, cfg.ny], dtype=usr['precision'])

EZ_P2 = np.zeros([cfg.nt, cfg.ny], dtype=usr['precision'])
HX_P2 = np.zeros([cfg.nt, cfg.ny], dtype=usr['precision'])
HY_P2 = np.zeros([cfg.nt, cfg.ny], dtype=usr['precision'])

EZZ = np.zeros([cfg.nt, cfg.ny], dtype=usr['precision'])
HXZ = np.zeros([cfg.nt, cfg.ny], dtype=usr['precision'])
HYZ = np.zeros([cfg.nt, cfg.ny], dtype=usr['precision'])

# =============================================================================
# Begin CPML
# =============================================================================

# Computational Domain Limits
offset = 2
ib = 0+offset
jb = 0+offset
ie = cfg.nx - ib
je = cfg.ny - jb

# Constants
m_grade = 4                                                              # Grading maximum exponent for sigma and kappa
m_alpha = 1                                                              # Grading maximum exponent for alpha
cpml_sigma_optimal = 0.8 * ( m_grade + 1 ) / ( cfg.eta * cfg.delta_x )   # Optimal sigma value
cpml_sigma_max = 1.2 * cpml_sigma_optimal                                # Maximum sigma value
cpml_alpha_max = 0.05                                                    # Maximum alpha value
cpml_kappa_max = 5                                                       # Maximum kappa value

# Define arrays
PSI_EZ_XLO = np.zeros([usr['num_cpml'], cfg.ny], dtype=usr['precision'])  # Ez correction field, low-side x-boundary
PSI_EZ_XHI = np.zeros([usr['num_cpml'], cfg.ny], dtype=usr['precision'])  # Ez correction field, high-side x-boundary
PSI_EZ_YLO = np.zeros([cfg.nx, usr['num_cpml']], dtype=usr['precision'])  # Ez correction field, low-side y-boundary
PSI_EZ_YHI = np.zeros([cfg.nx, usr['num_cpml']], dtype=usr['precision'])  # Ez correction field, high-side y-boundary

PSI_HX_YLO = np.zeros([cfg.nx, usr['num_cpml']], dtype=usr['precision'])  # Hx correction field, low-side y-boundary
PSI_HX_YHI = np.zeros([cfg.nx, usr['num_cpml']], dtype=usr['precision'])  # Hx correction field, high-side y-boundary

PSI_HY_XLO = np.zeros([usr['num_cpml'], cfg.ny], dtype=usr['precision'])  # Hy correction field, low-side x-boundary
PSI_HY_XHI = np.zeros([usr['num_cpml'], cfg.ny], dtype=usr['precision'])  # Hy correction field, high-side x-boundary

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
KXH = np.ones(cfg.nx, dtype=usr['precision']) * cfg.delta_x                    # Magnetic field x-direction kappa division
KYH = np.ones(cfg.ny, dtype=usr['precision']) * cfg.delta_x                    # Magnetic field y-direction kappa division

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
    KXH[ib+usr['num_cpml']-d] = KH[d] * cfg.delta_x      # dx
    KYH[jb+usr['num_cpml']-d] = KH[d] * cfg.delta_x      # dy
    KXE[ie-(d+1)] = KE[-(d+1)] * cfg.delta_x          # dx
    KYE[je-(d+1)] = KE[-(d+1)] * cfg.delta_x          # dy
    KXH[ie-(d+1)] = KH[-(d+1)] * cfg.delta_x          # dx
    KYH[je-(d+1)] = KH[-(d+1)] * cfg.delta_x          # dy

# =============================================================================
# GPU Processing Setup
# =============================================================================

# Declare thread sizes
tpbx = 1
tpby = 128

# Declare block sizes
bpgx = int(np.ceil(cfg.nx / tpbx))
bpgy = int(np.ceil(cfg.ny / tpby))
bpgx_cpml = int(np.ceil(usr['num_cpml'] / tpbx))
bpgy_cpml = int(np.ceil(usr['num_cpml'] / tpby))

# Combine into tuples
tpb = (tpbx, tpby)
bpg_xy = (bpgx, bpgy)
bpg_xp = (bpgx, bpgy_cpml)
bpg_py = (bpgx_cpml, bpgy)

# GPU process feedbak
cells_on_device = 10*(cfg.nx * cfg.ny) + 4*usr['num_cpml']*(cfg.nx + cfg.ny + 1) + 2*(cfg.nx + cfg.ny)
cells_on_device += 9*(cfg.nt * cfg.ny)
if usr['precision'] == np.float32:
    device_req_mem = 4*cells_on_device / 1024 / 1024
elif usr['precision'] == np.float64:
    device_req_mem = 8*cells_on_device / 1024 / 1024
print('Transferring {:.0f} Mcells onto GPU, requiring {:} MB'.format(cells_on_device/1e6, device_req_mem))

# Device arrays of size (nx*ny)
dEZ = cuda.to_device(EZ)
dHX = cuda.to_device(HX)
dHY = cuda.to_device(HY)
dCBZ = cuda.to_device(CBZ)

# Device arrays of size (num_cpml*ny)
dPSI_EZ_XLO = cuda.to_device(PSI_EZ_XLO)
dPSI_EZ_XHI = cuda.to_device(PSI_EZ_XHI)
dPSI_HY_XLO = cuda.to_device(PSI_HY_XLO)
dPSI_HY_XHI = cuda.to_device(PSI_HY_XHI)

# Device arrays of size (nx*num_cpml)
dPSI_EZ_YLO = cuda.to_device(PSI_EZ_YLO)
dPSI_EZ_YHI = cuda.to_device(PSI_EZ_YHI)
dPSI_HX_YLO = cuda.to_device(PSI_HX_YLO)
dPSI_HX_YHI = cuda.to_device(PSI_HX_YHI)

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

# Data recording arrays
dEZr = cuda.to_device(EZr)  # nx x ny
dHXr = cuda.to_device(HXr)  # nx x ny
dHYr = cuda.to_device(HYr)  # nx x ny
dEZi = cuda.to_device(EZi)  # nx x ny
dHXi = cuda.to_device(HXi)  # nx x ny
dHYi = cuda.to_device(HYi)  # nx x ny
dEZZ = cuda.to_device(EZZ)  # nt x ny
dHXZ = cuda.to_device(HXZ)  # nt x ny
dHYZ = cuda.to_device(HYZ)  # nt x ny
dEZ_P1 = cuda.to_device(EZ_P1)  # nt x ny
dHX_P1 = cuda.to_device(HX_P1)  # nt x ny
dHY_P1 = cuda.to_device(HY_P1)  # nt x ny
dEZ_P2 = cuda.to_device(EZ_P2)  # nt x ny
dHX_P2 = cuda.to_device(HX_P2)  # nt x ny
dHY_P2 = cuda.to_device(HY_P2)  # nt x ny

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
    funcs.update_ez[bpg_xy, tpb](dEZ, dHX, dHY, dKXE, dKYE, dCBZ, ib, jb, ie, je)

    # Update source conditions
    if usr['sim_type'] == 's-param':
        if direction == 'f':
            funcs.update_source[bpg_xy, tpb](dEZ, cfg.J_SRC[n], cfg.sx, cfg.by+source_shift+1, cfg.sx+1, cfg.by+source_shift+cfg.ny_swg-1)
        elif direction == 'b':
            funcs.update_source[bpg_xy, tpb](dEZ, cfg.J_SRC[n], cfg.nx-cfg.sx-1, cfg.by+source_shift+1, cfg.nx-cfg.sx, cfg.by+source_shift+cfg.ny_swg-1)
    else:
        funcs.update_source[bpg_xy, tpb](dEZ, cfg.J_SRC[n], cfg.sx, cfg.by+source_shift+1, cfg.sx+1, cfg.by+source_shift+cfg.ny_swg-1)

    # Update E-field components in PML
    funcs.update_pml_ez_xinc[bpg_py, tpb](dPSI_EZ_XLO, dPSI_EZ_XHI, dEZ, dHY, dBE, dCE, dCBZ, cfg.delta_x, usr['num_cpml'], ib, jb, ie, je)
    funcs.update_pml_ez_yinc[bpg_xp, tpb](dPSI_EZ_YLO, dPSI_EZ_YHI, dEZ, dHX, dBE, dCE, dCBZ, cfg.delta_x, usr['num_cpml'], ib, jb, ie, je)

    # Update H-field components
    funcs.update_hx[bpg_xy, tpb](dEZ, dHX, dKYH, DB, ib, jb, ie, je)
    funcs.update_hy[bpg_xy, tpb](dEZ, dHY, dKXH, DB, ib, jb, ie, je)

    # Update H-field components in PML
    funcs.update_pml_hx_yinc[bpg_xp, tpb](dPSI_HX_YLO, dPSI_HX_YHI, dEZ, dHX, dBH, dCH, DB, cfg.delta_x, usr['num_cpml'], ib, jb, ie, je)
    funcs.update_pml_hy_xinc[bpg_py, tpb](dPSI_HY_XLO, dPSI_HY_XHI, dEZ, dHY, dBH, dCH, DB, cfg.delta_x, usr['num_cpml'], ib, jb, ie, je)

    # Record Data
    # funcs.simul_fft_efield[bpg_xy, tpb](dEZr, dEZi, dEZ, cfg.cosE[n], cfg.sinE[n], usr['num_cpml'], usr['num_cpml'], cfg.nx-usr['num_cpml'], cfg.ny-usr['num_cpml'])
    # funcs.simul_fft_hfield[bpg_xy, tpb](dHXr, dHYr, dHXi, dHYi, dHX, dHY, cfg.cosH[n], cfg.sinH[n], usr['num_cpml'], usr['num_cpml'], cfg.nx-usr['num_cpml'], cfg.ny-usr['num_cpml'])
    # funcs.map_efield_yline[bpgy, tpby](dEZZ, dEZ, n, cfg.cx, cfg.ny)
    # funcs.map_hfield_yline[bpgy, tpby](dHXZ, dHYZ, dHX, dHY, n, cfg.cx, cfg.ny)

    funcs.map_efield_yline[bpgy, tpby](dEZ_P1, dEZ, n, cfg.p1x, cfg.ny)
    funcs.map_efield_yline[bpgy, tpby](dEZ_P2, dEZ, n, cfg.p2x, cfg.ny)
    funcs.map_hfield_yline[bpgy, tpby](dHX_P1, dHY_P1, dHX, dHY, n, cfg.p1x, cfg.ny)
    funcs.map_hfield_yline[bpgy, tpby](dHX_P2, dHY_P2, dHX, dHY, n, cfg.p2x, cfg.ny)

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
        print('Memory used:              {:6.3f} (MB)'.format(process.memory_info().rss/1024/1024))
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
dEZr.copy_to_host(EZr)
dHXr.copy_to_host(HXr)
dHYr.copy_to_host(HYr)
dEZi.copy_to_host(EZi)
dHXi.copy_to_host(HXi)
dHYi.copy_to_host(HYi)

if usr['precision'] == np.float32:
    EZF = np.zeros([cfg.nx, cfg.ny], dtype=np.complex64)
    HXF = np.zeros([cfg.nx, cfg.ny], dtype=np.complex64)
    HYF = np.zeros([cfg.nx, cfg.ny], dtype=np.complex64)
elif usr['precision'] == np.float64:
    EZF = np.zeros([cfg.nx, cfg.ny], dtype=np.complex128)
    HXF = np.zeros([cfg.nx, cfg.ny], dtype=np.complex128)
    HYF = np.zeros([cfg.nx, cfg.ny], dtype=np.complex128)

EZF = EZr + 1j*EZi
HXF = HXr + 1j*HXi
HYF = HYr + 1j*HYi

if usr['sim_type'] == 'full_fft':
    np.save('./{}/ez_fft{}'.format(usr['output_dir'], cfg.roughness_profile), EZF)
    np.save('./{}/hx_fft{}'.format(usr['output_dir'], cfg.roughness_profile), HXF)
    np.save('./{}/hy_fft{}'.format(usr['output_dir'], cfg.roughness_profile), HYF)
elif usr['sim_type'] == 'zwave':
    np.save('./{}/ez_zwave'.format(usr['output_dir']), dEZZ.copy_to_host(EZZ))
    np.save('./{}/hx_zwave'.format(usr['output_dir']), dHXZ.copy_to_host(HXZ))
    np.save('./{}/hy_zwave'.format(usr['output_dir']), dHYZ.copy_to_host(HYZ))
elif usr['sim_type'] == 's-param':
    if kind == 'i':
        if direction == 'f':
            np.save('./{}/ez_fi_l{}'.format(usr['output_dir'], line+1), dEZ_P1.copy_to_host(EZ_P1))
            np.save('./{}/hy_fi_l{}'.format(usr['output_dir'], line+1), dHY_P1.copy_to_host(HY_P1))
        elif direction == 'b':
            np.save('./{}/ez_bi_l{}'.format(usr['output_dir'], line+1), dEZ_P2.copy_to_host(EZ_P2))
            np.save('./{}/hy_bi_l{}'.format(usr['output_dir'], line+1), dHY_P2.copy_to_host(HY_P2))
    elif kind == 'r':
        if direction == 'f':
            np.save('./{}/ez_ft_l{}'.format(usr['output_dir'], line+1), dEZ_P1.copy_to_host(EZ_P1))
            np.save('./{}/hy_ft_l{}'.format(usr['output_dir'], line+1), dHY_P1.copy_to_host(HY_P1))
            np.save('./{}/ez_br_l{}'.format(usr['output_dir'], line+1), dEZ_P2.copy_to_host(EZ_P2))
            np.save('./{}/hy_br_l{}'.format(usr['output_dir'], line+1), dHY_P2.copy_to_host(HY_P2))
        elif direction == 'b':
            np.save('./{}/ez_fr_l{}'.format(usr['output_dir'], line+1), dEZ_P1.copy_to_host(EZ_P1))
            np.save('./{}/hy_fr_l{}'.format(usr['output_dir'], line+1), dHY_P1.copy_to_host(HY_P1))
            np.save('./{}/ez_bt_l{}'.format(usr['output_dir'], line+1), dEZ_P2.copy_to_host(EZ_P2))
            np.save('./{}/hy_bt_l{}'.format(usr['output_dir'], line+1), dHY_P2.copy_to_host(HY_P2))
else:
    pass

print('results saved!')
cuda.close()
