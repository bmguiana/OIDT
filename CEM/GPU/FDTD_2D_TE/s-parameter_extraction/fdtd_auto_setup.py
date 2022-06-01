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

# =============================================================================
# Import libraries
# =============================================================================

import numpy as np
import sys
import user_config as usr

# =============================================================================
# Automated simulation setup
# =============================================================================

# Physical constants and value normalization
eps = usr.eps_rel_bg / ( 36.0e9 * np.pi )            # Base permittivity (F/m)
mu = 4.0e-7 * np.pi                              # Base permeability (H/m)
c_bg = 1 / np.sqrt( eps * mu )                   # Base phase velocity (m/s)
eta = np.sqrt( mu / eps )                        # Base impedance (Ohms)
eps_rel_fg = usr.eps_rel_fg / usr.eps_rel_bg                         # Normalized foreground/core relative permittivity ()
f_max = usr.harmonics * usr.f0                           # Maximum simulation frequency (Hz)
vp = c_bg / np.sqrt( eps_rel_fg )                # Foreground/core phase velocity (m/s)
wl = vp / f_max                                  # Minimum wavelength (m)
wl_clad = c_bg / f_max                           # Cladding wavelength (m)
flight_time = usr.sgl_wg_length / vp                 # Single flight time (s)
sim_time = usr.num_flights * flight_time             # Simulation duration (s)
delta_x = wl / usr.points_per_wl                     # Spatial resolution (m/cell)
delta_t = delta_x / ( c_bg * np.sqrt( 2 ) )      # Temporal resolution (s/step)

delta_x = delta_x.astype(usr.precision)             # Change data type of dx to 4-byte precision
delta_t = delta_t.astype(usr.precision)             # Change data type of dt to 4-byte precision

buffer_yhat = int(usr.ny_clad_wl*wl_clad/delta_x)    # Buffer size, y-direction (cells)

# Convert length for S-param sims
args = sys.argv
if len(args) == 1:
    kind = 2
else:
    kind = args[3]  # Is this incident or reflected?

if kind == 'i':
    usr.sgl_wg_length = usr.port1_length + usr.port2_length

# Space-time discretization
nt = int( sim_time / delta_t )                   # Total simulation duration (steps)
nx_swg = int( usr.sgl_wg_length / delta_x )          # Single waveguide size, z-direction (cells)
ny_swg = int( usr.sgl_wg_height / delta_x )          # Single waveguide size, x-direction (cells)
pp = int( usr.pitch / delta_x )                                # Pitch between waveguides (cells)

bx = usr.num_cpml + usr.buffer_xhat                      # Border size, x-direction (cells)
by = usr.num_cpml + buffer_yhat                      # Border size, y-direction (cells)
nx = 2 * bx + nx_swg                                           # Computational domain size, x-direction (cells)
ny = 2 * by + ( usr.num_lines-1) * pp + ny_swg                 # Computational domain size, y-direction (cells)
rstd = usr.rough_std / delta_x                       # Normalized roughness standard deviation (cells)
racl = usr.rough_acl / delta_x                       # Normalized roughness autocorrelation length (cells)
tol_std = usr.tol_std / 100                                   # Normalized standard deviation tolerance ()
tol_acl = usr.tol_acl / 100                                   # Normalized correlation length tolerance ()

# Relational parameters
cx = nx // 2                                                   # Center cell, x-direction
cy = ny // 2                                                   # Center cell, y-direction
cx_swg = nx_swg // 2                                           # Single waveguide center cell, x-direction
cy_swg = ny_swg // 2                                           # Single waveguide center cell, y-direction
sx = bx + 0                                                    # Source point, x-direction
sy = by + cy_swg                                               # Source point, y-direction
ph = int( np.ceil( pp / 2 ) )

if buffer_yhat < pp//2:
    raise Exception('Boundary conditions too small for WG pitch setting. Reduce pitch or increase ny_clad_wl')

# =============================================================================
# Source assignment
# =============================================================================

# Gaussian pulse spread and peak time
gp_tsp = np.sqrt( -2 * np.log( 10**( usr.gauss_pulse_deg / 20 ) ) ) / ( 2 * np.pi * usr.f0 )
gp_tpk = 9 * gp_tsp

# Wave packet spread and peak time
wp_tsp = 2 * np.sqrt( np.log( 2 ) ) / ( usr.wave_packet_bw * 2 * np.pi * usr.f0 )
wp_tpk = 9 * wp_tsp

# Modify simulation time to include ramp-up period
if usr.source_type == 1:
    nt += int(gp_tpk/nt)
elif usr.source_type == 2:
    nt += int(wp_tpk/nt)
elif usr.source_type == 3:
    nt += int((gp_tpk+gp_tsp)/delta_t)
else:
    raise Exception('Pick a valid source type')     # Stop the program if an invalid source was chosen

# Array setup
GP = np.zeros(nt, dtype=usr.precision)       # Gaussian pulse shape array
WP = np.zeros(nt, dtype=usr.precision)       # Wave packet shape array
TH = np.zeros(nt, dtype=usr.precision)       # Time-harmonic signal shape array
RAMP = np.zeros(nt, dtype=usr.precision)     # Smooth ramp shape array

# Time step evaluation
for n in range(nt):
    GP[n] = np.exp(-0.5 * ( ( n*delta_t - gp_tpk ) / gp_tsp )**2 )
    WP[n] = np.exp(-0.5 * ( ( n*delta_t - wp_tpk ) / wp_tsp )**2 ) * np.exp( 2j * np.pi * usr.f0 * ( n * delta_t - wp_tpk ) ).real
    RAMP[n] = RAMP[n-1] + delta_t*GP[n]
    TH[n] = np.exp( 2j * np.pi * usr.f0 * ( n * delta_t - gp_tpk ) ).real
RAMP = RAMP / np.max( RAMP )                        # Set maximum magnitude of RAMP to 1

# Apply magnitude to shape
if usr.source_type == 1:
    J_SRC = usr.source_amp * GP            # Assign Gaussian pulse to source
elif usr.source_type == 2:
    J_SRC = usr.source_amp * WP            # Assign wave packet to source
elif usr.source_type == 3:
    J_SRC = usr.source_amp * RAMP * TH     # Assign time-harmonic + ramp up to source

p1x = sx + int( usr.port1_length / delta_x )
p2x = sx + nx_swg - int( usr.port2_length / delta_x )
if p2x < p1x:
    raise Exception('Port settling ranges overlap. Make the WG longer or decrease port settling range!')

profile_std = str(round(usr.rough_std*1e9))
profile_acl = str(round(usr.rough_acl*1e9))
roughness_profile = '_s{}nm_lc{}nm_r{}'.format(profile_std, profile_acl, usr.profile_number)
up = './rough_profiles/profile_upper'+roughness_profile
lp = './rough_profiles/profile_lower'+roughness_profile

cosE = np.zeros(nt, dtype=usr.precision)
cosH = np.zeros(nt, dtype=usr.precision)
sinE = np.zeros(nt, dtype=usr.precision)
sinH = np.zeros(nt, dtype=usr.precision)
for n in range(nt):
    cosE[n] = np.cos( 2 * np.pi * usr.f0 * (n+0.0) * delta_t, dtype=usr.precision)
    sinE[n] = np.sin( 2 * np.pi * usr.f0 * (n+0.0) * delta_t, dtype=usr.precision)
    cosH[n] = np.cos( 2 * np.pi * usr.f0 * (n+0.5) * delta_t, dtype=usr.precision)
    sinH[n] = np.sin( 2 * np.pi * usr.f0 * (n+0.5) * delta_t, dtype=usr.precision)
