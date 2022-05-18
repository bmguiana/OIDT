# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:30:51 2021

@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

# =============================================================================
# Import libraries
# =============================================================================

import numpy as np
from aux_funcs import find_neff

# Determine S-parameter simulation
sim = 4                         # Choose {1, 2, 3, 4}, Select 2 for singular rough sim

# =============================================================================
# Material parameters
# =============================================================================

# Physical properties
eps_rel_bg = 2.25               # Background/cladding relative permittivity (, float)
eps_rel_fg = 12.25              # Foreground/core relative permittivity (, float)

# Geometry
sgl_wg_length = 20.0e-6         # Single waveguide length (m, float)
sgl_wg_width = 200.0e-9         # Single waveguide width (m, float)
port_length = 5e-6

# Roughness
rough_std = 15.0e-9              # Roughness standard deviation (m, float)
rough_acl = 300.0e-9              # Roughness autocorrelation length (m, float)

# =============================================================================
# Switches
# =============================================================================

rough_toggle = True             # Toggle sidewall roughness on/off (, bool)
                                #     True: waveguide sidewalls will vary w.r.t. length
                                #     False: waveguide sidewalls are do not vary

# =============================================================================
# Outputs
# =============================================================================

# Directories
output_dir = 'Results'          # File output subfolder name (, str)

# Output files
if sim == 1:
    hz_port1_name = 'hz_p1i_p2i0'
    hz_port2_name = 'hold'
elif sim == 2:
    hz_port1_name = 'hz_p1t_p2i0'
    hz_port2_name = 'hz_p2r_p2i0'
elif sim == 3:
    hz_port1_name = 'hold'
    hz_port2_name = 'hz_p2i_p1i0'
elif sim == 4:
    hz_port1_name = 'hz_p1r_p1i0'
    hz_port2_name = 'hz_p2t_p1i0'

roughness_profile = '_s15_lc300_r0'

# =============================================================================
# Feedback
# =============================================================================

status_readout_interval = 500   # Number of time steps between feedback readouts (time steps, int)
                                # Feedback includes:
                                #     Current time step (steps)
                                #     Total time steps (steps)
                                #     Percentage completion (%)
                                #     How long the time stepping loop has been running (hr/min/s)
                                #     Average time for each time step (ms)
                                #     Estimated simulation time remaining (hr/min/s)
                                #     Memory Usage (GB)

# =============================================================================
# Source and resolution parameters
# =============================================================================

# Source shape
source_type = 2                 # Input source type [st] (, int)
                                #     1: Gaussian pulse
                                #     2: Wave packet
                                #     3: Time-harmonic
source_amp = 2.0                # Source magnitude (V / m, float)
wave_packet_bw = 0.8            # Wave-packet bandwidth; may be safely ignored when not using st 2 (%, float)
gauss_pulse_deg = -1            # Frequency domain signal strength at f0, must be negative; may be safely ignored when not using st 1 (dB, float)
f0 = 194.8e12                   # Fundamental frequency; used for st 2 and st 3 (Hz, float)

# Resolution and duration
harmonics = 2.000               # Number of harmonics above fundamental frequency to use (, float)
num_flights = 1.2               # Number of flight times to simulate (, float)
points_per_wl = 20              # Number of points per minimum wavelength (cells, int)

# =============================================================================
# Automated simulation setup
# =============================================================================

# Physical constants and value normalization
eps = eps_rel_bg / ( 36.0e9 * np.pi )               # Base permittivity (F/m)
mu = 4.0e-7 * np.pi                                 # Base permeability (H/m)
c_bg = 1 / np.sqrt( eps * mu )                      # Base phase velocity (m/s)
eta = np.sqrt( mu / eps )                           # Base impedance (Ohms)
eps_rel_fg /= eps_rel_bg                            # Normalized foreground/core relative permittivity ()
f_max = harmonics * f0                              # Maximum simulation frequency (Hz)
vp = c_bg / np.sqrt( eps_rel_fg )                   # Foreground/core phase velocity (m/s)
wl = vp / f_max                                     # Minimum wavelength (m)
flight_time = sgl_wg_length / vp                    # Single flight time (s)
sim_time = num_flights * flight_time                # Simulation duration (s)
delta_x = wl / points_per_wl                        # Spatial resolution (m/cell)
delta_t = delta_x / ( c_bg * np.sqrt( 2 ) )         # Temporal resolution (s/step)

# Boundary conditions
num_cpml = 20                   # CPML layer size around simulation space (cells, int)
buffer_xhat = 10                # Buffer between waveguide[s] and cpml region size, x-direction (cells, int)

# Get gamma value
k0 = 2*np.pi*f0*np.sqrt(mu*eps/eps_rel_bg)          # Free-space wave number (rad/m)
neff = find_neff(np.sqrt(eps_rel_fg*eps_rel_bg), np.sqrt(eps_rel_bg), sgl_wg_width/2, k0, mode='tm')
beta = neff*k0                                      # Waveguide wave number (rad/m)
gamma = np.sqrt(beta**2 - eps_rel_bg*k0**2)         # Effective half-width is d+1/gamma
buffer_yhat = int((5/gamma)/delta_x)                # Add appropriate buffer along y (cells)

# Correct waveguide length for S-parameter extraction
if sim == 1 or sim == 3:
    sgl_wg_length = 2*port_length

# Spatial cell conversion
cpml_range = num_cpml + 1                           # Total CPML layers, including PEC boundary (cells)
nx_swg = int( sgl_wg_length / delta_x )             # Single waveguide size, x-direction (cells)
ny_swg = int( sgl_wg_width / delta_x )              # Single waveguide size, y-direction (cells)
bx = num_cpml + buffer_xhat                         # Border size, x-direction (cells)
by = num_cpml + buffer_yhat                         # Border size, y-direction (cells)
nx = 2 * bx + nx_swg                                # Computational domain size, x-direction (cells)
ny = 2 * by + ny_swg                                # Computational domain size, y-direction (cells)
rstd = rough_std / delta_x                          # Normalized roughness standard deviation (cells)
racl = rough_acl / delta_x                          # Normalized roughness autocorrelation length (cells)

# Time step convertion
nt = int( sim_time / delta_t )                      # Total simulation duration (steps)

# Relational parameters
cx = nx // 2                                        # Center cell, x-direction
cy = ny // 2                                        # Center cell, y-direction
cx_swg = nx_swg // 2                                # Single waveguide center cell, x-direction
cy_swg = ny_swg // 2                                # Single waveguide center cell, y-direction
sx = bx + 0                                         # Source point, x-direction
sy = by + cy_swg                                    # Source point, y-direction

# =============================================================================
# Source assignment
# =============================================================================

# Gaussian pulse parameters
gp_tsp = np.sqrt( -2 * np.log( 10**( gauss_pulse_deg / 20 ) ) ) / ( 2 * np.pi * f0 )    # Pulse spread (s)
gp_tpk = 5 * gp_tsp                                                                     # Pulse peak time (s)

# Wave packet parameters
wp_tsp = 2 * np.sqrt( np.log( 2 ) ) / ( wave_packet_bw * 2 * np.pi * f0 )               # Packet spread (s)
wp_tpk = 9 * wp_tsp                                                                     # Packet peak time (s)

# Array setup
GP = np.zeros(nt)       # Gaussian pulse shape array
WP = np.zeros(nt)       # Wave packet shape array
TH = np.zeros(nt)       # Time-harmonic signal shape array
RAMP = np.zeros(nt)     # Smooth ramp shape array

# Time step evaluation
for n in range(nt):
    GP[n] = np.exp(-0.5 * ( ( n*delta_t - gp_tpk ) / gp_tsp )**2 )
    WP[n] = np.exp(-0.5 * ( ( n*delta_t - wp_tpk ) / wp_tsp )**2 ) * np.exp( 2j * np.pi * f0 * ( n * delta_t - wp_tpk ) ).real
    RAMP[n] = RAMP[n-1] + delta_t*GP[n]
    TH[n] = np.exp( 2j * np.pi * f0 * ( n * delta_t - gp_tpk ) ).real
RAMP = RAMP / np.max( RAMP )                        # Set maximum magnitude of RAMP to 1

# Apply magnitude to shape
if source_type == 1:
    J_SRC = source_amp * GP            # Assign Gaussian pulse to source
elif source_type == 2:
    J_SRC = source_amp * WP                         # Assign wave packet to source
elif source_type == 3:
    J_SRC = source_amp * RAMP * TH     # Assign time-harmonic + ramp up to source
else:
    raise Exception('Pick a valid source type')     # Stop the program if an invalid source was chosen

p1x = sx + int(port_length/delta_x)
p2x = -1*p1x
