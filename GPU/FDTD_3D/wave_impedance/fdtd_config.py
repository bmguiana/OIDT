"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

# =============================================================================
# Import libraries
# =============================================================================

import numpy as np
from aux_funcs import find_neff

# =============================================================================
# Material parameters
# =============================================================================

# Physical properties
eps_rel_bg = 1.5**2            # Background/cladding relative permittivity (, float)
eps_rel_fg = 3.5**2            # Foreground/core relative permittivity (, float)

# Geometry
sgl_wg_length = 14.0e-6        # Waveguide length (m, float)
sgl_wg_width = 2.0e-6          # Waveguide width (m, float)
sgl_wg_height = 0.2e-6         # Single waveguide height (m, float)
port1_length = 4.0e-6          # Mode settling length for port 1 (m, float)
port2_length = 0.0e-6          # Mode settling length for port 2 (m, float)

# Roughness
rough_std = 15.0e-9            # Target roughness standard deviation (m, float)
rough_acl = 700.0e-9           # Target roughness correlation length (m, float)
tol_std = 10.0                 # Standard deviation percentage tolerance (%, float)
tol_acl = 10.0                 # Correlation length percentage tolerance (%, float)

# =============================================================================
# Switches
# =============================================================================

rough_toggle = False           # Toggle sidewall roughness on/off (, bool)
                               #     True: waveguide sidewalls will vary w.r.t. length
                               #     False: waveguide sidewalls are do not vary

ctype = 3                      # Roughness profile correlation type for upper and lower profiles
                               #     1: Directly correlated (upper == lower => bend dominant)
                               #     2: Inversely correlated (upper == -lower => pinch dominant)
                               #     3: Uncorrelated (upper != lower => bend/pinch combo)
                               # The generation checks up to 300,000 unique profiles
                               #     for closeness to input parameters, or until
                               #     a valid profile is found. This is done for
                               #     both upper and lower profiles in "ctype=3"

# =============================================================================
# Outputs
# =============================================================================

# Directories
output_dir = 'Results'         # Output directory name (, str)
                               #     A new folder will be generated in the
                               #     working directory if one by the specified
                               #     name does not already exist. Do not
                               #     include the './' prefix.

# Output files
ey_zwave_name = 'ey_te_zwave'  # Time-domain output names (, str)
ez_zwave_name = 'ez_tm_zwave'  #     Only Ey and Hz are used in TE simulations
hy_zwave_name = 'hy_tm_zwave'  #     Only Ez and Hy are used in TM simulations
hz_zwave_name = 'hz_te_zwave'

profile_number = 0             # Roughness profile index number (, int)

# =============================================================================
# Source and resolution parameters
# =============================================================================

# Source parameters
source_type = 1                # Input source type (, int)
                               #     1: Gaussian pulse
                               #     2: Wave packet
                               #     3: Time-harmonic
source_amp = 5.0               # Source magnitude (V / m, float)
wave_packet_bw = 0.5           # Wave-packet bandwidth; may be safely ignored when not using st 2 (%, float)
gauss_pulse_deg = -6           # Frequency domain signal strength at f0, must be negative; may be safely ignored when not using st 1 (dB, float)
f0 = 194.8e12                  # Fundamental frequency; used for st 2 and st 3 (Hz, float)

source_condition = 'TM'        # Field excitation type, choose 'TE' or 'TM' (, str)

# Resolution and duration
harmonics = 1.000              # Number of harmonics above fundamental frequency to use (, float)
num_flights = 2.0              # Number of flight times to simulate (, float)
points_per_wl = 40             # Number of points per minimum wavelength (cells, int)

# Boundary conditions
num_cpml = 20                  # CPML layer size around simulation space (cells, int)
buffer_zhat = 20               # Buffer length between source and cpml region size, z-direction (cells, int)
nx_clad_wl = 0.75              # Buffer between waveguide core and cpml regions, x-direction (lambda, float)
ny_clad_wl = 0.75              # Buffer between waveguide core and cpml regions, y-direction (lambda, float)

# =============================================================================
# Automated simulation setup (Do not alter)
# =============================================================================

# Physical constants and value normalization
eps = eps_rel_bg / ( 36.0e9 * np.pi )            # Base permittivity (F/m)
mu = 4.0e-7 * np.pi                              # Base permeability (H/m)
c_bg = 1 / np.sqrt( eps * mu )                   # Base phase velocity (m/s)
eta = np.sqrt( mu / eps )                        # Base impedance (Ohms)
eps_rel_fg /= eps_rel_bg                         # Normalized foreground/core relative permittivity ()
f_max = harmonics * f0                           # Maximum simulation frequency (Hz)
vp = c_bg / np.sqrt( eps_rel_fg )                # Foreground/core phase velocity (m/s)
wl = vp / f_max                                  # Minimum wavelength (m)
wl_clad = c_bg / f_max                           # Cladding wavelength (m)
flight_time = sgl_wg_length / vp                 # Single flight time (s)
sim_time = num_flights * flight_time             # Simulation duration (s)
delta_x = wl / points_per_wl                     # Spatial resolution (m/cell)
delta_t = delta_x / ( c_bg * np.sqrt( 3 ) )      # Temporal resolution (s/step)

delta_x = delta_x.astype(np.float32)             # Change data type of dx to 4-byte precision
delta_t = delta_t.astype(np.float32)             # Change data type of dt to 4-byte precision

buffer_xhat = int(nx_clad_wl*wl_clad/delta_x)    # Buffer size, x-direction (cells)
buffer_yhat = int(ny_clad_wl*wl_clad/delta_x)    # Buffer size, y-direction (cells)

# Space-time discretization
nt = int( sim_time / delta_t )                   # Total simulation duration (steps)
nx_swg = int( sgl_wg_height / delta_x )          # Single waveguide size, x-direction (cells)
ny_swg = int( sgl_wg_width / delta_x )           # Single waveguide size, y-direction (cells)
nz_swg = int( sgl_wg_length / delta_x )          # Single waveguide size, z-direction (cells)
bx = num_cpml + buffer_xhat                      # Border size, x-direction (cells)
by = num_cpml + buffer_yhat                      # Border size, y-direction (cells)
bz = num_cpml + buffer_zhat                      # Border size, z-direction (cells)
nx = 2 * bx + nx_swg                             # Computational domain size, x-direction (cells)
ny = 2 * by + ny_swg                             # Computational domain size, y-direction (cells)
nz = 2 * bz + nz_swg                             # Computational domain size, z-direction (cells)
rstd = rough_std / delta_x                       # Normalized roughness standard deviation (cells)
racl = rough_acl / delta_x                       # Normalized roughness autocorrelation length (cells)
tol_std /= 100                                   # Normalized standard deviation tolerance ()
tol_acl /= 100                                   # Normalized correlation length tolerance ()

# Relational parameters
cx = nx // 2                                     # Center cell, x-direction
cy = ny // 2                                     # Center cell, y-direction
cz = nz // 2                                     # Center cell, z-direction
cx_swg = nx_swg // 2                             # Single waveguide center cell, x-direction
cy_swg = ny_swg // 2                             # Single waveguide center cell, y-direction
cz_swg = nz_swg // 2                             # Single waveguide center cell, z-direction
sx = bx + cx_swg                                 # Source point, x-direction
sy = by + cy_swg                                 # Source point, y-direction
sz = bz + 0                                      # Source point, z-direction

# =============================================================================
# Source assignment
# =============================================================================

# Gaussian pulse spread and peak time
gp_tsp = np.sqrt( -2 * np.log( 10**( gauss_pulse_deg / 20 ) ) ) / ( 2 * np.pi * f0 )
gp_tpk = 9 * gp_tsp

# Wave packet spread and peak time
wp_tsp = 2 * np.sqrt( np.log( 2 ) ) / ( wave_packet_bw * 2 * np.pi * f0 )
wp_tpk = 9 * wp_tsp

# Modify simulation time to include ramp-up period
if source_type == 1:
    nt += int(gp_tpk/nt)
elif source_type == 2:
    nt += int(wp_tpk/nt)
elif source_type == 3:
    nt += int((gp_tpk+gp_tsp)/delta_t)
else:
    raise Exception('Pick a valid source type')  # Stop the program if an invalid source was chosen

# Array setup
GP = np.zeros(nt, dtype=np.float32)              # Gaussian pulse shape array
WP = np.zeros(nt, dtype=np.float32)              # Wave packet shape array
TH = np.zeros(nt, dtype=np.float32)              # Time-harmonic signal shape array
RAMP = np.zeros(nt, dtype=np.float32)            # Smooth ramp shape array

# Time step evaluation
for n in range(nt):
    GP[n] = np.exp(-0.5 * ( ( n*delta_t - gp_tpk ) / gp_tsp )**2 )
    WP[n] = np.exp(-0.5 * ( ( n*delta_t - wp_tpk ) / wp_tsp )**2 ) * np.exp( 2j * np.pi * f0 * ( n * delta_t - wp_tpk ) ).real
    RAMP[n] = RAMP[n-1] + delta_t*GP[n]
    TH[n] = np.exp( 2j * np.pi * f0 * ( n * delta_t - gp_tpk ) ).real
RAMP = RAMP / np.max( RAMP )                     # Set maximum magnitude of RAMP to 1

# Apply magnitude to shape
if source_type == 1:
    J_SRC = source_amp * GP
elif source_type == 2:
    J_SRC = source_amp * WP
elif source_type == 3:
    J_SRC = source_amp * RAMP * TH

p1z = sz + int(port1_length/delta_x)
p2z = sz + nz_swg - int(port2_length/delta_x)
if p2z <= p1z:
    raise Exception('Port settling ranges overlap. Make the WG longer or decrease port settling range!')

profile_std = str(round(rough_std*1e9))
profile_acl = str(round(rough_acl*1e9))
roughness_profile = '_s'+profile_std+'nm_lc'+profile_acl+'nm_r'+str(profile_number)
up = './rough_profiles/profile_upper'+roughness_profile
lp = './rough_profiles/profile_lower'+roughness_profile

cosE = np.zeros(nt, dtype=np.float32)
cosH = np.zeros(nt, dtype=np.float32)
sinE = np.zeros(nt, dtype=np.float32)
sinH = np.zeros(nt, dtype=np.float32)
for n in range(nt):
    cosE[n] = np.cos( 2 * np.pi * f0 * (n+0.0) * delta_t, dtype=np.float32)
    sinE[n] = np.sin( 2 * np.pi * f0 * (n+0.0) * delta_t, dtype=np.float32)
    cosH[n] = np.cos( 2 * np.pi * f0 * (n+0.5) * delta_t, dtype=np.float32)
    sinH[n] = np.sin( 2 * np.pi * f0 * (n+0.5) * delta_t, dtype=np.float32)
