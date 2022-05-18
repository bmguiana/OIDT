"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

# =============================================================================
# Import Libraries
# =============================================================================

import numpy as np

# =============================================================================
# Material Parameters
# =============================================================================

# Physical properties
eps_rel_bg = 1.5**2            # Background/cladding relative permittivity (, float)
eps_rel_fg = 3.5**2            # Foreground/core relative permittivity (, float)

# Geometry
sgl_wg_length = 30e-6          # Waveguide length (m, float)
sgl_wg_width = 200e-9          # Single waveguide width (m, float)
port_length = 5e-6             # Port length for port 1 and 2 (m, float)

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
                               #     False: waveguide sidewalls do not vary

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
output_dir = 'Results'                           # Output directory name (, str)
                                                 #     A new folder will be generated in the
                                                 #     working directory if one by the specified
                                                 #     name does not already exist. Do not
                                                 #     include the './' prefix.

profile_number = 0             # Roughness profile index number (, int)

# =============================================================================
# Source Parameters
# =============================================================================

# Source shape
source_type = 1                # Input source type [st] (, int)
                               #     1: Gaussian pulse
                               #     2: Wave packet
                               #     3: Time-harmonic
source_amp = 5.0               # Source magnitude (V/m, float)
wave_packet_bw = 0.8           # Wave-packet bandwidth; may be safely ignored when not using st 2 (%, float)
gauss_pulse_deg = -6           # Frequency domain signal strength at f0, must be negative; may be safely ignored when not using st 1 (dB, float)
f0 = 194.8e12                  # Fundamental frequency; used for st 2 and st 3 (Hz, float)

# Resolution and duration
harmonics = 1.000              # Number of harmonics above fundamental frequency to use (, float)
num_flights = 2.0              # Number of flight times to simulate (, float)
points_per_wl = 40             # Number of points per minimum wavelength (cells, int)

# Boundary conditions
num_cpml = 20                  # CPML layer size around simulation space (cells, int)
buffer_xhat = 10               # Buffer between source and cpml region size, x-direction (cells, int)
ny_clad_wl = 2.0               # Buffer between waveguide core and cpml regions, y-direction (lambda, float)

# =============================================================================
# Automated Simulation Setup
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
delta_t = delta_x / ( c_bg * np.sqrt( 2 ) )      # Temporal resolution (s/step)

delta_x = delta_x.astype(np.float32)
delta_t = delta_t.astype(np.float32)
buffer_yhat = int(ny_clad_wl*wl_clad/delta_x)    # Buffer size, y-direction (cells)

# Spatial cell conversion
cpml_range = num_cpml + 1                        # Total CPML layers, including PEC boundary (cells)
nx_swg = int( sgl_wg_length / delta_x )          # Single waveguide size, x-direction (cells)
ny_swg = int( sgl_wg_width / delta_x )           # Single waveguide size, y-direction (cells)
bx = num_cpml + buffer_xhat                      # Border size, x-direction (cells)
by = num_cpml + buffer_yhat                      # Border size, y-direction (cells)
nx = 2 * bx + nx_swg                             # Computational domain size, x-direction (cells)
ny = 2 * by + ny_swg                             # Computational domain size, y-direction (cells)
rstd = rough_std / delta_x                       # Normalized roughness standard deviation (cells)
racl = rough_acl / delta_x                       # Normalized roughness autocorrelation length (cells)
tol_std /= 100                                   # Normalized standard deviation tolerance ()
tol_acl /= 100                                   # Normalized correlation length tolerance ()

# Time step conversion
nt = int( sim_time / delta_t )                   # Total simulation duration (steps)

# Relational parameters
cx = nx // 2                                     # Center cell, x-direction
cy = ny // 2                                     # Center cell, y-direction
cx_swg = nx_swg // 2                             # Single waveguide center cell, x-direction
cy_swg = ny_swg // 2                             # Single waveguide center cell, y-direction
sx = bx + 0                                      # Source point, x-direction
sy = by + cy_swg                                 # Source point, y-direction

# =============================================================================
# Source Assignment
# =============================================================================

# Gaussian pulse parameters
gp_tsp = np.sqrt( -2 * np.log( 10**( gauss_pulse_deg / 20 ) ) ) / ( 2 * np.pi * f0 )    # Pulse spread (s)
gp_tpk = 5 * gp_tsp                                                                     # Pulse peak time (s)

# Wave packet parameters
wp_tsp = 2 * np.sqrt(np.log(2)) / (wave_packet_bw * 2 * np.pi * f0)                       # Packet spread (s)
wp_tpk = 5 * wp_tsp                                                                     # Packet peak time (s)

# Array setup
GP = np.zeros(nt, dtype=np.float32)       # Gaussian pulse array
WP = np.zeros(nt, dtype=np.float32)       # Wave packet array
TH = np.zeros(nt, dtype=np.float32)       # Time harmonic signal array
RAMP = np.zeros(nt, dtype=np.float32)     # Auxiliary array for numeric integration of GP

# Time step evaluation
for n in range(nt):
    GP[n] = np.exp(-0.5 * ( (n*delta_t - gp_tpk) / gp_tsp )**2 )
    WP[n] = np.exp(-0.5 * ( (n*delta_t - wp_tpk) / wp_tsp )**2 ) * np.exp( 2j * np.pi * f0 * ( n * delta_t - wp_tpk ) ).real
    RAMP[n] = RAMP[n-1] + delta_t*GP[n]
    TH[n] = np.exp( 2j * np.pi * f0 * ( n * delta_t - gp_tpk ) ).real
RAMP = RAMP/np.max(RAMP)                            # Set maximum magnitude of RAMP to 1

# Apply magnitude to shape
if source_type == 1:
    J_SRC = source_amp * GP                         # Assign Gaussian pulse to source
elif source_type == 2:
    J_SRC = source_amp * WP            # Assign wave packet to source
elif source_type == 3:
    J_SRC = source_amp * delta_x**2 * RAMP * TH     # Assign time-harmonic + ramp up to source
else:
    raise Exception('Pick a valid source type')     # Stop the program if an invalid source was chosen

p1x = sx + int(port_length/delta_x)
p2x = nx - p1x
if p2x <= p1x:
    raise Exception('Port settling ranges overlap. Make the WG longer or decrease port settling range!')

profile_std = str(round(rough_std*1e9))
profile_acl = str(round(rough_acl*1e9))
roughness_profile = '_s'+profile_std+'nm_lc'+profile_acl+'nm_r'+str(profile_number)
up = './rough_profiles/profile_upper'+roughness_profile
lp = './rough_profiles/profile_lower'+roughness_profile
