"""
Author: B. Guiana

Description: User specified simulation parameters. A short description of each
             varaible follows as an in-line comment. Units and data types are
             provided in parentheses as (unit, dtype) where applicable.

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

# =============================================================================
# Material parameters
# =============================================================================

# Physical properties
eps_rel_bg = 1.5**2            # Background/cladding relative permittivity (, float)
eps_rel_fg = 3.5**2            # Foreground/core relative permittivity (, float)

# Geometry
sgl_wg_length = 15.0e-6        # Waveguide length (m, float)
sgl_wg_width = 0.5e-6          # Single waveguide width (m, float) MAKE THIS WIDER!!!
sgl_wg_height = 0.2e-6         # Single waveguide height (m, float)

pitch = 1.0e-6                 # Pitch between waveguides (m, float)
num_lines = 2                  # Number of parallel waveguides (, int)

port1_length = 2.0e-6          # Modal settling length for port 1 (m, float)
port2_length = 2.0e-6          # Modal settling length for port 2 (m, float)

# Roughness
rough_std = 20.0e-9            # Target roughness standard deviation (m, float)
rough_acl = 500.0e-9           # Target roughness correlation length (m, float)
tol_std = 10.0                 # Standard deviation percentage tolerance (%, float)
tol_acl = 10.0                 # Correlation length percentage tolerance (%, float)

# =============================================================================
# Switches
# =============================================================================

rough_toggle = True            # Toggle sidewall roughness on/off (, bool)
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

precision = np.float32         # Floating point arithmetic precision. Choose np.float32 or np.float64
sim_type = 's-param'           # Simulation type. Choose 's-param'. Other outputs not ready yet.
auto_condition = True          # Automatically condition settings for "Source
                               #     parameters" and "Resolution and duration"
                               #     for the provided "S-Parameter Extraction"
                               #     settings (, bool)

feedback_at_n_percent = 1      # Interval between feedback readouts (%, int or float)
profile_number = 0             # Roughness profile index number (, int)

# =============================================================================
# Simulation parameters
# =============================================================================

# Outputs
output_dir = 'Results'         # Output directory name (, str)
                               #     A new folder will be generated in the
                               #     working directory if one by the specified
                               #     name does not already exist. Do not
                               #     include the './' prefix.
prof_dir = 'rough_profiles'    # Storage directory for rough profiles (, str)
                               #     A new folder will be generated in the
                               #     working directory.

sparam_file = 'te_fine_rough'  # Output file name for S-Parameter extraction (, str)

# S-Parameter Extraction
sparam_fmin = 170e12           # Minimum frequency for S-Parameter output file (Hz, float)
sparam_fmax = 230e12           # Maximum frequency for S-Parameter output file (Hz, float)
sparam_num_freqs = 10001       # Number of frequencies between fmin and fmax, inclusive (, int)

# Source parameters
source_type = 2                # Input source type (, int)
                               #     1: Gaussian pulse
                               #     2: Wave packet
                               #     3: Time-harmonic
source_amp = 5.0               # Source magnitude (V / m, float)
wave_packet_bw = 0.5           # Wave-packet bandwidth; may be safely ignored when not using st 2 (%, float)
gauss_pulse_deg = -6           # Frequency domain signal strength at f0, must be negative; may be safely ignored when not using st 1 (dB, float)
f0 = 194.8e12                  # Fundamental frequency; used for st 2 and st 3 (Hz, float)

source_condition = 'TE'        # Field excitation type, choose 'TE' or 'TM' (, str)

# Resolution and duration
harmonics = 1.000              # Number of harmonics above fundamental frequency to use (, float)
num_flights = 2.0              # Number of flight times to simulate (, float)
points_per_wl = 40             # Number of points per minimum wavelength (cells, int)

# Boundary conditions
num_cpml = 20                  # CPML layer size around simulation space (cells, int)
buffer_zhat = 20               # Buffer length between source and cpml region size, z-direction (cells, int)
nx_clad_wl = 1.5               # Buffer between waveguide core and cpml regions, x-direction (lambda, float)
ny_clad_wl = 1.5               # Buffer between waveguide core and cpml regions, y-direction (lambda, float)
