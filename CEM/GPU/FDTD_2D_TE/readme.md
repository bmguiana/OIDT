# OIDT FDTD 2D TE
### Author: Brian Guiana
### Date: 17 May 2022
## Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].
## Description
This set of code uses the 2D TE mode of FDTD simulation. The code is presented as 3 example file sets. This readme describes the intended operation instructions for each of these, as well as the output files expected.

## How to Generate Results
### Wave Impedance
1. Run `fdtd_main.py`.
2. Run `zwave.py`.

### Scattering Loss
A rough profiles folder is required for operation.
1. Run `fdtd_main.py`. If the roughness generation process does not converge, increase the rough_std, rough_acl, or tolerance values and try again
2. Run `automated_results_collection.py`.
3. Run `analysis.py`.

## Inputs
### Wave Impedance
- `analyze_wave_impedance.py`: 2 .npy files generated from `fdtd_main.py`
- `aux_funcs.py`: None
- `fdtd_config.py`: User specified parameters
- `fdtd_funcs.py`: None
- `fdtd_main.py`: Parameters from `fdtd_config.py`

### Calculate Scattering Loss
- `aux_funcs.py`: None
- `analysis.py`: 1 .npy file generated from `automated_results_collection.py`
- `automated_results_collection.py`: .npy files generated from `fdtd_main.py`
- `fdtd_config.py`: User specified parameters
- `fdtd_funcs.py`: None
- `fdtd_main.py`: Parameters from `fdtd_config.py`

## Outputs
### Wave Impedance
- `analyze_wave_impedance.py`: 1 figures showing the direct comparison between analytical and FDTD wave impedance results.
- `aux_funcs.py`: None
- `fdtd_config.py`: None
- `fdtd_funcs.py`: None
- `fdtd_main.py`: 2 .npy files.


### Calculate Scattering Loss
- `aux_funcs.py`: None
- `analysis.py`: Printed analysis of the .npy file from `automated_results_collection.py`
- `automated_results_collection.py`: 1 .npy file containing relevant information wrt scattering loss
- `fdtd_config.py`: None
- `fdtd_funcs.py`: None
- `fdtd_main.py`: 2 .npy files.
