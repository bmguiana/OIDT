# OIDT FDTD 2D TM
### Author: Brian Guiana
### Date: 10 December 2021
## Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].
## Description
This set of code uses the 2D TM mode of FDTD simulation. The code is presented as 3 example file sets. This readme describes the intended operation instructions for each of these, as well as the output files expected.

## How to Generate Results
### Wave Impedance
1. Run `fdtd_main.py`.
2. Run `analyze_wave_impedance.py`.

### Extract S-Parameters
1. Open `fdtd_config.py`. Leave this open, it will be used to set `sim` multiple times.
2. Set `sim=1`.
3. Save `fdtd_config.py`.
4. Run `fdtd_main.py`.
5. Set `sim=2`.
6. Save `fdtd_config.py`.
7. Run `fdtd_main.py`.
8. Set `sim=3`.
9. Save `fdtd_config.py`.
10. Run `fdtd_main.py`.
11. Set `sim=4`.
12. Save `fdtd_config.py`.
13. Run `fdtd_main.py`.
14. Run `extract_sparams.py`

### Calculate Scattering Loss
A rough profiles folder is required for operation.
1. Open `fdtd_config.py`. Leave this open, it will be used to set `sim` multiple times.
2. Set `sim=1`.
3. Save `fdtd_config.py`.
4. Run `fdtd_main.py`.
5. Set `sim=2`.
6. Save `fdtd_config.py`.
7. Run `fdtd_main.py`. If the roughness generation process does not converge, increase the rough_std and rough_acl values and repeat steps 6 and 7
8. Set `sim=3`.
9. Save `fdtd_config.py`.
10. Run `fdtd_main.py`.
11. Set `sim=4`.
12. Save `fdtd_config.py`.
13. Run `fdtd_main.py`.
14. Run `extract_sparams.py`

## Inputs
### Wave Impedance
- `analyze_wave_impedance.py`: 2 .npy files generated from `fdtd_main.py`
- `aux_funcs.py`: None
- `fdtd_config.py`: User specified parameters
- `fdtd_funcs.py`: None
- `fdtd_main.py`: Parameters from `fdtd_config.py`

### Extract S-Parameters
- `aux_funcs.py`: None
- `extract_sparams.py`: 6 .npy files generated from `fdtd_main.py`
- `fdtd_config.py`: User specified parameters
- `fdtd_funcs.py`: None
- `fdtd_main.py`: Parameters from `fdtd_config.py`

### Calculate Scattering Loss
- `aux_funcs.py`: None
- `calculate_alpha.py`: 6 .npy files generated from `fdtd_main.py`
- `fdtd_config.py`: User specified parameters
- `fdtd_funcs.py`: None
- `fdtd_main.py`: Parameters from `fdtd_config.py`

## Outputs
### Wave Impedance
- `analyze_wave_impedance.py`: 2 figures. The first shows the direct comparison between analytical and FDTD wave impedance results. The second shows the percentage error between those results.
- `aux_funcs.py`: None
- `fdtd_config.py`: None
- `fdtd_funcs.py`: None
- `fdtd_main.py`: 2 .npy files.

### Extract S-Parameters
- `aux_funcs.py`: None
- `extract_sparams.py`: 1 figure containing 2 subfigures showing the S-parameters for this setup.
- `fdtd_config.py`: None
- `fdtd_funcs.py`: None
- `fdtd_main.py`: 6 .npy files.

### Calculate Scattering Loss
- `aux_funcs.py`: None
- `calculate_alpha.py`: 1 figure showing the scattering loss over frequency through S-parameter simulation and direct calculation.
- `fdtd_config.py`: None
- `fdtd_funcs.py`: None
- `fdtd_main.py`: 6 .npy files.
