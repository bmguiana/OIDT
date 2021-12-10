# OIDT FDTD 3D
### Author: Brian Guiana
### Date: 10 December 2021
## Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].
## Description
This set of code uses the 3D mode of FDTD simulation. The code is presented as 1 example file set. This readme describes the intended operation instructions for each of these, as well as the output files expected.

## Operating Note
3D FDTD can be highly computationally expensive. The presented parameters in `fdtd_config.py` were tested using 2 parallel Intel® Xeon® E5- 2687W v3 CPUs (40 logical cores), operating at 3.10 GHz. This test resulted in RAM usage of approximately 4 GB and simulation results were achieved after 10 minutes. Running this code on a single core machine would require simulation in excess of 6.5 hours. Increasing the spatial or temporal resolution would likely result in a memory requirement not possible on most machines. Making large changes to input parameters is not advised in this case.

## How to Generate Results
### Wave Impedance
1. Run `fdtd_main.py`.
2. Run `analyze_wave_impedance.py`.

## Inputs
### Wave Impedance
- `analyze_wave_impedance.py`: 2 .npy files generated from `fdtd_main.py`
- `aux_funcs.py`: None
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
