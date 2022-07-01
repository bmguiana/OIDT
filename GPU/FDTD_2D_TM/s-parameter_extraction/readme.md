# OIDT FDTD 2D TM
### Author: Brian Guiana
### Date: 1 July 2022
## Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, in the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

## Description
The code contained within this folder is set up to automatically extract S-paramters from multiple tightly coupled waveguides. The code here has a modified structure to improve end-user ease of use compared to the other examples provided. In this release, users can input simulation parameters into `user_config.py` and run simulations using the new launcher `simulation_launcher.py`. The launcher has been tested using Spyder IDE and terminal-based Python console. Details about each included file are below.

## Run Instructions
1. Open `user_config.py`. Set the desired simulation parameters as needed.
2. Save `user_config.py`. The file may be closed after saving.
3. Run `simulation_launcher.py`. Changes to this file are neither necessary nor advised.

*Note*: The default settings have been tested on the Nvidia Quadro K6000 Graphical Processing Unit (GPU), and each simulation can be run in under 1 minute. At these settings, less than 500 MB of VRAM is required. A total of 4 simulations are required _*per line*_ and results are available only after all simulations have finished. The total completion time for the default settings (2 lines) was under 10 minutes. A minimum of 500 MB of disk space is required, but 2 GB is recommended.

## Files
This example requires the 7 included python scripts.
- `aux_funcs.py`: This file contains auxiliary functions with usage cases outside the primary FDTD environment, i.e., post-processing.
- `extract_sparams.py`: This file automatically converts the FDTD simulation outputs into S-parameters in touchstone format. The package `scikit-rf` is required to run this file.
- `fdtd_auto_setup.py`: This file automatically converts the inputs from `user_config.py` into a FDTD usable format, e.g., translate meters to cells.
- `fdtd_funcs.py`: This file contains functions with direct usage in the primary FDTD environement. These are mostly looping structures and a CUDA handler.
- `fdtd_main.py`: This file is the primary FDTD environment. The file includes array definition and main update loop.
- `simulation_launcher.py`: This file is the driver simulation driver for S-parameter extraction. This file automatically launches `fdtd_main.py` with appropriate settings for S-parameter extraction, followed by automatically launching `extract_sparams.py`.
- `user_config.py`: This file contains common configuration parameters for simulation of dielectric slab waveguides exhibiting sidewall roughness. While parameters in this file can be (mostly) freely modified, it is not recommended to edit any of the parameters in other files.
- `Results\example.s4p`: Example 4-port (2 line) S-parameter file extracted using this code. This uses roughness and does not correlate to the included pdf.
- `smooth_4-port_example.pdf`: Example 4-port (2 line) S-parameters across the output frequency range. These are shown in dB and are the result of smooth waveguide simulation. To reproduce this plot, simply change `rough_toggle` in `user_config.py` to `False`.
- Additional files are part of the SROPEE [16].

## Input Parameters
- `aux_funcs.py`: None.
- `extract_sparams.py`: None.
- `fdtd_auto_setup.py`: None.
- `fdtd_funcs.py`: None.
- `fdtd_main.py`: None.
- `simulation_launcher.py`: None.
- `user_config.py`: Geometrical properties, material properties, source conditions, some boundary conditions. Descriptions for each individual parameter are outlined within the file.

## Outputs
- `aux_funcs.py`: None.
- `extract_sparams.py`: S-parameters in touchstone format (`.sNp`). These are saved as real/imaginary over the frequency range of 100 THz to 300 THz.
- `fdtd_auto_setup.py`: None.
- `fdtd_funcs.py`: None.
- `fdtd_main.py`: Time-domain data in the form of `.npy` files. Each file is approximately 25 MB at default settings with 24 of those being generated for a 2 line S-parameter extraction simulation. During rough simulations the folder `rough_profiles` will be created if it does not already exist. During all simulation the folder named by `output_dir` in `user_config.py` will be created.
- `simulation_launcher.py`: Everything from `extract_sparams.py` and `fdtd_main.py`.
- `user_config.py`: None.

## Python Info
The below packages are required (in addition to those listed in the `CEM` directory):
- `scikit-rf`: 0.18.1
