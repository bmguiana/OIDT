# OIDT FDTD 3D
### Author: Brian Guiana
### Date: 1 July 2022
## Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, in the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

## Description
The code contained within this folder is set up to automatically extract S-paramters from multiple tightly coupled waveguides. Compared to the other example folders, the code here has a modified structure to improve end-user ease of use. In this release, users can input simulation parameters into `user_config.py` and run simulations using the new launcher `simulation_launcher.py`. The launcher has been tested using Spyder IDE and terminal-based Python console. Details about each included file are below.

## Run Instructions
1. Open `user_config.py`. Set the desired simulation parameters as needed.
2. Save `user_config.py`. The file may be closed after saving.
3. Run `simulation_launcher.py`. Changes to this file are neither necessary nor advised.

*Note*: The default settings have been tested on the Nvidia Quadro K6000 Graphical Processing Unit (GPU). Each incident simulation can be run in approximately 2 minutes, and each reflected simulation can be run in approximately 6 minutes. At these settings, approximately 10.5 GB of VRAM is required. A total of 4 simulations are required _*per line*_ and results are available only after all simulations have finished. The total completion time for the default settings (2 lines) was approximately 40 minutes. A minimum of 50 GB of disk space is required, but 100 GB is recommended. All files with the `.npy` extention may be deleted after the touchstone file has been created.

On the Nvidia RTX A6000 GPU, the incident and reflected simulations finished in approximately 20 seconds and 50 seconds, respectively. Computation speeds for both the K6000 and A6000 are below. Numbers are shown in millions of cells per second (MCells/s) [15]. Incident simulations have a shorter completion time but lower MCells/s speed due to the reduced simulation size.

K6000
Incident: 512 MCells/s
Reflected: 550 MCells/s

A6000:
Incident: 3447 MCells/s
Reflected: 4033 MCells/s


## Files
### Main Program Files
This example requires the 7 included python scripts.
- `aux_funcs.py`: This file contains auxiliary functions with usage cases outside the primary FDTD environment, i.e., post-processing.
- `extract_sparams.py`: This file automatically converts the FDTD simulation outputs into S-parameters in touchstone format. The package `scikit-rf` is required to run this file.
- `fdtd_auto_setup.py`: This file automatically converts the inputs from `user_config.py` into a FDTD usable format, e.g., translate meters to cells. The source conditions here use the _Analytical_ naming convention, which is counter to the _FDTD_ naming convention. Analytical TE is most similar to FDTD TM. Likewise, analytical TM is most similar to FDTD TE.
- `fdtd_funcs.py`: This file contains functions with direct usage in the primary FDTD environement. These are mostly looping structures and a CUDA handler.
- `fdtd_main.py`: This file is the primary FDTD environment. The file includes array definition and main update loop.
- `simulation_launcher.py`: This file is the driver simulation driver for S-parameter extraction. This file automatically launches `fdtd_main.py` with appropriate settings for S-parameter extraction, followed by automatically launching `extract_sparams.py`.
- `user_config.py`: This file contains common configuration parameters for simulation of dielectric slab waveguides exhibiting sidewall roughness. While parameters in this file can be (mostly) freely modified, it is not recommended to edit any of the parameters in other files.

### Supporting Files
- `FDTD_TM-like_example.pdf`: Example 4-port (2 line) S-parameters in the available frequency range (100 THz to 300 THz). This example was generated for smooth waveguides using E-field excitation (along width), and is most similar to the simulations found in the folder `FDTD_2D_TM`. This mode can be activated by using `souce_condition = 'TE'` in `user_config.py`. See section 2.2.3 in [8] for coordinate tranforms from FDTD TM to Analytical TE mode operation.
- `FDTD_TE-like_example.pdf`:Example 4-port (2 line) S-parameters in the available frequency range (100 THz to 300 THz). This example was generated for smooth waveguides using H-field excitation (along width), and is most similar to the simulations found in the folder `FDTD_2D_TE`. This mode can be activated by using `souce_condition = 'TM'` in `user_config.py`. Section 2.2.3 in [8] details coordinate tranforms from FDTD TM to Analytical TE mode operation, but this same operation can be used to translate FDTD TE to Analytical TM.
- `S-Parameter_geometry.pdf`: Layout of 3D waveguides for multiple line S-parameter extraction. See Fig. 1 and Fig. 3 in [14] for single-line geometry and simulation layout details. The layout shown in `S-Parameter_geometry.pdf` is for 3 lines, but default settings and the other example files in this folder use 2 lines. Lines are stacked from bottom to top, sequentially.
- `Results\te_rough_example.s4p`: Example 4-port (2 line) S-parameter file extracted using this code. This file uses `source_condition = 'TE'` to generate results, uses roughness, and does not correlate to the included pdf.
- `Resutls\tm_rough_example.s4p`: Example 4-port (2 line) S-parameter file extracted using this code. This file uses `source_condition = 'TM'` to generate results, uses roughness, and does not correlate to the included pdf.

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
