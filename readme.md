# Optical Interconnect Designer Tool (OIDT)

### Author: Brian Guiana
### Created: 10 December 2021
### Updated: 1 July 2022

## Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, in the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

## Project Description
The computational electromagnetics (CEM) portion of the OIDT is based on the method of Finite-Difference Time-Domain (FDTD) for numerically solving electromagnetic fields with the Python language. The current version allows users to choose between CPU and GPU parallelization to enhance the speed at which numerical experiments can be completed. The code contained herein is capable of simulating fully 3D structures as well as the 2D transverse electric (TE) and transverse magnetic (TM) modes, each in a cartesian coordinate system. The OIDT is optimized for simulation of dielectric waveguide structures operating in the terahertz (THz) regime, but simulation of additional related structures is possible. Each code body is validated using the surface wave impedance as measured _below_ the waveguide structure, where _below_ is defined as along the negative y-direction. This is currently the limit of the 3D code body on CPU, but both modes of the 2D CPU code body are additionally capable of extracting scattering parameters (S-parameters), simulating random surface roughness, and calculating the scattering loss (alpha) resulting from that roughness. All GPU code bodies are capable of calculating alpha in addition to wave impedance. Examples of each of the primary functions are demonstrated using sample code.

The circuit synthesis portion of the OIDT was authored by Rasul Choupanzadeh and modified to fit with the CEM portion of the OIDT. The currently implemented version converts touchstone format scattering parameters into an equivalent SPICE circuit [16].

## Simulation Geometry
The geometry for the 3D simulations is shown in geometry_3d.pdf, and the geometry for 2D simulations is shown in geometry_2d.pdf. In the 3D code body, the electric field intensity (E-field) and magnetic field intensity (H-field) both have components in the x-, y-, and z-directions (Ex, Ey, Ez, Hx, Hy, Hz). In the 2D code body, the assumed geometry forces the partial derivative w.r.t. the z-direction to zero, so variability is limited to only the x- and y-directions. The TE mode has field components Ex, Ey, and Hz. The TM mode has field components Ez, Hx, and Hy. In the 3D body the fields are excited by the Ex field component on CPU and either the Ey or Hy field component on GPU, and wave propagation is along the z-direction. In the 2D body the fields are excited by whichever field component is in the z-direction (Ez for TM and Hz for TE) and wave propagation is along the x-direction.

The fundamental structure of the simulations does not change dramatically between 2D and 3D simulation. These have been clearly labelled in geometry_2d.pdf, but several labels have been omitted from geometry_3d.pdf. On the exterior of all simulations is a perfect electric conductor (PEC) bounding box. It is assumed that outside the simulation space the PEC region extends toward infinity in all directions. Within the PEC box there exists a small region of convolution perfectly matched layer (CPML). The remainder of the simulation space can be divided into core region, cladding region, and test region. The core region is bound by the core/cladding permittivity interface, and it extends into the CPML region fully, making contact with the PEC bounding box. This effectively simulates infinite extent of the general shape of the core region in the outward traveling directions. The cladding region exists everytwhere not included in the core. The test rgion refers to a portion of waveguide length that is dedicated to nonideal properties. This includes things like surface roughness. The effective length of the test region is used to find alpha as a per-unit-length loss characteristic. Field values are recorded along the full line at port 1 and port 2 during S-parameter simulations.
The 2D simulation setup can be thought of as a center cross-section of the 3D simulation setup. There is a coordinate system mapping that takes place when translating between a 2D cross-section of the default 3D model and the default 2D model. This mapping only affects the name of the directions and not the inherent orientation.

## Included Files
The code body is separated into 2D TE mode, 2D TM mode, and 3D bodies. A separate readme is included for each of these detailing intended use, expected results, operation instructions, and additional details specific to those sets of code.

S-parameter simulations also include a Keysight ADS demo file showing the correlation between SPICE netlist and touchstone format S-Parameters. This correlation is currently performed only in the frequency domain. The current implementation requires that frequency domain simulation in ADS be done at the same frequency samples used to generate the SPICE netlist. These are stored as *.7zads archives in the corresponding folders. ADS simulation has been completed and results are shown in fdtd_circuit.dds.

## General Program Functionality
Simulation parameters are adjusted in the files names **fdtd_config.py**. Some notes on these parameters are listed below. It is not advised to alter any of the other parameters. In the example files included, some of the features have been removed to improve the simulation flow. The version used for calculate alpha is always a complete model.
- `eps_rel_bg`: The background relative permittivity. This is a unitless quantity with a default of 2.25, corresponding to high-quality silicon dioxide. This is also referred to as the _cladding permittivity_. Several simulation parameters are normalized using this quantity.
- `eps_rel_fg`: The foreground relative permittivity. This is a unitless quantity with a default value of 12.25, corresponding to high quality intrinsic silicon. This is also referred to as the _core permittivity_.
- `sgl_wg_length`: Total desired physical length of the waveguide. This value is along the x-direction in 2D simulations, and along the z-direction in 3D simulations.
- `sgl_wg_height`: **3D simulations only** Total desired waveguide height. This value is typically smaller than sgl_wg_width.
- `sgl_wg_width`: Total desired waveguide width.
- `port_length`: Field excitation mode settling range. This parameter is the setup length needed for the field excitation to settle into a guided mode after excitation at the source point. It is **_subtractive_** from sgl_wg_length, i.e. sgl_wg_length should be larger than 2 times port_length for proper simulation. The default value is 5 micrometers in 2D sims, but adequate results may also be possible with smaller values, depending on the physical setup. This parameter has been split into port1_length and port2_length for TE mode on GPU.
- `rough_std`: Expected standard deviation of a desired surface roughness profile.
- `rough_acl`: Expected autocorrelation length of a desired surface roughness profile.
- `rough_toggle`: Switch for simulations involving surface roughness. When rough_toggle is **True** rough_std and rough_acl should be non-zero to ensure convergence of the profile generation process. When rough_toggle is **False** the simulation will be completed with _smooth_ sidewalls, and there will be no recorded value for alpha. **_NOTE:_** The profile generation process requires a storage folder titled **rough_profiles** to save the profile. The program will crash if this folder does not exist or if the folder has any different name.
- `output_dir`: This string should match the folder in which outputs are stored. The default value is **Results**, but this can and should be changed to match the desired output folder name. **_NOTE:_** The program will crash if this folder does not exist or if the folder has any different name.
- `prof_dir`: **S-Parameter Simulations Only** This string should match the folder in which rough profiles are stored. The default value is **rough_profiles**, but this can be changed and the program will automatically place rough profiles there.
- `output files`: This category is not recommended for change, but can be changed to match whatever output is desired. Additional changes must be made in **fdtd_main.py** for all variables involved with this output change.
- `roughness_profile`: This is the name attached to the generated rough profile as well as the end of output files from the corresponding simulation. The naming convention used by default is `_sXX_lcYY_rZZ`, where XX is the value for rough_std in nanometers, YY is the value for rough_acl in nanometers, and ZZ is the profile number (this can be any value).
- `source_type`: Shape of the field excitation. Options include Gaussian pulse, frequency modulated Gaussian pulse, and time-harmonic signal. Both pulse excitations have a build and decay, but the time-harmonic excitation only has a build and then remains with a constant sinusoidal signal with the specified amplitude for the remainder of the simulation.
- `wave_packet_bw`: Frequency modulated Gaussian pulse bandwidth. This defines the frequency domain percentage bandwidth. The default value is 80%.
- `gauss_pulse_deg`: Gaussian pulse decay at f0. This defines the frequency domain amplitude in dB. The default value is -1 dB, e.g. if the source amplitude is 1, then the decay of -1 dB at f0 is approximately 0.9.
- `f0`: Fundamental frequency. This value is used to modulate the field excitation shape. It is also used to determine the maximum time-step.
- `harmonics`: Maximum harmonics. This value defines the number of harmonics above the fundamental frequency will be simulated. The maximum frequency corresponds to the minimum wavelength, which will in turn determine the spatial and temporal discretization.
- `num_flights`: Number of flights along the length. This determines how many time steps are simulated. 1 flight time corresponds to the peak value of the field excitation traveling 1 sgl_wg_length distance. This should typically be slightly larger than 1.0. The default value is 1.2.
- `points_per_wl`: Points per minimum wavelength. This additionally sets the spatial and temporal resolution by setting how many time samples are included in each minimum wavelength.

In addition to these manually set parameters, the boundary conditions are automatically set using physical waveguide parameters.

Functions used directly in the FDTD update scheme are defined in **fdtd_funcs.py**. These functions are not needed outside of this context. Auxiliary functions are defined in **aux_funcs.py**.

## Python build
This project was created with the Anaconda Python distribution, and the following packages:
- Python 3.7.11
- IPython 7.26.0
- Numpy 1.20.3
- Matplotlib 3.4.2
- Numba 0.53.1
- Pyspeckle 0.3.1
- Psutil 5.8.0

## Licensing
This project is licensed under GNU GPL v.3.0. See the license file for details.

## References
```
[1] A. Zadehgol, "SHF: SMALL: A Novel Algorithm for Automated Synthesis of
	Passive, Causal, and Stable Models for Optical Interconnects,"
	National Science Foundation, Award #1816542. Jun. 22, 2018.

[2] A. Taflove and S. Hagness, "Computational Electrodynamics: The
	Finite-Difference Time-Domain Method," 3rd ed. Norwood, MA, USA:
	Artech House, 2005.

[3] J. Simpson, "FDTD/fdtd3D_CPML.f90," GitHub. Jun. 17, 2010. [Online]
	Available: https://github.com/cvarin/FDTD/blob/master/Taflove/fdtd3D_CPML.f90
	[Accessed May 18, 2021].

[4] J.P.R. Lacey and F.P. Payne, "Radiation loss from planar
	waveguides with random wall imperfections," In IEE Proceedings,
	vol. 137, Pt. J, No. 4, pp. 282-288, 1990.

[5] F.P. Payne and J.P.R. Lacey, "A theoretical analysis of
	scattering loss from planar optical waveguides," Optical and
	Quantum Electronics, vol. 26, pp. 977-986, 1994.

[6] A. Zadehgol, "Complex s-Plane Modeling and 2D Characterization of the
	Stochastic Scattering Loss in Symmetric Dielectric Slab Waveguides
	Exhibiting Ergodic Surface-Roughness With an Exponential Autocorrelation
	Function," IEEE Access, vol. 9, pp. 92326-92344, 2021.

[7] C. Balanis, "Advanced Engineering Electromagnetics," 1st ed.
	Hoboken, NJ, USA, John Wiley & Sons, 1989.
	
[8] B. Guiana and A. Zadehgol, "Characterizing THz Scattering Loss in
        Nano-scale SOI Waveguides Exhibiting Stochastic Surface Roughness
        with Exponential Autocorrelation," Electronics, vol. 11, no. 3, 2022.
        [Online]. Available: https://www.mdpi.com/2079-9292/11/3/307

[9] B. Guiana and A. Zadehgol, "FDTD Simulation of Stochastic Scattering Loss Due to Surface
        Roughness in Optical Interconnects," in 2022 United States National
        Committee of URSI National Radio Science Meeting (USNC-URSI NRSM), Jan 2022,
        pp. 1--2.	

[10] B. Guiana and A. Zadehgol, "Stochastic FDTD Modeling of Propagation Loss 
	due to Random Surface Roughness in Sidewalls of Optical Interconnects," 
	United States National Committee URSI National Radio Science Meeting 
	(USNC-URSI NRSM), pp. 266-267, 2021.

[11] B. Guiana and A. Zadehgol, "S-parameter Extraction Methodology in FDTD for Nano-scale
        Optical Interconnects," in 15th International Conference on Advanced
        Technologies, Systems and Services in Telecommunications (TELSIKS)}, October 20-22
        2021, pp. 1--4.

[12] B. Guiana and A. Zadehgol, "A 1D Gaussian Function for Efficient
	Generation of Plane Waves in 1D, 2D, and 3D FDTD," 2020 IEEE
	International Symposium on Antennas and Propagation and North
	American Radio Science Meeting, pp. 2009-2010, 2020.

[13] D. M. Sullivan, Electromagnetic Simulation Using the FDTD
        Method, 2nd~ed. Piscataway, NJ, USA: Wiley-IEEE Press, 2013.

[14] B. Guiana and A. Zadehgol, "Analytical Models of Stochastic Scattering 
        Loss for TM and TE Modes in Dielectric Waveguides Exhibiting Exponential Surface 
        Roughness, and a Validation Methodology in 3D FDTD," 2022;
        TechRxiv Preprint Available: https://doi.org/10.36227/techrxiv.19799737.v1 

[15] P. Sypek, A. Dziekonski, and M. Mrozowski, "How to Render FDTD Computations More 
	Effective Using a Graphics Accelerator," IEEE Transactions on Magnetics, 
	Vol. 45, No. 3, Mar. 2009, pp. 1324--1327, DOI: 10.1109/TMAG.2009.2012614

[16] Rasul Choupanzadeh "SROPEE," GitHub, 28 June 2022, Accessed: 1 July 2022,
	Online, Available: https://github.com/RasulChoupanzadeh/SROPEE
```
