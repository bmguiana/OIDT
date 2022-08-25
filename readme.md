# Optical Interconnect Designer Tool (OIDT)

### Author: Brian Guiana
### Created: 10 December 2021
### Updated: 25 August 2022

## Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, at the Applied/Computational Electromagnetics and Signal/Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

## Project Description
The OIDT is a computational electromagnetic program. It is split into two portions (1) the Finite-Difference Time-Domain (FDTD) and scattering parameter extraction portion, both of which are located in the `.\FDTD` folder and (2) the equivalent circuit synthesis and passivity enforcement portion located in the `.\SROPEE` folder. A more detailed description is contained within the corresponding folders for each portion. Instructions for running the program from start to finish are in the `.\Integration_FDTD+SROPEE`.

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

[16] Rasul Choupanzadeh "SROPEE," GitHub, 28 June 2022, Accessed: 25 August 2022,
	Online, Available: https://github.com/RasulChoupanzadeh/SROPEE
```

