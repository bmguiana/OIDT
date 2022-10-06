# Optical Interconnect Designer Tool Visualizer

### Created On: 9 August 2022
### Updated On: 6 October 2022
### Author: B. Guiana

## Acknowledgement:
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, at the Applied/Computational Electromagnetics and Signal/Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

## Program Description:
This is a graphical interface for visualizing 2D simulation data. 

## Installation:
1. Download all files from this repository. The folder `Application Files` must be kept in the same directory as the `*.application` file and `setup.exe`.
2. Download additional animation files from the below link and place them into the `animations` folder: https://vandalsuidaho-my.sharepoint.com/:f:/g/personal/azadehgol_uidaho_edu/EnMeyGbo_WRBt2_IOM8PTnUBCPGX-uG-1tyBaKw8c8ZGww
3. Run setup.exe. There may be a security warning on first launch, but clicking on "run anyway" (this button might be hidden behind a "more info" button) will automatically install and launch the program. A second security warning may appear before the main application starts, but this can be bypassed in a similar manner.

## Usage:
### First Run:
1. Run the `*.application` file to open the program after installation. This step may be skipped if running directly after installation as `setup.exe` should launch the program automatically.
2. Click any button with a picutre. This will open a file dialog.
3. Navigate to the `animations` folder and select any `*.mp4` file. The animation will start playing in the viewer pane on the right side of the window.
4. Click the `full screen` button to enter fullscreen mode. This button will then turn into a `demo mode` button.

### Demo Mode:
1. Click `Start Demo Mode` to automatically cycle through animations. The button will then change to `In Demo Mode` and clicking on it will do nothing further.
2. Click any other button to exit demo mode and play the corresponding animation.

### Closing Methods:
In fullscreen mode, the title bar (including the minimize, maximize, and close buttons) and task bar are both removed. The following methods will exit the program from fullscreen mode.
1. With the program window highlighted, simultaneously press `Alt + F4`. This will exit the program directly is the quickest method to exit.
2. At any point simultaneously press `Alt + Tab` to deselect the window and navigate to another. This will bring up the task bar and cycle to another open window. The program can then be closed using conventional methods.
3. Simultaneously press `Command + D` (`Windows + D`) to go directly to the desktop. The program can again be closed using conventional methods.

## Warnings and Notes:
- The program was designed using Visual Basic and has been tested with Windows 10 Education and Pro Editions. Functionality with other operating systems is not guaranteed.
- The program was designed for specific display resolution: 1360x768. Any resolution larger will distort (stretch) the buttons and text fields. Any resolution smaller may make some features unviewable (off-screen).
- The animations will not display correctly if the program is running on a system with more than a single display (screen). Fault states include visual artifacting for several seconds or animations not playing.
- The corresponding animation files are 9.3 MB each, on average, so there are only 3 sample animation files included on GitHub with this release (of the total 324 files generated for this program). These are contained within the VB project folder "animations". For the complete set of animation files, go to the below link:

https://vandalsuidaho-my.sharepoint.com/:f:/g/personal/azadehgol_uidaho_edu/EnMeyGbo_WRBt2_IOM8PTnUBCPGX-uG-1tyBaKw8c8ZGww