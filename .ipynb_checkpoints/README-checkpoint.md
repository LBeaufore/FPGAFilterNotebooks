# FPGAFilterNotebooks
These notebooks simulate and interact with the PUEO FPGA implemented filters. There is some bloat as this is my personal workspace. For non-Lucas users, start with 001_BiquadDebug.ipynb. This notebook should only use other Python modules and files stored in 001_files.

## 001_BiquadDebug.ipynb
This notebook is for testing the behavior of the Biquad in the FPGA. It is by far most legible in JupyterLab or Jupyter 7. Open the table of contents. The core simulation code is stored in python modlues imported at the top, the notebook is largely for running these simluations and basic analysis.

### Biquad Coeffs
These can be precalculated, and then read in from a file.

### Individual responses
Plotting responses to individual inputs

### Frequency response using gaussians
This uses Prof. Jim Beatty's suggested gaussian noise input to check a frequency response of a system. This includes some active develoment investigating the effect of pole-zero cancellation on passband ripple.

### Verilog Testing
This compares the python simulations to outputs from running a full simulation of the biquad as written in Verilog. This requires specific simulations in verilog that uses edited submodules (which can be a pain in our workflow). Just in case someone wants to recreate my simulations instead of using the existing output, I have included the relevant changed files in this repository under the verilog/ directory.

## Simulation Modules
### iir_biquad.py
This is nearly directly adapted from Dr. Patrick Allison's simulation. It mimics the PUEO biquad 8-fold pipelined, two clock clustered look ahead using floating point.

### IIRSim.py
This is a simulation of PUEO's Biquad, similar to iir_biquad.py, but includes tools to do all the calculations as fixed point numbers, exactly mimicing the FPGA implementation.

### FPGATestIO.py
Contains functions to generate ADC simulated readings for tones, pulses, and gaussian noise with hanning windows. Also contains the function for reading in the samples from files. The recommended use case is to only write out these files once and then read them in for analysis (some already exist in this repository).

### FrequencyResponse.py
This largely contains old and out of date functions for plotting. 


