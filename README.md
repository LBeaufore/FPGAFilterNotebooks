# FPGAFilterNotebooks
These notebooks simulate and interact with the PUEO FPGA implemented filters. There is some bloat as this is my personal workspace. For non-Lucas users, start with 001_BiquadDebug.ipynb. This notebook should only use other Python modules and files stored in 001_files.

## 001_BiquadDebug.ipynb
This notebook is for testing the behavior of the Biquad in the FPGA. It is by far most legible in JupyterLab or Jupyter 7. Open the table of contents. The core simulation code is stored in python modlues imported at the top, the notebook is largely for running these simluations and basic analysis.

### Simulation Modules
## iir_biquad.py
This is nearly directly adapted from Dr. Patrick Allison's simulation. It mimics the PUEO biquad 8-fold pipelined, two clock clustered look ahead using floating point. 
## IIRSim.py
This is a simulation of PUEO's Biquad, similar to iir_biquad.py, but includes tools to do all the calculations as fixed point numbers, exactly mimicing the FPGA implementation.
