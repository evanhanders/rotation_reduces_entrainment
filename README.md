# rotation_reduces_entrainment
This repository contains the files used to run the simulations presented in the article "Rotation reduces mixing of composition gradients in Jupiter and other gas giants" by J.R. Fuentes et al.
This repository is backed up on zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7950969.svg)](https://doi.org/10.5281/zenodo.7950969)

## Dependencies

The simulation scripts (rotating_simulation/rotating_entrainment.py and nonrotating_simulation/entrainment.py) can be run in parallel on a processor mesh that is a power of 2 (e.g., 1024 or 2048 cores) and relies on version 3 of the Dedalus pseudospectral framework.

See [The Dedalus Repository](https://github.com/DedalusProject/dedalus) for more information on how to install Dedalus. This code uses the Dedalus master branch and should run using commit with short-sha [29f3a59](https://github.com/DedalusProject/dedalus/commit/29f3a59c5ee7cbb7be5d846e35f0c514ac032af6).

Most of the plotting scripts rely on Evan Anders' pip-installable [plotpal](https://github.com/evanhanders/plotpal) repository, and should run using version 1.0.0.

## Running at different parameters

To run a simulation at a different parameter (e.g., coefficient resolution or Rayleigh number) than those used in the paper, simply change the corresponding parameter in the 'control_parameters.py' file in the corresponding run directory.
