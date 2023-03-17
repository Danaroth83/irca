# IRCA
Python library for the interferometry response characterization algorithm (IRCA), containing a set of utilities for estimating the parametric spectral response of the image spectrometer on chip (ImSPOC) device.
Main functionalities:
- Modeling the optical transfer function of a Fabry-Perot interferometer
- Characterizing the spectral response of an ImSPOC interferometer
- Simulating an ImSPOC acquisition from a hyperspectral image

## Demonstration scripts

This repository provides three Jupyter notebooks associated to the main functionalities of the library:

### Model:
- Demo: `jupyter notebook ./notebooks/model.ipynb`
- Shows the effect of modifying parameters on the Fabry-Perot transfer function
- Provides an interactive tool to manually fit the curve to a reference

<img src="assets/model.png" width="50%" height="50%">

### Characterization:
- Demo: `jupyter notebook ./notebooks/characterization.ipynb`
- Loads the characterization options for the IRCA proposed method
- Estimates the parameters of the transfer function for each interferometer
- Visualizes the parameters and compares the result with the training acquisitions

<img src="assets/characterization.png" width="50%" height="50%">

### Simulator:
- Demo: `jupyter notebook ./notebooks/characterization.ipynb`
- Loads a conventional hyperspectral image and the characterization of an ImSPOC
- Simulates an acquisition obtained by an ImSPOC device with those characteristics

<img src="assets/simulator.png" width="70%" height="70%">

## Project Structure

    ├── assets                      <- Figures for this document
    │
    ├── data                        <- Data folder
    │   ├── acquisitions            <- Sample ImSPOC raw acquisitions
    │   ├── characterization        <- Characterization options and parameters
    │   ├── device                  <- Device information
    │   └── hyperspectral           <- Sample hyperspectral image
    │
    ├── notebooks                   <- Jupyter notebooks interactive demo scripts
    │   ├── characterization.ipynb  <- Characterization demo script
    │   ├── model.ipynb             <- Interferometry model demo script
    │   └── simulator.ipyng         <- ImSPOC simulator demo script
    │
    └── src                         <- Source code folder
        ├── characterization        <- Characterization module
        ├── demo                    <- Demo scripts
        ├── inversion               <- Interface scripts
        └── lib                     <- General custom library module


## Requirements

- Python v3.8+
- NumPy: https://numpy.org/
- SciPy: https://scipy.org/
- Matplotlib: https://matplotlib.org/
- Pydantic: https://docs.pydantic.dev/
- Jupyter Notebook: https://jupyter.org/

To install the requirements:
- `pip install -r requirements.txt`

## Citation
To use this code, please cite:

```
@article{picone2023irca,
author = {Picone, Daniele and Gousset, Silvere and Dalla Mura, Mauro and Yann, Ferrec and Le Coarer, Etienne},
title = {The {ImSPOC} snapshot image spectrometer: {Image} formation model and spectral characterization},
booktitle = {arXiv},
note = {Under review.},
month = mar,
year = {2023},
}
```
