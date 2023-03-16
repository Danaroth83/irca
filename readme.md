# IRCA
Python library for the ImSPOC response characterization algorithm (IRCA).
Main functionalities:
- Modeling the optical transfer function of a Fabry-Perot interferometer
- Characterizing the spectral response of an ImSPOC interferometer
- Simulating an ImSPOC acquisition from a hyperspectral image

## Demonstration scripts

This repository provides three Jupyter notebooks associated to the three main functionalities of the library:

### Characterization:
- Demo: `notebooks\characterization.ipynb`
- Loads the characterization options for the IRCA proposed method
- Estimates the parameters of the transfer function for each interferometer
- Visualizes the parameters and compares the result with the training acquisitions

### Model:
- Demo: `notebooks\model.ipynb`
- Shows the effect of modifying parameters on the Fabry-Perot transfer function
- Provides an interactive tool to manually fit the curve to a reference

### Simulator:
- Demo: `notebooks\simulator.ipynb`
- Loads a conventional hyperspectral image and the characterization of an ImSPOC
- Simulates an acquisition obtained by an ImSPOC device with those characteristics



## Project Structure

    ├── README.md                   <- This file
    ├── requirements.txt            <- Lock file of package requirements
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

To install the requirements:
- `pip install -r requirements.txt`
