# IRCA

[![DOI](https://zenodo.org/badge/614794669.svg)](https://zenodo.org/badge/latestdoi/614794669)

Python library for the interferometer response characterization algorithm (IRCA), containing a set of utilities for estimating the parametric transmittance response of Fabry-Perot interferometers, with particular attention to multi-aperture imaging spectrometers based on that technology.
Main functionalities:
- Modeling the optical trasmittance response of a Fabry-Perot interferometer
- Characterizing that transmittance response their sampled version
- Simulating the acquisition of a multi-aperture interferometric imaging spectrometer from a hyperspectral image

## Demonstration scripts

This repository provides three Jupyter notebooks associated to the main functionalities of the library:

### Model:
- Demo: `jupyter notebook ./notebooks/model.ipynb`
- Shows the effect of modifying parameters on the Fabry-Perot transmittance response
- Provides an interactive tool to manually fit the curve to a reference

<img src="data/assets/model.png" width="50%" height="50%">

### Characterization:
- Demo: `jupyter notebook ./notebooks/characterization.ipynb`
- Loads the characterization options for the IRCA proposed method
- Estimates the parameters of the transfer function for each interferometer
- Visualizes the parameters and compares the result with the training acquisitions

<img src="data/assets/characterization.png" width="50%" height="50%">

### Simulator:
- Demo: `jupyter notebook ./notebooks/characterization.ipynb`
- Loads a conventional hyperspectral image and the characterization of the inteferometric device
- Simulates an acquisition obtained by a device with those characteristics

<img src="data/assets/simulator.png" width="85%" height="85%">

## Project Structure

    ├── data                        <- Data folder
    │   ├── acquisitions            <- Sample raw acquisitions from Image SPectrometer On Chip (ImSPOC)
    │   ├── assets                  <- Web assets for this document
    │   ├── characterization        <- Characterization options and parameters
    │   ├── device                  <- Device information
    │   └── hyperspectral           <- Sample hyperspectral image
    │
    ├── notebooks                   <- Jupyter notebooks interactive demo scripts
    │   ├── characterization.ipynb  <- Characterization demo script
    │   ├── model.ipynb             <- Interferometry model demo script
    │   └── simulator.ipyng         <- Simulator demo script
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
      title={Interferometer response characterization algorithm for multi-aperture Fabry-Perot imaging spectrometers}, 
      author={Picone, Daniele and Gousset, Silv\`ere and Dalla Mura, Mauro and Ferrec, Yann and Le Coarer, Etienne},
      booktitle={arXiv},
      year={2023},
      month=mar,
      eprint={2303.14076},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      doi={https://doi.org/10.48550/arXiv.2303.14076},
      note={Under review.},
}
```
