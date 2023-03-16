from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

from src.interface.input import ImspocOptions
from src.characterization.preprocessing import Characterization
from src.interface.imspoc_simulator import HyperspectralImage


def main():
    fov = 10  # in degrees
    characterization_folder = "characterization/imspoc_uv_2"
    device_folder = "device/imspoc_uv_2.json"
    rgb = (70, 53, 19)

    data_folder = Path(__file__).resolve().parents[2] / "data"
    device = ImspocOptions.parse_file(data_folder / device_folder)
    hyperspectral = HyperspectralImage.load(data_folder / "hyperspectral/color_checker")
    hyperspectral.visualize(rgb=rgb)

    characterization = Characterization.load(
        data_folder / characterization_folder
    )
    imspoc = hyperspectral.to_imspoc(
        device=device, characterization=characterization, fov=fov
    )
    imspoc.visualize()
    plt.show()


if __name__ == "__main__":
    main()
