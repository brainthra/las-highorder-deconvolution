"""Boltzmann spectrum generators."""

from __future__ import annotations

from typing import Any

import numpy as np
from lasim.spectra import boltzmann


CreationParams = dict[str | int, Any]

class BoltzmannGenerator:
    """Generate normalised Boltzmann spectra over temperature ranges."""

    def __init__(
        self,
        TARGET_ENERGIES: np.ndarray,
        FULL_ENERGIES: np.ndarray,
        temp_lim: tuple[float, float],
    ) -> None:
        """Initialise the generator.

        Args:
            TARGET_ENERGIES: Target energy grid used by downstream consumers.
            FULL_ENERGIES: Energy grid used to compute spectra.
            temp_lim: Inclusive lower and upper bounds for temperature sampling.
        """
        self.TARGET_ENERGIES = TARGET_ENERGIES
        self.FULL_ENERGIES = FULL_ENERGIES
        self.temp_lim = temp_lim

    def gridpoint(
        self,
        gridsize: int,
        shuffle: bool = False,
        random_state: int = -1,
        rounder: int = 4,
    ) -> tuple[np.ndarray, CreationParams]:
        """Generate spectra on a linearly spaced temperature grid.

        Args:
            gridsize: Number of temperature points to sample.
            shuffle: Whether to randomly reorder the generated spectra.
            random_state: Seed used when ``shuffle`` is enabled.
            rounder: Decimal precision applied to the sampled temperatures.

        Returns:
            A tuple containing the spectra array and metadata for each sample.
        """
        temperatures = np.linspace(*self.temp_lim, gridsize)
        temperatures = np.round(temperatures, rounder)
        spectra = []
        for T in temperatures:
            boltz_spectrum = boltzmann(self.FULL_ENERGIES, float(T), 1.0)
            boltz_spectrum /= boltz_spectrum.sum()
            spectra.append(boltz_spectrum)

        if shuffle:
            np.random.seed(random_state) if random_state >= 0 else None
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            temperatures = np.array(temperatures)[shuffled_indices]

        creation_params = {
            "temps": temperatures
        }

        for i in range(len(spectra)):
            creation_params[i] = {"temps": temperatures[i]}

        return np.array(spectra), creation_params

    def gridstep(
        self,
        step_size: float,
        shuffle: bool = False,
        random_state: int = -1,
    ) -> tuple[np.ndarray, CreationParams]:
        """Generate spectra on a temperature grid with a fixed step.

        Args:
            step_size: Distance between consecutive temperature samples.
            shuffle: Whether to randomly reorder the generated spectra.
            random_state: Seed used when ``shuffle`` is enabled.

        Returns:
            A tuple containing the spectra array and metadata for each sample.
        """
        temperatures = np.arange(self.temp_lim[0], self.temp_lim[1], step_size)
        spectra = []
        for T in temperatures:
            boltz_spectrum = boltzmann(self.FULL_ENERGIES, float(T), 1.0)
            boltz_spectrum /= boltz_spectrum.sum()
            spectra.append(boltz_spectrum)

        if shuffle:
            np.random.seed(random_state) if random_state >= 0 else None
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            temperatures = np.array(temperatures)[shuffled_indices]

        creation_params = {
            "temps": temperatures
        }

        for i in range(len(spectra)):
            creation_params[i] = {"temps": temperatures[i]}

        return np.array(spectra), creation_params

    def generate(
        self,
        datapoints: int = 1000,
        random_state: int = -1,
        shuffle: bool = True,
    ) -> tuple[np.ndarray, CreationParams]:
        """Generate spectra from a uniform temperature distribution.

        Args:
            datapoints: Number of spectra to generate.
            random_state: Seed used for the random temperature samples.
            shuffle: Whether to randomly reorder the generated spectra.

        Returns:
            A tuple containing the spectra array and metadata for each sample.
        """
        np.random.seed(random_state) if random_state >= 0 else None
        temperatures = np.random.uniform(*self.temp_lim, datapoints)
        spectra = []
        for T in temperatures:
            boltz_spectrum = boltzmann(self.FULL_ENERGIES, T, 1.0)
            boltz_spectrum /= boltz_spectrum.sum()
            spectra.append(boltz_spectrum)

        if shuffle:
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            temperatures = np.array(temperatures)[shuffled_indices]

        creation_params = {
            "temps": temperatures
        }

        for i in range(len(spectra)):
            creation_params[i] = {"temps": temperatures[i]}

        return np.array(spectra), creation_params
    
    def igenerate(
        self,
        datapoints: int = 1000,
        random_state: int = -1,
        shuffle: bool = True,
    ) -> tuple[np.ndarray, CreationParams]:
        """Generate spectra from a uniform inverse-temperature distribution.

        Args:
            datapoints: Number of spectra to generate.
            random_state: Seed used for the random temperature samples.
            shuffle: Whether to randomly reorder the generated spectra.

        Returns:
            A tuple containing the spectra array and metadata for each sample.
        """
        np.random.seed(random_state) if random_state >= 0 else None
        temperatures = 1 / np.random.uniform(1/self.temp_lim[0], 1/self.temp_lim[1], datapoints)
        spectra = []
        for T in temperatures:
            boltz_spectrum = boltzmann(self.FULL_ENERGIES, T, 1.0)
            boltz_spectrum /= boltz_spectrum.sum()
            spectra.append(boltz_spectrum)

        if shuffle:
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            temperatures = np.array(temperatures)[shuffled_indices]

        creation_params = {
            "temps": temperatures
        }

        for i in range(len(spectra)):
            creation_params[i] = {"temps": temperatures[i]}

        return np.array(spectra), creation_params