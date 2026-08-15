"""Gaussian spectrum generators."""

from __future__ import annotations

from typing import Any

import numpy as np
from lasim.spectra import gaussian


CreationParams = dict[str | int, Any]

class GaussianGenerator:
    """Generate normalised Gaussian spectra over parameter ranges."""

    def __init__(
        self,
        TARGET_ENERGIES: np.ndarray,
        FULL_ENERGIES: np.ndarray,
        mu_lim: tuple[float, float],
        sg_lim: tuple[float, float],
    ) -> None:
        """Initialise the generator.

        Args:
            TARGET_ENERGIES: Target energy grid used by downstream consumers.
            FULL_ENERGIES: Energy grid used to compute spectra.
            mu_lim: Inclusive lower and upper bounds for the mean parameter.
            sg_lim: Inclusive lower and upper bounds for the sigma parameter.
        """
        self.TARGET_ENERGIES = TARGET_ENERGIES
        self.FULL_ENERGIES = FULL_ENERGIES
        self.mu_lim = mu_lim
        self.sg_lim = sg_lim

    def gridpoint(
        self,
        gridsize: int,
        shuffle: bool = False,
        random_state: int = -1,
        rounder: int = 4,
    ) -> tuple[np.ndarray, CreationParams]:
        """Generate spectra on a linearly spaced parameter grid.

        Args:
            gridsize: Number of samples to draw per parameter axis.
            shuffle: Whether to randomly reorder the generated spectra.
            random_state: Seed used when ``shuffle`` is enabled.
            rounder: Decimal precision applied to sampled parameters.

        Returns:
            A tuple containing the spectra array and metadata for each sample.
        """
        mus = np.linspace(*self.mu_lim, gridsize)
        sigmas = np.linspace(*self.sg_lim, gridsize)
        mus = np.round(mus, rounder)
        sigmas = np.round(sigmas, rounder)
        used_mus, used_sigmas = [], []
        spectra = []
        for mu in mus:
            for sigma in sigmas:
                gaussian_spectrum = gaussian(self.FULL_ENERGIES, mu, sigma, 1.0)
                gaussian_spectrum /= gaussian_spectrum.sum()
                spectra.append(gaussian_spectrum)
                used_mus.append(mu)
                used_sigmas.append(sigma)

        if shuffle:
            np.random.seed(random_state) if random_state >= 0 else None
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            used_mus = np.array(used_mus)[shuffled_indices]
            used_sigmas = np.array(used_sigmas)[shuffled_indices]

        creation_params = {
            "mu": used_mus,
            "sigma": used_sigmas
        }

        for i in range(len(spectra)):
            creation_params[i] = {"mu": used_mus[i], "sigma": used_sigmas[i]}

        return np.array(spectra), creation_params

    def gridstep(
        self,
        step_size: float,
        shuffle: bool = False,
        random_state: int = -1,
    ) -> tuple[np.ndarray, CreationParams]:
        """Generate spectra on a fixed-step parameter grid.

        Args:
            step_size: Distance between consecutive samples on each axis.
            shuffle: Whether to randomly reorder the generated spectra.
            random_state: Seed used when ``shuffle`` is enabled.

        Returns:
            A tuple containing the spectra array and metadata for each sample.
        """
        mus = np.arange(*self.mu_lim, step_size)
        sigmas = np.arange(*self.sg_lim, step_size)
        spectra = []
        for mu in mus:
            for sigma in sigmas:
                gaussian_spectrum = gaussian(self.FULL_ENERGIES, mu, sigma, 1.0)
                gaussian_spectrum /= gaussian_spectrum.sum()
                spectra.append(gaussian_spectrum)

        if shuffle:
            np.random.seed(random_state) if random_state >= 0 else None
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            mus = np.array(mus)[shuffled_indices]
            sigmas = np.array(sigmas)[shuffled_indices]

        creation_params = {
            "mu": mus,
            "sigma": sigmas
        }

        for i in range(len(spectra)):
            creation_params[i] = {"mu": mus[i], "sigma": sigmas[i]}

        return np.array(spectra), creation_params

    def generate(
        self,
        datapoints: int = 1000,
        random_state: int = -1,
        shuffle: bool = True,
    ) -> tuple[np.ndarray, CreationParams]:
        """Generate spectra from uniform parameter sampling.

        Args:
            datapoints: Number of spectra to generate.
            random_state: Seed used for the random parameter samples.
            shuffle: Whether to randomly reorder the generated spectra.

        Returns:
            A tuple containing the spectra array and metadata for each sample.
        """
        np.random.seed(random_state) if random_state >= 0 else None
        mus = np.random.uniform(*self.mu_lim, datapoints)
        sigmas = np.random.uniform(*self.sg_lim, datapoints)
        spectra = []
        for mu, sigma in zip(mus, sigmas):
            gaussian_spectrum = gaussian(self.FULL_ENERGIES, mu, sigma, 1.0)
            gaussian_spectrum /= gaussian_spectrum.sum()
            spectra.append(gaussian_spectrum)

        if shuffle:
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            mus = np.array(mus)[shuffled_indices]
            sigmas = np.array(sigmas)[shuffled_indices]

        creation_params = {
            "mu": mus,
            "sigma": sigmas
        }

        for i in range(len(spectra)):
            creation_params[i] = {"mu": mus[i], "sigma": sigmas[i]}

        return np.array(spectra), creation_params