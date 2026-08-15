"""Skewed Gaussian spectrum generators."""

from __future__ import annotations

from typing import Any

import numpy as np
from lasim.spectra import skewed_gaussian


CreationParams = dict[str | int, Any]

class SkewedGaussianGenerator:
    """Generate normalised skewed Gaussian spectra over parameter ranges."""

    def __init__(
        self,
        TARGET_ENERGIES: np.ndarray,
        FULL_ENERGIES: np.ndarray,
        mu_lim: tuple[float, float],
        sg_lim: tuple[float, float],
        alpha_lim: tuple[float, float],
    ) -> None:
        """Initialise the generator.

        Args:
            TARGET_ENERGIES: Target energy grid used by downstream consumers.
            FULL_ENERGIES: Energy grid used to compute spectra.
            mu_lim: Inclusive lower and upper bounds for the mean parameter.
            sg_lim: Inclusive lower and upper bounds for the sigma parameter.
            alpha_lim: Inclusive lower and upper bounds for the skew parameter.
        """
        self.TARGET_ENERGIES = TARGET_ENERGIES
        self.FULL_ENERGIES = FULL_ENERGIES
        self.mu_lim = mu_lim
        self.sg_lim = sg_lim
        self.alpha_lim = alpha_lim

    def gridpoint(
        self,
        gridsize: int | tuple[int, int, int],
        shuffle: bool = False,
        random_state: int = -1,
        rounder: int = 4,
    ) -> tuple[np.ndarray, CreationParams]:
        """Generate spectra on a linearly spaced parameter grid.

        Args:
            gridsize: Number of samples per axis, or a 3-tuple of axis sizes.
            shuffle: Whether to randomly reorder the generated spectra.
            random_state: Seed used when ``shuffle`` is enabled.
            rounder: Decimal precision applied to sampled parameters.

        Returns:
            A tuple containing the spectra array and metadata for each sample.
        """
        if type(gridsize) is int:
            gridsize = (gridsize, gridsize, gridsize)
        mus = np.linspace(*self.mu_lim, gridsize[0])
        sigmas = np.linspace(*self.sg_lim, gridsize[1])
        alphas = np.linspace(*self.alpha_lim, gridsize[2])
        mus = np.round(mus, rounder)
        sigmas = np.round(sigmas, rounder)
        alphas = np.round(alphas, rounder)
        used_mus, used_sigmas, used_alphas = [], [], []
        spectra = []
        for mu in mus:
            for sigma in sigmas:
                for alpha in alphas:
                    skewed_gaussian_spectrum = skewed_gaussian(self.FULL_ENERGIES, mu, sigma, alpha, 1.0)
                    skewed_gaussian_spectrum /= skewed_gaussian_spectrum.sum()
                    spectra.append(skewed_gaussian_spectrum)
                    used_mus.append(mu)
                    used_sigmas.append(sigma)
                    used_alphas.append(alpha)

        if shuffle:
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            mus = np.array(used_mus)[shuffled_indices]
            sigmas = np.array(used_sigmas)[shuffled_indices]
            alphas = np.array(used_alphas)[shuffled_indices]

        creation_params = {
            "mu": mus,
            "sigma": sigmas,
            "alpha": alphas
        }

        for i in range(len(spectra)):
            creation_params[i] = {"mu": mus[i], "sigma": sigmas[i], "alpha": alphas[i]}

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
        alphas = np.arange(*self.alpha_lim, step_size)
        spectra = []
        for mu in mus:
            for sigma in sigmas:
                for alpha in alphas:
                    skewed_gaussian_spectrum = skewed_gaussian(self.FULL_ENERGIES, mu, sigma, alpha, 1.0)
                    skewed_gaussian_spectrum /= skewed_gaussian_spectrum.sum()
                    spectra.append(skewed_gaussian_spectrum)

        if shuffle:
            np.random.seed(random_state) if random_state >= 0 else None
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            mus = np.array(mus)[shuffled_indices]
            sigmas = np.array(sigmas)[shuffled_indices]

        creation_params = {
            "mu": mus,
            "sigma": sigmas,
            "alpha": alphas
        }

        for i in range(len(spectra)):
            creation_params[i] = {"mu": mus[i], "sigma": sigmas[i], "alpha": alphas[i]}

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
        alphas = np.random.uniform(*self.alpha_lim, datapoints)
        spectra = []
        for mu, sigma, alpha in zip(mus, sigmas, alphas):
            skewed_gaussian_spectrum = skewed_gaussian(self.FULL_ENERGIES, mu, sigma, alpha, 1.0)
            skewed_gaussian_spectrum /= skewed_gaussian_spectrum.sum()
            spectra.append(skewed_gaussian_spectrum)

        if shuffle:
            shuffled_indices = np.random.permutation(len(spectra))
            spectra = np.array(spectra)[shuffled_indices]
            mus = np.array(mus)[shuffled_indices]
            sigmas = np.array(sigmas)[shuffled_indices]
            alphas = np.array(alphas)[shuffled_indices]

        creation_params = {
            "mu": mus,
            "sigma": sigmas,
            "alpha": alphas
        }

        for i in range(len(spectra)):
            creation_params[i] = {"mu": mus[i], "sigma": sigmas[i], "alpha": alphas[i]}

        return np.array(spectra), creation_params