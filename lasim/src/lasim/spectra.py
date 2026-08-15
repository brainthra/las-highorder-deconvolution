"""
Incident energy spectrum generation.

This module provides functions for generating common incident energy spectra, including Boltzmann, Gaussian and skewed-Gaussian distributions.

Two conventions are provided for distributions:

- ``N`` as the flux.
- ``A`` as the amplitude.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import skewnorm


FloatArray: TypeAlias = NDArray[np.float64]


# region Boltzmann spectra
# ---------------------------------------------------------------------------


def boltzmann(E: ArrayLike, T: float, N: float) -> FloatArray:
    """Calculate Boltzmann spectrum parameterised by flux.

    Args:
        E: Array of energy values.
        T: Temperature.
        N: Flux

    Returns:
        Spectrum over ``E``.
    """
    energy = np.asarray(E, dtype=float)
    return (N / T) * np.exp(-energy / T)


def boltzmann_peak(E: ArrayLike, T: float, A: float) -> FloatArray:
    """Calculate a peak-normalised Boltzmann spectrum.

    Args:
        E: Array of energy values.
        T: Temperature.
        A: Spectrum amplitude at zero energy.

    Returns:
        Spectrum over ``E``.
    """
    energy = np.asarray(E, dtype=float)
    return A * np.exp(-energy / T)


boltz = boltzmann
"""Alias for :func:`boltzmann`."""

# endregion
# ---------------------------------------------------------------------------

#  region Gaussian spectra
# ---------------------------------------------------------------------------


def gaussian(
    E: ArrayLike,
    mu: float,
    sigma: float,
    N: float,
) -> FloatArray:
    """Calculate Gaussian spectrum parameterised by flux.

    Args:
        E: Array of energy values.
        mu: Mean, corresponding to the centre of the Gaussian.
        sigma: Standard deviation of the Gaussian. Must be positive.
        N: Flux.

    Returns:
        Spectrum over ``E``.
    """
    energy = np.asarray(E, dtype=float)

    normalisation = N / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -((energy - mu) ** 2) / (2.0 * sigma**2)

    return normalisation * np.exp(exponent)


def gaussian_peak(
    E: ArrayLike,
    mu: float,
    sigma: float,
    A: float,
) -> FloatArray:
    """Calculate a peak-normalised Gaussian spectrum.

    Args:
        E: Array of energy values.
        mu: Mean, corresponding to the location of the peak.
        sigma: Standard deviation of the Gaussian. Must be positive.
        A: Peak amplitude.

    Returns:
        Spectrum over ``E``.
    """
    energy = np.asarray(E, dtype=float)
    exponent = -((energy - mu) ** 2) / (2.0 * sigma**2)

    return A * np.exp(exponent)


gauss = gaussian
"""Alias for :func:`gaussian`."""

# endregion
# ---------------------------------------------------------------------------

# region Skewed Gaussian spectra
# ---------------------------------------------------------------------------


def skewed_gaussian(
    E: ArrayLike,
    mu: float,
    sigma: float,
    alpha: float,
    N: float,
) -> FloatArray:
    """Calculate an area-normalised skewed Gaussian spectrum.

    Args:
        E: Array of energy values.
        mu: Location parameter of the skew-normal distribution.
        sigma: Scale parameter of the distribution. Must be positive.
        alpha: Skewness parameter.
        N: Integrated spectrum normalisation.

    Returns:
        Spectrum over ``E``.
    """
    energy = np.asarray(E, dtype=float)

    return N * skewnorm.pdf(
        energy,
        alpha,
        loc=mu,
        scale=sigma,
    )


def skewed_gaussian_peak(
    E: ArrayLike,
    mu: float,
    sigma: float,
    alpha: float,
    A: float,
) -> FloatArray:
    """Calculate a peak-normalised skewed Gaussian spectrum.

    Args:
        E: Array of energy values.
        mu: Location parameter of the skew-normal distribution.
        sigma: Scale parameter of the distribution. Must be positive.
        alpha: Skewness parameter. Positive values produce right skew and
            negative values produce left skew.
        A: Desired peak amplitude.

    Returns:
        Spectrum over ``E`` with a maximum amplitude of approximately ``A``.
    """
    energy = np.asarray(E, dtype=float)

    pdf_values = skewnorm.pdf(
        energy,
        alpha,
        loc=mu,
        scale=sigma,
    )

    dense_energy = np.linspace(
        mu - 10.0 * sigma,
        mu + 10.0 * sigma,
        10_000,
    )

    pdf_max = np.max(
        skewnorm.pdf(
            dense_energy,
            alpha,
            loc=mu,
            scale=sigma,
        )
    )

    return A * pdf_values / pdf_max


skew = skewed_gaussian
"""Alias for :func:`skewed_gaussian`."""

skewgauss = skewed_gaussian
"""Alias for :func:`skewed_gaussian`."""
# endregion

__all__ = [
    "boltzmann",
    "boltzmann_peak",
    "boltz",
    "gaussian",
    "gaussian_peak",
    "gauss",
    "skewed_gaussian",
    "skewed_gaussian_peak",
    "skew",
    "skewgauss",
]