from __future__ import annotations

import random
from typing import Any
import numpy as np

from lasim.spectra import *


SampleTuple = tuple[int, int]
CreationParams = dict[str | int, Any]

def _sample_tuples(
    max_boltz: int,
    max_gauss: int,
    num_samples: int = 1,
    max_comps: int = 100,
    mul_rat: int = 2,
    random_state: int = -1,
) -> list[SampleTuple]:
    """Sample component-count tuples with weighted probability.

    Args:
        max_boltz: Maximum number of Boltzmann components to consider.
        max_gauss: Maximum number of Gaussian components to consider.
        num_samples: Number of tuples to draw.
        max_comps: Maximum weighted component count allowed for a tuple.
        mul_rat: Exponent applied to the tuple weight.
        random_state: Seed used when sampling.

    Returns:
        A list of sampled ``(n_boltz, n_gauss)`` tuples.
    """
    tuples = [(b, g) for b in range(0, max_boltz+1) for g in range(0, max_gauss+1)]
    tuples.remove((0, 0))
    for (b, g) in tuples[:]:
        if (b*2 if b > 1 else b) + (g*4 if g > 1 else g*3) > max_comps:
            tuples.remove((b, g))

    weights = [((b * 2 if b > 1 else b) + (g * 4 if g > 1 else g * 3)) ** mul_rat
               for (b, g) in tuples]
    random.seed(random_state if random_state >=0 else None)
    return random.choices(tuples, weights=weights, k=num_samples)


class CompositeGenerator:
    """Generate composite spectra made from Boltzmann and Gaussian peaks."""

    def __init__(
        self,
        TARGET_ENERGIES: np.ndarray,
        FULL_ENERGIES: np.ndarray,
        temp_lim: tuple[float, float],
        mu_lim: tuple[float, float],
        sg_lim: tuple[float, float],
        alpha_lim: tuple[float, float],
        boltz_amp_lim: tuple[float, float],
        gauss_amp_lim: tuple[float, float],
    ) -> None:
        """Initialise the generator.

        Args:
            TARGET_ENERGIES: Target energy grid used by downstream consumers.
            FULL_ENERGIES: Energy grid used to compute spectra.
            temp_lim: Inclusive lower and upper bounds for Boltzmann temperature.
            mu_lim: Inclusive lower and upper bounds for Gaussian mean.
            sg_lim: Inclusive lower and upper bounds for Gaussian sigma.
            alpha_lim: Inclusive lower and upper bounds for Gaussian skew.
            boltz_amp_lim: Inclusive lower and upper bounds for Boltzmann amplitude.
            gauss_amp_lim: Inclusive lower and upper bounds for Gaussian amplitude.
        """
        self.TARGET_ENERGIES = TARGET_ENERGIES
        self.FULL_ENERGIES = FULL_ENERGIES
        
        self.temp_lim = temp_lim
        
        self.mu_lim = mu_lim
        self.sg_lim = sg_lim
        self.alpha_lim = alpha_lim
        
        self.boltz_amp_lim = boltz_amp_lim
        self.gauss_amp_lim = gauss_amp_lim
    
    def generate(
        self,
        datapoints: int,
        max_gauss: int,
        max_boltz: int,
        max_comps: int = 100,
        random_state: int = 0,
    ) -> tuple[np.ndarray, list[CreationParams]]:
        """Generate composite spectra from uniform parameter sampling.

        Args:
            datapoints: Number of spectra to generate.
            max_gauss: Maximum number of Gaussian components per spectrum.
            max_boltz: Maximum number of Boltzmann components per spectrum.
            max_comps: Maximum weighted component count allowed for sampling.
            random_state: Seed used for sampling and parameter generation.

        Returns:
            A tuple containing the spectra array and one parameter dictionary per sample.
        """
        spectra = []
        params = []

        samples = _sample_tuples(max_boltz, max_gauss, datapoints, max_comps=max_comps, random_state=random_state)
        np.random.seed(random_state)
        
        for (n_boltz, n_gauss) in samples:
            param = {"n_boltz": n_boltz, "n_gauss": n_gauss, "Ts": [], "mus": [], "sgs": [], "alphas": [], "boltz_amps": [], "gauss_amps": []}
            for _ in range(n_boltz):
                amplitude = np.random.uniform(self.boltz_amp_lim[0], self.boltz_amp_lim[1])
                temp = np.random.uniform(self.temp_lim[0], self.temp_lim[1])
                param["Ts"].append(temp)
                param["boltz_amps"].append(amplitude)

            for _ in range(n_gauss):
                amplitude = np.random.uniform(self.gauss_amp_lim[0], self.gauss_amp_lim[1])
                mu = np.random.uniform(self.mu_lim[0], self.mu_lim[1])
                sg = np.random.uniform(self.sg_lim[0], self.sg_lim[1])
                alpha = np.random.uniform(self.alpha_lim[0], self.alpha_lim[1])
                param["gauss_amps"].append(amplitude)
                param["mus"].append(mu)
                param["sgs"].append(sg)
                param["alphas"].append(alpha)
            
            spectrum = np.zeros_like(self.FULL_ENERGIES)
            for amplitude, temp in zip(param["boltz_amps"], param["Ts"]):
                spectrum += boltzmann_peak(self.FULL_ENERGIES, temp, amplitude)
            for amplitude, mu, sg, alpha in zip(param["gauss_amps"], param["mus"], param["sgs"], param["alphas"]):
                spectrum += skewed_gaussian_peak(self.FULL_ENERGIES, mu, sg, alpha, amplitude)

            spectrum /= np.sum(spectrum)
            spectra.append(spectrum)
            params.append(param)

        # params["n_gauss"] = [p["n_gauss"] for p in params]
        # params["n_boltz"] = [p["n_boltz"] for p in params]
        
        return np.array(spectra), params
    
    def igenerate(
        self,
        datapoints: int,
        max_gauss: int,
        max_boltz: int,
        max_comps: int = 100,
        random_state: int = 0,
    ) -> tuple[np.ndarray, list[CreationParams]]:
        """Generate composite spectra with inverse-temperature Boltzmann sampling.

        Args:
            datapoints: Number of spectra to generate.
            max_gauss: Maximum number of Gaussian components per spectrum.
            max_boltz: Maximum number of Boltzmann components per spectrum.
            max_comps: Maximum weighted component count allowed for sampling.
            random_state: Seed used for sampling and parameter generation.

        Returns:
            A tuple containing the spectra array and one parameter dictionary per sample.
        """
        spectra = []
        params = []

        samples = _sample_tuples(max_boltz, max_gauss, datapoints, max_comps=max_comps, random_state=random_state)
        np.random.seed(random_state)
        
        for (n_boltz, n_gauss) in samples:
            param = {"n_boltz": n_boltz, "n_gauss": n_gauss, "Ts": [], "mus": [], "sgs": [], "alphas": [], "boltz_amps": [], "gauss_amps": []}
            for _ in range(n_boltz):
                amplitude = np.random.uniform(self.boltz_amp_lim[0], self.boltz_amp_lim[1])
                temp = 1/np.random.uniform(1/self.temp_lim[0], 1/self.temp_lim[1])
                param["Ts"].append(temp)
                param["boltz_amps"].append(amplitude)

            for _ in range(n_gauss):
                amplitude = np.random.uniform(self.gauss_amp_lim[0], self.gauss_amp_lim[1])
                mu = np.random.uniform(self.mu_lim[0], self.mu_lim[1])
                sg = np.random.uniform(self.sg_lim[0], self.sg_lim[1])
                alpha = np.random.uniform(self.alpha_lim[0], self.alpha_lim[1])
                param["gauss_amps"].append(amplitude)
                param["mus"].append(mu)
                param["sgs"].append(sg)
                param["alphas"].append(alpha)
            
            spectrum = np.zeros_like(self.FULL_ENERGIES)
            for amplitude, temp in zip(param["boltz_amps"], param["Ts"]):
                spectrum += boltzmann_peak(self.FULL_ENERGIES, temp, amplitude)
            for amplitude, mu, sg, alpha in zip(param["gauss_amps"], param["mus"], param["sgs"], param["alphas"]):
                spectrum += skewed_gaussian_peak(self.FULL_ENERGIES, mu, sg, alpha, amplitude)

            spectrum /= np.sum(spectrum)
            spectra.append(spectrum)
            params.append(param)

        # params["n_gauss"] = [p["n_gauss"] for p in params]
        # params["n_boltz"] = [p["n_boltz"] for p in params]
        
        return np.array(spectra), params