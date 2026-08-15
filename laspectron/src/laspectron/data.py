"""Dataset utilities for Laser Absorption Spectrometer data."""

from __future__ import annotations

from types import NotImplementedType
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from lasim.system import Spectrometer
from sklearn.preprocessing import Normalizer
from .digitisers import Camera
from .scalers import Scaler

# Type alias for index inputs
IndexLike = Union[slice, Sequence[int], np.ndarray]


class LASDataset:
    def __init__(
        self,
        spectra: np.ndarray,
        design: Spectrometer,
        sample_idxs: Optional[np.ndarray] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            spectra: Spectral samples with shape ``(n_samples, n_energies)``.
            design: Spectrometer used to generate measured outputs.
            sample_idxs: Optional indices selecting the target energy bins.

        Raises:
            ValueError: If ``design`` is not provided.
        """
        if design is None:
            raise ValueError("design must be provided for LASDataset.")

        self.spectra = spectra
        self.design = design
        self.sample_idxs = (
            sample_idxs if sample_idxs is not None
            else np.arange(design.ENERGIES.shape[0])
        )
        self.target_bins = design.ENERGIES[self.sample_idxs]

        # Simulate spectrometer measurements
        self.measureds = np.zeros((self.spectra.shape[0], design.outputs))
        for i, spec in enumerate(self.spectra):
            self.measureds[i] = design.unit(spec)

    def subset(self, indices: IndexLike) -> LASSubset:
        """Create a subset view of the dataset.

        Args:
            indices: Integer indices, slices, or boolean masks selecting rows.

        Returns:
            A subset view over the selected samples.
        """
        idx = self._normalise_indices(indices)
        return LASSubset(self, idx)

    def split(
        self,
        split_ratio: list[float] = [0.6, 0.15, 0.25],
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ) -> Tuple[LASSubset, LASSubset, LASSubset]:
        """Split the dataset into train, validation, and test subsets.

        Args:
            split_ratio: Three-way split fractions or absolute sizes.
            shuffle: Whether to shuffle indices before splitting.
            random_state: Seed used when shuffling is enabled.

        Returns:
            A ``(train, validation, test)`` tuple of dataset subsets.
        """

        n = len(self)
        train_n, val_n, test_n = self._resolve_split_sizes(n, *split_ratio)

        rng = np.random.default_rng(random_state)
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        train_idx = indices[:train_n]
        val_idx = indices[train_n:train_n + val_n]
        test_idx = indices[train_n + val_n:train_n + val_n + test_n]

        return self.subset(train_idx), self.subset(val_idx), self.subset(test_idx)

    def get_optimal_flux(
        self,
        digitiser: Camera,
        up_offset: float = 0.05,
    ) -> float:
        """Compute a flux level that keeps measured values below saturation.

        Args:
            digitiser: Digitiser used to determine the saturation level.
            up_offset: Safety margin above the observed maximum measurement.

        Returns:
            A flux value suitable for scaling the current measured data.

        Raises:
            ValueError: If ``digitiser`` is not provided.
        """
        if not digitiser:
            raise ValueError("Digitiser must be provided.")

        optimal_flux = digitiser.saturation / (np.max(self.measureds) * (1 + up_offset))
        return optimal_flux


    def get_min_max(
        self,
        min_flux: float,
        max_flux: float,
        digitiser: Camera,
        down_offset: float = 0.5,
        up_offset: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the min and max values after flux scaling and digitisation.

        Args:
            min_flux: Minimum flux value to evaluate.
            max_flux: Maximum flux value to evaluate.
            digitiser: Digitiser used to quantise the scaled measurements.
            down_offset: Safety margin applied to the lower flux bound.
            up_offset: Safety margin applied to the upper flux bound.

        Returns:
            A tuple containing the global minimum and maximum target values.
        """
        at_min_flux = self.measureds * min_flux
        at_max_flux = self.measureds * max_flux

        at_min_flux = digitiser.digitise(at_min_flux * (1 - down_offset))
        at_max_flux = digitiser.digitise(at_max_flux * (1 + up_offset))

        # print(at_min_flux.min(axis=0))
        # print(at_min_flux.max(axis=0))
        # print(at_max_flux.min(axis=0))
        # print(at_max_flux.max(axis=0))

        # print("Normalizing min/max spectra...")

        at_min_flux = at_min_flux / min_flux
        at_max_flux = at_max_flux / max_flux

        # print([f"{f:.8f}" for f in at_min_flux.min(axis=0)])
        # print([f"{f:.8f}" for f in at_max_flux.min(axis=0)])
        # print([f"{f:.8f}" for f in at_min_flux.max(axis=0)])
        # print([f"{f:.8f}" for f in at_max_flux.max(axis=0)])

        min_at_min_flux = np.min(at_min_flux, axis=0)
        max_at_min_flux = np.max(at_min_flux, axis=0)
        min_at_max_flux = np.min(at_max_flux, axis=0)
        max_at_max_flux = np.max(at_max_flux, axis=0)

        mined = np.min(np.concatenate([at_min_flux, at_max_flux], axis=0), axis=0)
        maxed = np.max(np.concatenate([at_min_flux, at_max_flux], axis=0), axis=0)

        # print("Finding global min/max...")
        # print([f"{f:.8f}" for f in mined])
        # print([f"{f:.8f}" for f in maxed])

        return mined, maxed


    def __mul__(self, flux: Union[int, float, np.ndarray]) -> LASDataset | NotImplementedType:
        """Scale the dataset by a scalar or per-sample flux vector.

        Args:
            flux: Scalar flux or a one-dimensional array of per-sample fluxes.

        Returns:
            A new scaled dataset, or ``NotImplemented`` for unsupported inputs.
        """
        if isinstance(flux, (int, float)):
            new_dataset = LASDataset(self.spectra * flux, self.design, self.sample_idxs)
            return new_dataset
        if isinstance(flux, np.ndarray) and flux.shape == (self.spectra.shape[0],):
            new_dataset = LASDataset(self.spectra * flux[:, np.newaxis], self.design, self.sample_idxs)
            return new_dataset
        return NotImplemented

    def __len__(self) -> int:
        """Return the number of spectra in the dataset."""
        return self.spectra.shape[0]

    def datapair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a measured/spectrum pair.

        Args:
            idx: Sample index.

        Returns:
            A tuple containing the measured output and target spectrum.
        """
        return self.measureds[idx], self.spectra[idx, self.sample_idxs]

    def __getitem__(self, idx: Union[int, IndexLike]) -> tuple[np.ndarray, np.ndarray] | LASSubset:
        """Access a datapoint or subset by index.

        Args:
            idx: Integer index or a slice/array/boolean mask.

        Returns:
            A raw measured/spectrum pair for integer indices, or a subset view.
        """
        if isinstance(idx, (int, np.integer)):
            return self.measureds[idx], self.spectra[idx]
        return self.subset(idx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_split_sizes(n: int, a: float | int, b: float | int, c: float | int) -> Tuple[int, int, int]:
        """Resolve split sizes from either fractions or absolute counts.

        Args:
            n: Total dataset size.
            a: Train split fraction or size.
            b: Validation split fraction or size.
            c: Test split fraction or size.

        Returns:
            Integer counts for train, validation, and test splits.
        """
        if all(isinstance(x, (float, np.floating)) for x in (a, b, c)):
            total = a + b + c
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"Fractions must sum to 1.0 (got {total}).")
            train_n = int(round(a * n))
            val_n = int(round(b * n))
            test_n = n - train_n - val_n
        else:
            train_n, val_n, test_n = int(a), int(b), int(c)
            if train_n + val_n + test_n > n:
                raise ValueError("Requested split sizes exceed dataset length.")
        return train_n, val_n, test_n

    def _normalise_indices(self, indices: IndexLike) -> np.ndarray:
        if isinstance(indices, slice):
            return np.arange(len(self))[indices]
        indices = np.asarray(indices)
        if indices.dtype == bool:
            if indices.shape[0] != len(self):
                raise ValueError("Boolean mask length must match dataset length.")
            return np.flatnonzero(indices)
        return indices.astype(int)


class LASSubset:
    """Subset view over an :class:`LASDataset`."""

    def __init__(self, parent: LASDataset, indices: np.ndarray) -> None:
        """Initialise the subset.

        Args:
            parent: Parent dataset owning the underlying arrays.
            indices: Parent indices included in this subset.
        """
        self.parent = parent
        self.indices = np.asarray(indices, dtype=int)

        self.spectra = parent.spectra[self.indices]
        self.measureds = parent.measureds[self.indices]

        # Inherit target mapping
        self.sample_idxs = parent.sample_idxs
        self.target_bins = parent.target_bins

    def subset(self, indices: IndexLike) -> LASSubset:
        """Create a subset of this subset.

        Args:
            indices: Integer indices, slices, or boolean masks selecting rows.

        Returns:
            A subset view over the selected samples.
        """
        sub_idx = self._normalise_indices(indices)
        # Map local indices back to parent index space
        mapped = self.indices[sub_idx]
        return LASSubset(self.parent, mapped)

    def subset_random(self, ratio: float, random_state: Optional[int] = None) -> LASSubset:
        """Create a random subset of the current subset.

        Args:
            ratio: Fraction of the current subset to sample.
            random_state: Seed used for sampling.

        Returns:
            A randomly sampled subset view.

        Raises:
            ValueError: If ``ratio`` is not between 0 and 1.
        """
        if not (0 < ratio < 1):
            raise ValueError("Ratio must be between 0 and 1.")

        np.random.seed(random_state)
        sample_indices = np.random.choice(self.indices, size=int(len(self) * ratio), replace=False)
        mapped = sample_indices
        return LASSubset(self.parent, mapped)
    
    def __mul__(self, flux: Union[int, float, np.ndarray]) -> LASSubset | NotImplementedType:
        """Scale the subset by a scalar or per-sample flux vector.

        Args:
            flux: Scalar flux or a one-dimensional array of per-sample fluxes.

        Returns:
            A new scaled subset, or ``NotImplemented`` for unsupported inputs.
        """
        if isinstance(flux, (int, float)):
            new_subset = LASSubset(self.parent, self.indices)
            new_subset.measureds = self.measureds * flux
            new_subset.spectra = self.spectra * flux
            return new_subset
        if isinstance(flux, np.ndarray) and flux.shape == (self.measureds.shape[0],):
            new_subset = LASSubset(self.parent, self.indices)
            new_subset.measureds = self.measureds * flux[:, np.newaxis]
            new_subset.spectra = self.spectra * flux[:, np.newaxis]
            return new_subset
        return NotImplemented
    
    def prepare(self, out_scaler: Scaler) -> None:
        """Prepare scaled labels for model training.

        Args:
            out_scaler: Scaler used to transform the target spectra.
        """
        self.labels = out_scaler.transform(self.spectra[:, self.sample_idxs])

    # -------------------------------
    # PyTorch-style interface
    # -------------------------------
    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return self.spectra.shape[0]

    def datapair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a raw measured/spectrum pair.

        Args:
            idx: Sample index.

        Returns:
            A tuple containing the measured output and target spectrum.
        """
        return self.measureds[idx], self.spectra[idx]

    def datapoint(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a processed feature/label pair.

        Args:
            idx: Sample index.

        Returns:
            A tuple containing the processed features and labels.

        Raises:
            ValueError: If the subset has not been prepared.
        """
        if not self.processed:
            raise ValueError("Subset not processed. Call `prepare` before accessing training data.")
        return self.data[idx], self.labels[idx]

    def __getitem__(self, idx: Union[int, IndexLike]) -> tuple[np.ndarray, np.ndarray] | LASSubset:
        """Access a datapoint or subset by index.

        Args:
            idx: Integer index or a slice/array/boolean mask.

        Returns:
            A raw measured/spectrum pair for integer indices, or a subset view.
        """
        if isinstance(idx, (int, np.integer)):
            return self.measureds[idx], self.labels[idx]
        return self.subset(idx)

    # -------------------------------
    # Helpers
    # -------------------------------
    def _normalise_indices(self, indices: IndexLike) -> np.ndarray:
        """Normalise indices into an integer NumPy array.

        Args:
            indices: Integer indices, slices, or boolean masks.

        Returns:
            Integer indices suitable for array indexing.
        """
        if isinstance(indices, slice):
            return np.arange(len(self))[indices]
        indices = np.asarray(indices)
        if indices.dtype == bool:
            if indices.shape[0] != len(self):
                raise ValueError("Boolean mask length must match subset length.")
            return np.flatnonzero(indices)
        return indices.astype(int)


class Mutator:
    def __init__(
        self,
        digitiser: Camera,
        flux_range: Tuple[float, float],
        noise_range: Optional[Tuple[float, float] | list[float] | int],
        flux_scale: str,
        noise_scale: str,
        mature: float = 1.0
    ) -> None:
        """Initialize the mutator (data augmentation).

        Args:
            digitiser: Digitiser used to quantise measurements.
            flux_range: Lower and upper bounds for flux scaling.
            noise_range: Noise bounds, percentage bounds, or sentinel values.
            flux_scale: Scaling mode for flux progression across epochs.
            noise_scale: Scaling mode for noise progression across epochs.
            mature: Fraction of epochs over which the schedule varies.
        """
        self.digitiser = digitiser
        self.flux_range = flux_range
        if noise_range is None:
            self.noise_range = (0.0, 1.0)
            self.noise_type = 'sqrt'
        elif type(noise_range) in (tuple, list) and len(noise_range) == 2:
            if noise_range[0] <= 1.0 and noise_range[1] <= 1.0 and noise_range[0] >= 0.0 and noise_range[1] >= 0.0:
                self.noise_range = tuple(noise_range)
                self.noise_type = 'sqrt'
            else:
                self.noise_range = (noise_range[0]/100, noise_range[1]/100)
                self.noise_type = 'fixed'
        elif noise_range == -1:
            self.noise_range = (0.0, 1.0)
            self.noise_type = 'sqrt'
        else:
            raise ValueError("Invalid noise_range parameter.")
        self.flux_scale = flux_scale
        self.noise_scale = noise_scale
        self.mature = mature

    def range_for_epoch(
        self,
        epoch: int,
        total_epochs: int,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute flux and noise ranges for a given epoch.

        Args:
            epoch: Current epoch index.
            total_epochs: Total number of training epochs.

        Returns:
            A tuple containing the flux range and the noise range for the epoch.

        Raises:
            ValueError: If the epoch configuration or scale settings are invalid.
        """
        mature = self.mature
        flux_scale = self.flux_scale
        noise_scale = self.noise_scale

        # ---- validation ----
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive.")
        if not (0.0 <= mature <= 1.0):
            raise ValueError("mature must be in [0, 1].")
        if epoch < 0 or epoch >= total_epochs:
            raise ValueError("epoch must be in [0, total_epochs-1].")

        min_flux, max_flux = self.flux_range  # assume (min, max)
        if min_flux <= 0 or max_flux <= 0:
            raise ValueError("Flux values must be positive for log10 spacing.")

        min_noise, max_noise = self.noise_range

        # ---- maturity handling ----
        # Number of epochs during which we vary the flux and noise ranges
        vary_epochs = int(round(mature * total_epochs))
        if vary_epochs <= 0:
            # mature == 0, full ranges immediately
            return (min_flux, max_flux), (min_noise, max_noise)
        if epoch >= vary_epochs:
            # after maturity, full ranges
            return (min_flux, max_flux), (min_noise, max_noise)

        # Map epoch in [0, vary_epochs-1] in [0,1]
        denom = max(vary_epochs - 1, 1)
        t = epoch / denom

        def lerp(a: float, b: float, u: float) -> float:
            return a + (b - a) * u

        def ginterp(a: float, b: float, u: float) -> float:
            if a <= 0 or b <= 0:
                raise ValueError("Logarithmic scaling requires positive values.")
            return a * (b / a) ** u

        flux_end = max_flux  # constant
        if flux_scale in ('linear', 'lin'):
            flux_start = 10 ** lerp(np.log10(max_flux), np.log10(min_flux), t)
        elif flux_scale in ('log', 'logarithmic'):
            flux_start = 10 ** ginterp(np.log10(max_flux), np.log10(min_flux), t)
        else:
            raise ValueError("flux_scale must be 'linear'/'lin' or 'log'/'logarithmic'.")

        noise_start = min_noise 
        if noise_scale in ('linear', 'lin'):
            noise_end = lerp(min_noise, max_noise, t)
        elif noise_scale in ('log', 'logarithmic'):
            noise_end = ginterp(min_noise, max_noise, t)
        else:
            raise ValueError("noise_scale must be 'linear'/'lin' or 'log'/'logarithmic'.")

        return (flux_start, flux_end), (noise_start, noise_end)

    def prepare(
        self,
        unit_measurables: np.ndarray,
        flux_range: Tuple[float, float],
        noise_range: Tuple[float, float],
        return_flux: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Apply flux scaling, noise, and digitisation to unit measurements.

        Args:
            unit_measurables: Input measurements on a unit flux scale.
            flux_range: Lower and upper bounds for flux scaling.
            noise_range: Lower and upper bounds for noise generation.
            return_flux: Whether to also return the sampled flux values.

        Returns:
            The digitised measured values, optionally paired with sampled fluxes.
        """

        fluxes = np.random.uniform(np.log10(flux_range[0]), np.log10(flux_range[1]), size=unit_measurables.shape[0])
        fluxes = 10 ** fluxes
        measurables = unit_measurables * fluxes[:, np.newaxis]
        measureds = measurables.clip(max=self.digitiser.saturation)

        if self.noise_type == 'fixed':
            signs = np.random.choice([-1, 1], size=measureds.shape)
            # Ensure not all signs are the same in each 1D array
            if measureds.ndim == 1:
                # For 1D case
                if np.all(signs == 1):
                    signs[np.random.randint(len(signs))] = -1
                elif np.all(signs == -1):
                    signs[np.random.randint(len(signs))] = 1
            elif measureds.ndim > 1:
                # For 2D+ arrays - ensure variety per row (axis=1)
                for i in range(signs.shape[0]):
                    row = signs[i]
                    if np.all(row == 1):
                        row[np.random.randint(len(row))] = -1
                    elif np.all(row == -1):
                        row[np.random.randint(len(row))] = 1
                    signs[i] = row

            noise = (
                np.random.uniform(noise_range[0], noise_range[1], size=measureds.shape)
                * signs
                * measureds
            )

        elif self.noise_type == 'sqrt':
            signs = np.random.choice([-1, 1], size=measureds.shape)
            if measureds.ndim == 1:
                if np.all(signs == 1):
                    signs[np.random.randint(len(signs))] = -1
                elif np.all(signs == -1):
                    signs[np.random.randint(len(signs))] = 1
            elif measureds.ndim > 1:
                for i in range(signs.shape[0]):
                    row = signs[i]
                    if np.all(row == 1):
                        row[np.random.randint(len(row))] = -1
                    elif np.all(row == -1):
                        row[np.random.randint(len(row))] = 1
                    signs[i] = row

            noise = (
                np.random.uniform(noise_range[0], noise_range[1], size=measureds.shape)
                * signs
                * np.sqrt(measureds)
            )

        else:
            raise ValueError("Invalid noise type.")
        noised_measureds = measureds + noise
        noised_measureds = self.digitiser.digitise(noised_measureds)

        if not return_flux:
            return noised_measureds
        else:
            return noised_measureds, fluxes

    def mutate(
        self,
        unit_measurables: np.ndarray,
        epoch: int,
        total_epochs: int,
    ) -> np.ndarray:
        """Mutate unit measurements using the epoch-specific ranges.

        Args:
            unit_measurables: Input measurements on a unit flux scale.
            epoch: Current epoch index.
            total_epochs: Total number of training epochs.

        Returns:
            The digitised, noised measured values.

        Raises:
            ValueError: If no digitiser is configured.
        """
        if not self.digitiser:
            raise ValueError("Digitiser must be provided.")
        flux_range, noise_range = self.range_for_epoch(
            epoch,
            total_epochs
        )
        X = self.prepare(unit_measurables, flux_range, noise_range)
        return X
