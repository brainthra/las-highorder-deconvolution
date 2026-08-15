"""Reconstruction models and inference helpers."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from warnings import warn

from typing import Any

import numpy as np
import os
import torch
import tqdm
from . import models as models
from . import metrics as metrics
from .data import LASDataset, LASSubset, Mutator
from lasim.system import Spectrometer
from .scalers import Scaler
from sklearn.preprocessing import Normalizer
from .data import LASDataset
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d, PchipInterpolator
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def init_func(m: Any) -> None:
    """Reset module parameters when available.

    Args:
        m: Module or layer object being initialised.
    """
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

class InferenceInfo():
    """Container for reconstruction outputs and optional uncertainty estimates."""

    def __init__(
        self,
        measured: np.ndarray,
        measured_unit: np.ndarray,
        flux: float,
        spec_tar: np.ndarray,
        spec_tar_unit_sub: np.ndarray,
        spec_tar_unit: np.ndarray,
        spec_full_unit: np.ndarray,
        spec_full: np.ndarray,
        spec_intp_unit_sub: np.ndarray,
        spec_tar_unit_sub_var: np.ndarray | None = None,
        spec_intp_unit_sub_var: np.ndarray | None = None,
        spec_tar_var: np.ndarray | None = None,
        spec_full_var: np.ndarray | None = None,
        spec_tar_unit_subs: np.ndarray | None = None,
    ) -> None:
        """Initialise inference metadata.

        Args:
            measured: Original measured spectrum.
            measured_unit: Measured spectrum in unit-flux space.
            flux: Estimated flux factor.
            spec_tar: Reconstructed target spectrum.
            spec_tar_unit_sub: Target-region spectrum before interpolation.
            spec_tar_unit: Interpolated target-region spectrum in unit space.
            spec_full_unit: Full reconstructed spectrum in unit space.
            spec_full: Full reconstructed spectrum in measured space.
            spec_intp_unit_sub: Interpolated spectrum on the sampled target bins.
            spec_tar_unit_sub_var: Optional variance for target-bin predictions.
            spec_intp_unit_sub_var: Optional variance for interpolated predictions.
            spec_tar_var: Optional variance for the reconstructed target spectrum.
            spec_full_var: Optional variance for the reconstructed full spectrum.
            spec_tar_unit_subs: Optional ensemble predictions on target bins.
        """
        self.measured = measured
        self.measured_unit = measured_unit
        self.flux = flux
        self.spec_tar = spec_tar
        self.spec_tar_unit_sub = spec_tar_unit_sub
        self.spec_tar_unit = spec_tar_unit
        self.spec_full_unit = spec_full_unit
        self.spec_full = spec_full
        self.spec_intp_unit_sub = spec_intp_unit_sub
        self.spec_tar_unit_sub_var = spec_tar_unit_sub_var
        self.spec_intp_unit_sub_var = spec_intp_unit_sub_var
        self.spec_tar_var = spec_tar_var
        self.spec_full_var = spec_full_var
        self.spec_tar_unit_subs = spec_tar_unit_subs


class Reconstructor:
    """Abstract base class for reconstructor implementations."""
    
    def __init__(self, spectrometer: Spectrometer, model: Any = None, in_scaler: Scaler | None = None, out_scaler: Scaler | None = None) -> None:
        """Initialise the reconstructor base class.

        Args:
            spectrometer: Spectrometer describing the energy grid.
            model: Underlying reconstruction model.
            in_scaler: Optional input scaler.
            out_scaler: Optional output scaler.
        """
        warn("This is an abstract class and should not be instantiated directly.", UserWarning)

    def fit(self, train_dataset: LASDataset, **kwargs: Any) -> Any:
        """Train the reconstructor.

        Args:
            train_dataset: Training dataset.
            **kwargs: Additional implementation-specific arguments.

        Raises:
            NotImplementedError: Always raised by the abstract base class.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Any:
        """Predict output features.

        Args:
            X: Input features.

        Raises:
            NotImplementedError: Always raised by the abstract base class.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the reconstructor.

        Args:
            path: Destination path.

        Raises:
            NotImplementedError: Always raised by the abstract base class.
        """
        raise NotImplementedError

class MultiReconstructor(Reconstructor):
    """Ensemble reconstructor with multiple models."""

    def __init__(
        self,
        spectrometer: Spectrometer,
        model: models.Model,
        num_models: int,
        sample_idxs: np.ndarray,
        target_range: list[float],
        in_scaler: Scaler | None = None,
        out_scaler: Scaler | None = None,
    ) -> None:
        """Initialise the ensemble reconstructor.

        Args:
            spectrometer: Spectrometer describing the energy grid.
            model: Base reconstruction model to clone.
            num_models: Number of ensemble members.
            sample_idxs: Energy-bin indices used as model outputs.
            target_range: Two-element range defining the target energy span.
            in_scaler: Optional input scaler.
            out_scaler: Optional output scaler.
        """
        self.seeds = [0, 1, 5, 6, 9, 10, 15, 33, 42, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123][:num_models]
        self.spectrometer = spectrometer
        self.FULL_ENERGIES = spectrometer.ENERGIES
        self.models = [deepcopy(model) for _ in range(num_models)]
        if isinstance(model, torch.nn.Module):
            print("Using torch model")
            for i in range(num_models):
                torch.manual_seed(self.seeds[i])
                self.models[i].apply(init_func)

        self.preprocessor = Normalizer(norm="l1")
        self.in_scaler = in_scaler
        self.out_scaler = out_scaler
        self.model_type = "torch" if isinstance(model, torch.nn.Module) else "sklearn"

        self.sample_idxs = sample_idxs
        self.sample_energy_bins = spectrometer.ENERGIES[sample_idxs]
        self.max_target_energy = target_range[1]
        self.max_target_idx = np.argmin(np.abs(self.FULL_ENERGIES - self.max_target_energy))
        self.TARGET_ENERGIES = self.FULL_ENERGIES[:self.max_target_idx+1]
        
        self.target_idxs = self.sample_idxs[self.sample_idxs <= self.max_target_idx]
        self.target_energy_bins = self.sample_energy_bins[self.sample_idxs <= self.max_target_idx]

        self.extra_idxs = self.sample_idxs[self.sample_idxs > self.max_target_idx]
        self.extra_energy_bins = self.sample_energy_bins[self.sample_idxs > self.max_target_idx]

        self.num_targets = len(self.target_idxs)
    

    def fit(self, dataset: LASDataset, split_ratio: list[float] = [0.6, 0.15, 0.25], return_info: bool = False, mutator: Mutator | None = None, **kwargs: Any) -> Any:
        """Train the ensemble.

        Args:
            dataset: Full dataset to split and train on.
            split_ratio: Train/validation/test split ratios.
            return_info: Whether to return fit metadata instead of the splits.
            mutator: Optional mutator applied during training.
            **kwargs: Additional model-specific arguments.

        Returns:
            Fit metadata or the train/validation/test splits.
        """
        train_ds, val_ds, test_ds = dataset.split(split_ratio)
        verbose = kwargs.get("verbose", False)
        kwargs["verbose"] = True if verbose==2 else False

        train_ds.prepare(self.out_scaler)
        val_ds.prepare(self.out_scaler)
        test_ds.prepare(self.out_scaler)

        fit_infos = []
        if self.model_type == "torch":
            optimisers = [kwargs["optimizer"](self.models[i].parameters(), kwargs["lr"]) for i in range(len(self.models))] if "weight_decay" not in kwargs else [kwargs["optimizer"](self.models[i].parameters(), kwargs["lr"], weight_decay=kwargs["weight_decay"]) for i in range(len(self.models))]
            kwargs_copies = [deepcopy(kwargs) for _ in range(len(self.models))]
            for i in tqdm(range(len(self.models)), desc="Training models ") if verbose==1 else range(len(self.models)):
                model = self.models[i]
                # subset of train data of 30%
                train_ss = train_ds.subset_random(0.3, random_state=self.seeds[i])
                train_ss.prepare(self.out_scaler)
                kwargs_copies[i]["optimizer"] = optimisers[i]
                kwargs_copies[i].pop("lr", None)
                kwargs_copies[i].pop("weight_decay", None) if "weight_decay" in kwargs_copies[i] else None
                kwargs_copies[i]["mutator"] = mutator
                fit_infos += [model.fit(train_ss, val_ds, **kwargs_copies[i])]
                
                if verbose==2:
                    print(f"Model {i+1}/{len(self.models)} trained.")
        elif self.model_type == "sklearn":
            kwargs.pop("dlr", None)
            kwargs.pop("verbose", None)
            for i, model in tqdm(enumerate(self.models), desc="Training models ") if verbose==1 else enumerate(self.models):
                # subset of train data of 30%
                train_ss = train_ds.subset_random(0.3, random_state=self.seeds[i])
                fit_infos += [model.fit(train_ss.data, train_ss.labels)]
                if verbose==2:
                    print(f"Model {i+1}/{len(self.models)} trained.")
        else:
            raise ValueError("Model type not supported")
        return fit_infos if return_info else (train_ds, val_ds, test_ds)

    def predict(self, X: np.ndarray, return_error: bool = False, return_all: bool = False) -> Any:
        """Predict outputs using the ensemble mean.

        Args:
            X: Input features.
            return_error: Whether to return ensemble standard deviation.
            return_all: Whether to also return every ensemble member prediction.

        Returns:
            Ensemble mean predictions, optionally with uncertainty and all outputs.
        """
        X = X.reshape(1, -1) if len(X.shape) == 1 else X
        outputs = []
        for model in self.models:
            if self.model_type == "torch":
                model.eval()
                outputs += [model(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy()]
            elif self.model_type == "sklearn":
                outputs += [model.predict(X)]
            else:
                raise ValueError("Model type not supported")
        outputs = np.array(outputs)
        pred = np.mean(outputs, axis=0)
        if return_error:
            outputs_var = np.std(outputs, axis=0)
            if return_all:
                return pred, outputs_var, outputs
            return pred, outputs_var
        if return_all:
            return pred, outputs
        return pred

    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs and invert output scaling.

        Args:
            X: Input features.

        Returns:
            Predictions transformed back to the original output scale.
        """
        X_scaled = self.in_scaler.transform(X) if self.in_scaler else X
        return self.out_scaler.inverse_transform(self.model.predict(X_scaled)) if self.out_scaler else self.model.predict(X_scaled)
        
    def inference_method(self, measured: np.ndarray, method: str = "opt", digitiser: Spectrometer | None = None, return_err: bool = False, return_info: bool = False, debug: bool = False) -> Any:
        """Run ensemble inference on a single sample.

        Args:
            measured: Measured spectrum.
            method: Flux estimation method.
            digitiser: Optional digitiser used for saturation bounds.
            return_err: Whether to return uncertainty estimates.
            return_info: Whether to return a detailed :class:`InferenceInfo` object.
            debug: Whether to print intermediate values.

        Returns:
            The reconstructed target spectrum, optionally paired with uncertainties and metadata.
        """
        min_sat = digitiser.visible if digitiser else 1e-10
        max_sat = digitiser.saturation if digitiser else np.inf
        # Normalise inputs
        true_unit_measured = self.preprocessor.transform(measured)
        true_unit_measured = self.in_scaler.transform(true_unit_measured) if self.in_scaler else true_unit_measured

        # Forward pass
        spec_intp_unit_sub_norm, spec_intp_unit_sub_norm_var, spec_intp_unit_sub_norms = self.predict(true_unit_measured, return_error=True, return_all=True)
        # Denormalise outputs
        spec_intp_unit_sub = self.out_scaler.inverse_transform(spec_intp_unit_sub_norm).flatten() if self.out_scaler else spec_intp_unit_sub_norm.flatten()
        if return_err:
            # spec_intp_unit_sub_var = self.out_scaler.inverse_transform(spec_intp_unit_sub_norm_var).flatten() if self.out_scaler else spec_intp_unit_sub_norm_var.flatten()
            spec_intp_unit_subs = np.array([self.out_scaler.inverse_transform(spec_intp_unit_sub_norms[i]) for i in range(len(spec_intp_unit_sub_norms))] if self.out_scaler else spec_intp_unit_sub_norms)
            spec_intp_unit_sub_var = np.std(spec_intp_unit_subs, axis=0).squeeze()
        else:
            spec_intp_unit_sub_var = None
            spec_intp_unit_subs = None
        # print(spec_intp_unit_sub.shape)
        
        spec_tar_unit_sub = spec_intp_unit_sub[:self.num_targets]
        extrapec = spec_intp_unit_sub[self.num_targets:]
        spec_tar_unit_subs = spec_intp_unit_subs[:, :self.num_targets] if return_err else None
        spec_tar_unit_sub_var = spec_intp_unit_sub_var[:self.num_targets] if return_err else None
        extrapec_var = spec_intp_unit_sub_var[self.num_targets:] if return_err else None

        interp = interp1d(self.target_energy_bins, spec_tar_unit_sub, kind='cubic')
        spec_tar_unit = interp(self.TARGET_ENERGIES)
        spec_tar_unit = np.clip(spec_tar_unit, 0, None)

        if return_err:
            # print(len(self.target_energy_bins), len(spec_tar_unit_sub_var))
            interp_var = interp1d(self.target_energy_bins, spec_tar_unit_sub_var, kind='cubic')
            spec_tar_unit_var = interp_var(self.TARGET_ENERGIES)
            spec_tar_unit_var = np.clip(spec_tar_unit_var, 0, None)
        else:
            spec_tar_unit_var = None

        interp = PchipInterpolator(np.concatenate([self.TARGET_ENERGIES, self.extra_energy_bins]), np.concatenate([spec_tar_unit, extrapec]))
        spec_full_unit = interp(self.FULL_ENERGIES).clip(0, None)

        if return_err:
            interp_var = PchipInterpolator(np.concatenate([self.TARGET_ENERGIES, self.extra_energy_bins]), np.concatenate([spec_tar_unit_var, extrapec_var]))
            spec_full_unit_var = interp_var(self.FULL_ENERGIES).clip(0, None)

        measured_unit = self.spectrometer.unit(spec_full_unit)

        if method == "max":
            flux = np.max(measured/measured_unit)
        elif method in ["mean", "average"]:
            flux = np.mean(measured/measured_unit)
        elif method == "opt":
            mask = np.logical_and(np.logical_and(measured.squeeze() > min_sat, measured_unit.squeeze() > 1e-10), measured.squeeze() < max_sat)
            numerator = np.dot(measured.squeeze()[mask], measured_unit.squeeze()[mask])
            denominator = np.dot(measured_unit.squeeze()[mask], measured_unit.squeeze()[mask])
            flux = numerator / denominator
        if debug:
            print("e_dep =", measured)
            print("unit_e_dep =", measured_unit)
            print("flux_div =", measured/measured_unit)
            print("flux =", flux)
        spec_tar = spec_tar_unit * flux
        if return_err:
            spec_tar_var = spec_tar_unit_var * flux
        else:
            spec_tar_var = None
        # interp = interp1d(ENERGIES[sample_linear_indices(ENERGIES, 10)], sub_spec, kind='cubic')
        # spec = interp(ENERGIES)
        spec_tar = spec_tar.squeeze().astype(np.uint64)
        if return_info:
            spec_full = spec_full_unit * flux
            if return_err:
                spec_full_var = spec_full_unit_var * flux
            else:
                spec_full_var = None
            info = InferenceInfo(measured, measured_unit, flux, spec_tar, spec_tar_unit_sub, spec_tar_unit, spec_full_unit, spec_full, spec_intp_unit_sub, spec_tar_unit_sub_var=spec_tar_unit_sub_var, spec_intp_unit_sub_var=spec_intp_unit_sub_var, spec_tar_var=spec_tar_var, spec_full_var=spec_full_var, spec_tar_unit_subs=spec_tar_unit_subs)
            if return_err:
                return spec_tar, spec_tar_var, info
            return spec_tar, info
        if return_err:
            return spec_tar, spec_tar_var
        return spec_tar

    def inference(self, e_dep: np.ndarray, method: str = "opt", digitiser: Spectrometer | None = None, return_err: bool = False, return_info: bool = False, debug: bool = False) -> Any:
        """Run ensemble inference on an energy-deposition sample.

        Args:
            e_dep: Measured energy-deposition spectrum.
            method: Flux estimation method.
            digitiser: Optional digitiser used for saturation bounds.
            return_err: Whether to return uncertainty estimates.
            return_info: Whether to return a detailed :class:`InferenceInfo` object.
            debug: Whether to print intermediate values.

        Returns:
            The reconstructed target spectrum, optionally paired with uncertainties and metadata.
        """
        e_dep = e_dep.reshape(1, -1) if len(e_dep.shape) == 1 else e_dep
        return self.inference_method(e_dep, method=method, digitiser=digitiser, return_err=return_err, return_info=return_info, debug=debug)

    def inference_all(self, e_dep: np.ndarray, method: str = "opt", digitiser: Spectrometer | None = None) -> tuple[np.ndarray, InferenceInfo]:
        """Run inference and always return the detailed metadata.

        Args:
            e_dep: Measured energy-deposition spectrum.
            method: Flux estimation method.
            digitiser: Optional digitiser used for saturation bounds.

        Returns:
            The reconstructed target spectrum and an :class:`InferenceInfo` object.
        """
        return self.inference_method(e_dep, method=method, digitiser=digitiser, return_info=True)

    def save(self, path: str) -> None:
        """Save the reconstructor.

        Args:
            path: Destination path.
        """
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """Load the reconstructor.

        Args:
            path: Source path.
        """
        raise NotImplementedError        