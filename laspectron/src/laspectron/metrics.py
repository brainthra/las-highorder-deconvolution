"""Metrics used to assess reconstruction quality."""

from __future__ import annotations

import numpy as np
from types import SimpleNamespace

def kldiv(
    p: np.ndarray,
    q: np.ndarray,
    base: float = 2,
    axis: int = -1,
    epsilon: float = 1e-12,
) -> float | np.ndarray:
    """Compute the Kullback-Leibler divergence.

    Args:
        p: First probability distribution.
        q: Second probability distribution.
        base: Logarithm base to use.
        axis: Axis along which to compute the divergence.
        epsilon: Small value to avoid division by zero and log of zero.

    Returns:
        The KL divergence along the specified axis.
    """
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    log_base = 1.0 if base is None or np.isclose(base, np.e) else np.log(base)
    return np.sum(p * (np.log(p) - np.log(q)) / log_base, axis=axis)

def jsdiv(
    p: np.ndarray,
    q: np.ndarray,
    base: float = 2,
    axis: int = -1,
    epsilon: float = 1e-12,
) -> float | np.ndarray:
    """Compute the Jensen-Shannon divergence.

    Args:
        p: First probability distribution.
        q: Second probability distribution.
        base: Logarithm base to use.
        axis: Axis along which to compute the divergence.
        epsilon: Small value added to probabilities to avoid log(0).

    Returns:
        The Jensen-Shannon divergence between ``p`` and ``q``.
    """
    m = 0.5 * (p + q)
    return 0.5 * kldiv(p, m, base=base, axis=axis, epsilon=epsilon) + \
           0.5 * kldiv(q, m, base=base, axis=axis, epsilon=epsilon)

def mse(a: np.ndarray, b: np.ndarray, axis: int | tuple[int, ...] = -1) -> float | np.ndarray:
    """Compute the mean squared error.

    Args:
        a: First input array.
        b: Second input array.
        axis: Axis or axes along which the mean is computed.

    Returns:
        The mean squared error along the specified axis.
    """
    return np.mean((a - b) ** 2, axis=axis)

def mae(a: np.ndarray, b: np.ndarray, axis: int | tuple[int, ...] = -1) -> float | np.ndarray:
    """Compute the mean absolute error.

    Args:
        a: First input array.
        b: Second input array to compare with ``a``.
        axis: Axis or axes along which the MAE is computed.

    Returns:
        The mean absolute error along the specified axis.
    """
    return np.mean(np.abs(a - b), axis=axis)

def rae(pred: np.ndarray, true: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the relative absolute error.

    Args:
        pred: Predicted values.
        true: True values.
        axis: Axis along which the mean is computed.

    Returns:
        The relative absolute error computed along the specified axis.
    """
    return np.abs(pred - true) / true

def shapeErrors(
    pred_fulls: np.ndarray,
    true_fulls: np.ndarray,
    tar_limit: int = 100,
    epsilon: float = 1e-6,
    func: str = "jenshan",
    base: float = 2,
) -> SimpleNamespace:
    """Compute shape errors on full and target regions.

    Args:
        pred_fulls: Predicted full spectra.
        true_fulls: Ground-truth full spectra.
        tar_limit: Number of leading bins to treat as the target window.
        epsilon: Small value used to clip spectra away from zero.
        func: Error function to use. Supported values are ``jenshan``, ``jsdiv``, ``js``, ``js_div``, ``mse``, and ``mae``.
        base: Logarithm base used for divergence-based metrics.

    Returns:
        A namespace with ``target`` and ``full`` error values.

    Raises:
        ValueError: If the input arrays have incompatible shapes.
    """
    P = np.asarray(pred_fulls, dtype=float)
    T = np.asarray(true_fulls, dtype=float)

    if P.shape != T.shape:
        raise ValueError("pred_fulls and true_fulls must have the same shape.")
    if P.ndim == 0:
        raise ValueError("Inputs must be 1D or 2D arrays, not scalars.")
    if P.ndim > 2:
        raise ValueError(f"Inputs must be 1D or 2D; got ndim={P.ndim}.")

    # Reshape to (N, L) uniformly without branching (1D -> (1, L); 2D unchanged)
    N, L = (1, P.shape[-1]) if P.ndim == 1 else P.shape
    P = P.reshape(-1, L)
    T = T.reshape(-1, L)

    # Clip and slice target window
    P = np.clip(P, epsilon, None)
    T = np.clip(T, epsilon, None)
    limit = min(tar_limit, L)
    Pt = P[:, :limit]
    Tt = T[:, :limit]

    mode = func.lower()
    if mode in {"jenshan", "jsdiv", "js", "js_div"}:
        # Normalise to probability distributions along last axis
        Pf = P / np.sum(P, axis=-1, keepdims=True)
        Tf = T / np.sum(T, axis=-1, keepdims=True)
        Pts = Pt / np.sum(Pt, axis=-1, keepdims=True)
        Tts = Tt / np.sum(Tt, axis=-1, keepdims=True)

        full_errors = jsdiv(Pf, Tf, base=base, axis=-1, epsilon=epsilon)
        target_errors = jsdiv(Pts, Tts, base=base, axis=-1, epsilon=epsilon)

    elif mode == "mse":
        full_errors = mse(P, T, axis=-1)
        target_errors = mse(Pt, Tt, axis=-1)

    elif mode == "mae":
        full_errors = mae(P, T, axis=-1)
        target_errors = mae(Pt, Tt, axis=-1)

    else:
        raise ValueError(f"Unknown func '{func}'")

    # Return scalars if single sample, else vectors
    if N == 1:
        full_errors = float(full_errors[0])
        target_errors = float(target_errors[0])

    return SimpleNamespace(target=target_errors, full=full_errors)

def FluxError(pred_fulls: np.ndarray, true_fulls: np.ndarray) -> float | list[float]:
    """Compute the relative absolute error for flux.

    Args:
        pred_fulls: Predicted flux arrays.
        true_fulls: True flux arrays.

    Returns:
        The relative absolute errors for each sample.
    """
    pred_fulls = np.asarray(pred_fulls)
    true_fulls = np.asarray(true_fulls)
    pred_sums = np.sum(pred_fulls, axis=-1)
    true_sums = np.sum(true_fulls, axis=-1)
    errors = np.abs(pred_sums - true_sums) / true_sums
    return errors.tolist() if errors.ndim > 0 else float(errors)

# def Error()