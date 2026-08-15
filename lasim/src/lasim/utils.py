""""
Utility functions.
"""
import numpy as np
from numpy.typing import NDArray


def mev_to_kev(energy_mev: float | NDArray) -> float | NDArray:
    """Convert energy from MeV to keV.

    Args:
        energy_mev: Energy value or array of energy values in
            megaelectronvolts (MeV).

    Returns:
        Energy value or array of energy values in kiloelectronvolts (keV).
    """
    return energy_mev * 1e3

def kev_to_mev(energy_kev: float | NDArray) -> float | NDArray:
    """Convert energy from keV to MeV.

    Args:
        energy_kev: Energy value or array of energy values in
            kiloelectronvolts (keV).

    Returns:
        Energy value or array of energy values in megaelectronvolts (MeV).
    """
    return energy_kev / 1e3