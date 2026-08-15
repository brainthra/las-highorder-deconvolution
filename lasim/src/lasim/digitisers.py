"""
Camera digitisation.
"""

import numpy as np
from numpy.typing import DTypeLike, NDArray


class Camera:
    """Simulate the digitisation characteristics of a camera.

    The camera :
    - clips measurable values to its visible range, 
    - quantises them according to a fixed step size, and 
    - converts the result to the configured output data type.

    Args:
        visible: Minimum measurable value that can be represented.
        saturation: Maximum measurable value.
        step: Quantisation step size.
        dtype: NumPy-compatible data type used for the digitised output. Defaults to ``np.int64``.
    """

    def __init__(
        self,
        visible: int,
        saturation: int,
        step: int,
        dtype: DTypeLike = np.int64,
    ) -> None:
        """Initialise the camera.

        Args:
            visible: Minimum measurable value that can be represented.
            saturation: Maximum measurable value before saturation occurs.
            step: Quantisation step size.
            dtype: NumPy-compatible data type used for the digitised output.
                Defaults to ``np.int64``.
        """
        self.visible = visible
        self.saturation = saturation
        self.step = step
        self.dtype = dtype

    def digitise(self, measurable: NDArray[np.number]) -> np.ndarray:
        """Simulate the digitisation of measurable values.

        Values are:
        - quantised to the nearest multiple of :attr:`step`, 
        - clipped to the camera's measurable range, and 
        - converted to :attr:`dtype`.

        Args:
            measurable: Array of measurable values to digitise.

        Returns:
            Array containing the digitised values with data type
            :attr:`dtype`.
        """
        measured = np.round(np.clip(measurable, self.visible, self.saturation) / self.step) * self.step
        measured = np.clip(measured, self.visible, self.saturation)

        return measured.astype(self.dtype)