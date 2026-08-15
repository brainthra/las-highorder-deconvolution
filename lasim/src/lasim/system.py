"""
Simulate the behaviour of linear absorption spectrometers.

The module provides material, layer, and spectrometer abstractions for building linear absorption spectrometer models. 
Response matrices are constructed from material cross-sections supplied by :mod:`macr`, and measurements can be reported as either deposited energy or detector counts.

The main public classes are :class:`Filter`, :class:`Scintillator`, :class:`Layer`, and :class:`Spectrometer`.
"""

from typing import Any, Literal, Optional
import warnings

from macr.engine import Engine  # type: ignore
from macr.material import Material  # type: ignore
from matplotlib import gridspec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["Filter", "Scintillator", "Layer", "Spectrometer"]


class Filter:
    """Represent a passive filter material.

    The material cross-sections are populated by :meth:`build_crosssections` for a supplied energy grid.

    Args:
        symbol: Material formula or symbol, for example ``"C4H12O6"``.
        density: Material density in g/cm³.
        generic_name: Optional human-readable name. If omitted, ``symbol`` is used.
    """

    def __init__(
        self,
        symbol: str,
        density: float,
        generic_name: Optional[str] = None,
    ) -> None:
        self.symbol: str = symbol
        self.density: float = density
        self.sigma: np.ndarray = np.array([])
        self.generic_name: str = generic_name if generic_name is not None else symbol

    def build_crosssections(self, energies: np.ndarray, engine: Engine) -> None:
        """Build material cross-sections for an energy grid.

        Args:
            energies: Energies at which to evaluate the cross-sections.
            engine: Cross-section engine used by :class:`macr.material.Material`.
        """
        mat = Material(self.symbol, self.density, energies, engine=engine)
        self.sigma = mat.sigma  # type: ignore

    def is_scintillator(self) -> bool:
        """Return whether this material is a scintillator.

        Returns:
            ``False`` for a passive filter.
        """
        return False


class Scintillator(Filter):
    """Represent an active scintillator material.

    Args:
        symbol: Material formula or symbol.
        density: Material density in g/cm³.
        kappa: Conversion factor from deposited energy to detector counts.
    """

    def __init__(self, symbol: str, density: float, kappa: float) -> None:
        super().__init__(symbol, density)
        self.kappa: float = kappa

    def is_scintillator(self) -> bool:
        """Return whether this material is a scintillator.

        Returns:
            ``True`` for a scintillator.
        """
        return True


class Layer:
    """Represent one material layer in a linear absorption spectrometer.

    Args:
        material: Filter or scintillator material forming the layer.
        thickness: Layer thickness in millimetres.
    """

    def __init__(self, material: Filter | Scintillator, thickness: float) -> None:
        self.material: Filter | Scintillator = material
        self.thickness: float = thickness

    def get_transmission(self) -> np.ndarray:
        """Calculate x-ray transmission through the layer.

        Returns:
            Transmission fraction at each energy
        """
        return np.exp(
            -self.material.sigma
            * self.material.density
            * (self.thickness * 0.1)
        )

    def get_absorption(self) -> np.ndarray:
        """Calculate the absorbed fraction of incident x-rays.

        Returns:
            Absorbed fraction at each energy.
        """
        if not self.is_scintillator():
            print(f"Warning: {self.material.generic_name} is not a scintillator")

        return 1 - self.get_transmission()

    def is_scintillator(self) -> bool:
        """Return whether the layer contains a scintillator.

        Returns:
            ``True`` if the layer material is a scintillator, otherwise ``False``.
        """
        return self.material.is_scintillator()

    def build_crosssections(self, energies: np.ndarray, engine: Engine) -> None:
        """Build cross-sections for the layer material.

        Args:
            energies: Energies at which to evaluate the cross-sections.
            engine: Cross-section engine to use.
        """
        self.material.build_crosssections(energies, engine)


class Spectrometer:
    """Simulate a Linear Absorption Spectrometer.

    Layers are added in their physical order along the spectrometer rail. 
    A response matrix is then built by propagating transmission through the stack and recording absorption in scintillator layers.

    Args:
        energies: incident energy grid.
        seperator: Optional separator layer inserted between added layers.
        measure_type: Default measurement mode. Supported values are ``"counts_clip"``, ``"counts"``, ``"counts_cont"``, and ``"energy_deposits"``.
        clip_range: Lower and upper clipping limits used by ``"counts_clip"``.
        build_engine: Cross-section engine passed to :class:`macr.engine.Engine`.
        layers: All layers in rail order, including inserted separators.
        scintillators: Active scintillator layers.
        seperator: Separator layer inserted between layers when requested.
        engine: Cross-section calculation engine.
        ENERGIES: Incident energy grid.
        response_matrix: Energy-deposition response matrix after construction.
        counts_matrix: Count response matrix after construction.
        z: Total rail thickness in millimetres.
        outputs: Number of scintillator outputs.
        name: Generated spectrometer design name.
        measure_type: Default measurement mode.
        clip_range: Lower and upper clipping limits.
    """

    def __init__(
        self,
        energies: np.ndarray,
        seperator: Optional[Layer] = None,
        measure_type: str = "counts",
        clip_range: tuple[int, int] = (0, None),
        build_engine: Literal["NIST", "SRIM", "ESTAR", "PSTAR"] = "NIST",
    ) -> None:
        """
        Args:
            energies: incident energy grid.
            seperator: Optional separator layer inserted between added layers.
            measure_type: Default measurement mode. Supported values are ``"counts_clip"``, ``"counts"``, ``"counts_cont"``, and ``"energy_deposits"``.
            clip_range: Lower and upper clipping limits used by ``"counts_clip"``.
            build_engine: Cross-section engine passed to :class:`macr.engine.Engine`.
        
        Raises:
            ValueError: If ``energies``, ``seperator``, or ``measure_type`` has an invalid type, or if ``measure_type`` is unsupported.
        """
        if not isinstance(energies, np.ndarray):
            raise ValueError("Energies must be a numpy array")
        if not isinstance(seperator, Layer) and seperator is not None:
            raise ValueError("Seperator must be a Layer")
        if not isinstance(measure_type, str):
            raise ValueError("Measure type must be a string")
        else:
            if measure_type not in [
                "counts_clip",
                "counts",
                "counts_cont",
                "energy_deposits",
            ]:
                raise ValueError(
                    "Measure type must be 'counts_clip', 'counts', "
                    "'counts_cont', or 'energy_deposits'"
                )

        self.layers: list[Layer] = []
        self.scintillators: list[Layer] = []
        self.seperator: Layer = seperator
        self.engine: Engine = Engine(build_engine)
        self.ENERGIES: np.ndarray = energies
        self.response_matrix: np.ndarray = None
        self.z: float = 0
        self.outputs: int = 0
        self.name: str = ""
        self.measure_type: str = measure_type
        self.clip_range: tuple[int] = clip_range

    def add_layer(self, layer: Layer, add_seperator: bool = True) -> None:
        """Add a layer to the spectrometer rail.

        When enabled, the configured separator is inserted before each layer after the first. 
        The total thickness and scintillator output attributes are updated at the same time.

        Args:
            layer: Layer to append.
            add_seperator: Whether to insert the configured separator first.
        """
        if add_seperator:
            if self.seperator is not None and len(self.layers) > 0:
                self.layers.append(self.seperator)
                self.z += self.seperator.thickness

        self.layers.append(layer)
        self.z += layer.thickness

        if layer.is_scintillator():
            self.outputs += 1
            self.scintillators.append(layer)

    def build_response(self) -> np.ndarray:
        """Build the spectrometer response matrix.

        Returns:
            Energy-deposition response matrix. Rows correspond to scintillator
            layers and columns correspond to values in :attr:`ENERGIES`.

        Raises:
            ValueError: If no layers have been added.

        Warns:
            UserWarning: If an existing response matrix is being rebuilt.
        """
        if len(self.layers) == 0:
            raise ValueError("Spectrometer has no layers")
        if self.response_matrix is not None:
            warnings.warn(
                "Response matrix already built. Rebuilding response matrix.",
                UserWarning,
            )

        ts = np.ones_like(self.ENERGIES)
        response_matrix = []
        counts_matrix = []
        for layer in self.layers:
            layer.build_crosssections(self.ENERGIES, self.engine)
            t1 = layer.get_transmission()
            if layer.is_scintillator():
                response_matrix.append(
                    layer.get_absorption() * ts * self.ENERGIES
                )
                counts_matrix.append(
                    response_matrix[-1] * layer.material.kappa
                )
            ts *= t1

        self.response_matrix = np.array(response_matrix)
        self.counts_matrix = np.array(counts_matrix)

        return self.response_matrix

    def finalise(self) -> None:
        """Finalise the spectrometer configuration.

        This builds the response matrix and generates the design name.
        """
        self.build_response()
        self.generate_name(self.seperator)

    def energy_deposits(self, spectrum: np.ndarray) -> np.ndarray:
        r"""Calculate deposited energy in each scintillator.

        The deposited energy is evaluated as

        .. math::

            E_{dep} = \int_{0}^{\infty} S(E_{in})\,\Gamma(E_{in})\,dE_{in}.

        Args:
            spectrum: Incident energy spectrum sampled on :attr:`ENERGIES`.

        Returns:
            Deposited energy for each scintillator output.

        Raises:
            ValueError: If the response matrix has not been built.
        """
        if self.response_matrix is None:
            raise ValueError("Response matrix not built")
        return np.trapezoid(
            spectrum * self.response_matrix,
            x=self.ENERGIES,
        )

    def counts(
        self,
        spectrum: np.ndarray,
        visible_lim: int = None,
        saturation_lim: list[int] = None,
    ) -> np.ndarray:
        """Calculate detector counts for each scintillator.

        Args:
            spectrum: Incident energy spectrum sampled on :attr:`ENERGIES`.
            visible_lim: Lower count limit. If omitted, the configured lower clipping limit is used.
            saturation_lim: Upper count limit. If omitted, the configured upper clipping limit is used.

        Returns:
            Count response for each scintillator.
        """
        if visible_lim is None:
            visible_lim = self.clip_range[0]
        if saturation_lim is None:
            saturation_lim = self.clip_range[1]

        energy_deposits = self.energy_deposits(spectrum)

        for i, scintillator_layer in enumerate(self.scintillators):
            energy_deposits[i] *= scintillator_layer.material.kappa

        return energy_deposits

    def measure(
        self,
        spectrum: np.ndarray,
        visible_lim: int = None,
        saturation_lim: int = None,
    ) -> np.ndarray:
        """Measure a spectrum using the configured measurement mode.

        Args:
            spectrum: Incident energy spectrum sampled on :attr:`ENERGIES`.
            visible_lim: Optional lower clipping limit for ``"counts_clip"``.
            saturation_lim: Optional upper clipping limit for ``"counts_clip"``.

        Returns:
            Measured detector response for the configured measurement mode.
        """
        if self.measure_type == "counts_clip":
            visible_lim = self.clip_range[0] if visible_lim is None else visible_lim
            saturation_lim = (
                self.clip_range[1]
                if saturation_lim is None
                else saturation_lim
            )
            return (
                self.counts(spectrum, visible_lim, saturation_lim)
                .clip(visible_lim, saturation_lim)
                .astype(int)
            )
        elif self.measure_type == "counts":
            return self.counts(spectrum).astype(int)
        elif self.measure_type == "counts_cont":
            return self.counts(spectrum)
        elif self.measure_type == "energy_deposits":
            return self.energy_deposits(spectrum)

    def unit(self, spectrum: np.ndarray) -> np.ndarray:
        """Return an unclipped detector response for a spectrum.
        Method is expected to receive a probability-like distribution.

        Args:
            spectrum: Incident energy spectrum sampled on :attr:`ENERGIES`.

        Returns:
            Count response for count-based modes, or deposited energy for ``"energy_deposits"`` mode.
        """
        if self.measure_type in {"counts_clip", "counts", "counts_cont"}:
            return self.counts(spectrum)
        elif self.measure_type == "energy_deposits":
            return self.energy_deposits(spectrum)

    def visualise_system(
        self,
        ax: plt.Axes = None,
        yoff: float = 0,
        legend: bool = True,
        legend_pos: str = "center left",
        bbox: tuple[float, float] = (1, 0.5),
        x_max: float = None,
        r_match: bool = False,
    ) -> None:
        """Visualise the spectrometer rail design.

        Args:
            ax: Matplotlib axes to draw on. A new figure and axes are created if omitted.
            yoff: Vertical offset for the rail drawing.
            legend: Whether to show the material legend.
            legend_pos: Matplotlib legend location.
            bbox: Bounding-box anchor supplied to the legend.
            x_max: Optional maximum x-axis value in millimetres.
            r_match: Whether to draw scintillator boundary lines.
        """
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        materials = [layer.material.symbol for layer in self.layers]
        unique_materials = [
            layer.material
            for idx, layer in enumerate(self.layers)
            if idx in np.unique(materials, return_index=True)[1]
        ]

        material_color_mapping = {}
        legendentry = []
        for idx, material in enumerate(unique_materials):
            material_color_mapping[material.symbol] = colors[idx]
            if material.is_scintillator():
                legendentry.append(
                    Patch(
                        facecolor=colors[idx],
                        edgecolor="y",
                        linewidth=5,
                        alpha=0.25,
                    )
                )
            else:
                legendentry.append(Patch(facecolor=colors[idx], alpha=0.25))

        if ax == None:
            fig, ax = plt.subplots(1, 1)

        x_left = 0
        for sc, layer in enumerate(self.layers):
            y_down = 0 + yoff
            y_up = 1 + yoff
            x_right = x_left + layer.thickness

            xs = [x_left, x_right]
            y1s = [y_down, y_down]
            y2s = [y_up, y_up]

            ax.fill_between(
                xs,
                y1s,
                y2s,
                facecolor=material_color_mapping[layer.material.symbol],
                alpha=0.25,
            )

            if layer.is_scintillator() and r_match:
                ax.plot(xs, y1s, c=colors[sc % len(colors)])
                ax.plot(xs, y2s, c=colors[sc % len(colors)])

            x_left = x_right

        ax.set_xlabel("z (mm)")
        ax.set_yticks([])

        if legend:
            ax.legend(
                legendentry,
                material_color_mapping,
                loc=legend_pos,
                bbox_to_anchor=bbox,
            )
            for text in ax.get_legend().get_texts():
                text.set_fontsize("small")

        if x_max is not None:
            ax.set_xlim([0, x_max])

    def visualise_response(
        self,
        ax: plt.axes = None,
        response: str = "Edep",
        xlim: Any = None,
        ylim: Any = None,
        xscale: str = "log",
        yscale: str = "log",
    ) -> None:
        """Visualise the spectrometer response curves.

        Args:
            ax: Matplotlib axes to draw on. A new figure and axes are created if
                omitted.
            response: ``"Edep"`` to plot deposited-energy response; any other
                value plots count response, preserving the original behaviour.
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.
            xscale: Matplotlib x-axis scale.
            yscale: Matplotlib y-axis scale.
        """
        ax: plt.Axes
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        for i, res in enumerate(self.response_matrix):
            if response == "Edep":
                ax.plot(self.ENERGIES, res)
            else:
                ax.plot(self.ENERGIES, self.counts_matrix[i])
        if response == "Edep":
            ax.plot(self.ENERGIES, self.ENERGIES, "k--")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)

        ax.set_ylabel(
            r"$E_{dep}/\gamma$"
            if response == "Edep"
            else r"$\text{Counts}/\gamma$"
        )
        ax.xaxis.tick_top()
        ax.set_xlabel("Energy(MeV)")
        ax.xaxis.set_label_position("top")

    def visualise(
        self,
        figsize: tuple[float, float] = (18, 6),
        xlim: Any = None,
        edep_ylim: Any = None,
        counts_ylim: Any = None,
        xscale: str = "log",
        edep_yscale: str = "log",
        counts_yscale: str = "log",
        yoff: float = 0,
        legend: bool = True,
        legend_pos: str = "center left",
        bbox: tuple[float, float] = (1, 0.5),
        x_max: float = None,
    ) -> None:
        """Visualise response curves and the spectrometer rail together.

        Args:
            figsize: Figure size passed to Matplotlib.
            xlim: Optional shared response x-axis limits.
            edep_ylim: Optional deposited-energy y-axis limits.
            counts_ylim: Optional counts y-axis limits.
            xscale: Response x-axis scale.
            edep_yscale: Deposited-energy y-axis scale.
            counts_yscale: Counts y-axis scale.
            yoff: Vertical offset for the rail drawing.
            legend: Whether to show the rail material legend.
            legend_pos: Matplotlib legend location.
            bbox: Bounding-box anchor supplied to the legend.
            x_max: Optional maximum rail x-axis value in millimetres.
        """
        fig = plt.figure(figsize=figsize)
        ax = [None] * 3
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[4, 1])
        ax[0] = fig.add_subplot(gs[0, 0])
        ax[1] = fig.add_subplot(gs[0, 1])
        ax[2] = fig.add_subplot(gs[1, :])

        self.visualise_response(
            ax[0],
            response="Edep",
            ylim=edep_ylim,
            xscale=xscale,
            yscale=edep_yscale,
        )
        self.visualise_response(
            ax[1],
            response="Counts",
            ylim=counts_ylim,
            xscale=xscale,
            yscale=counts_yscale,
        )
        self.visualise_system(
            ax[2],
            yoff=yoff,
            legend=legend,
            legend_pos=legend_pos,
            bbox=bbox,
            x_max=x_max,
        )

        ax[0].set_title("Spectrometer Response ($E_{dep}$)", fontsize=14)
        ax[1].set_title("Spectrometer Response (Counts)", fontsize=14)
        ax[2].set_title("Spectrometer Rail Design", fontsize=14)

        fig.suptitle("Design: " + self.name, fontsize=16)
        fig.tight_layout()
        plt.show()

    def generate_name(self, separator: Layer = None) -> None:
        """Generate and store a compact name for the spectrometer design.

        Args:
            separator: Separator layer to exclude from the generated name. If
                omitted, :attr:`seperator` is used by the existing logic.
        """
        names = []
        materialsandlengths = list(
            zip(
                [layer.material.symbol for layer in self.layers],
                [layer.thickness for layer in self.layers],
            )
        )

        if separator is not None or self.seperator is not None:
            while (separator.material.symbol, separator.thickness) in materialsandlengths:
                materialsandlengths.remove(
                    (separator.material.symbol, separator.thickness)
                )

        materialsandlengths_unique = list(set(materialsandlengths))
        materialsandlengths_unique = sorted(
            materialsandlengths_unique,
            key=lambda x: materialsandlengths.index(x),
        )

        for m in materialsandlengths_unique:
            indexes = [i for i, x in enumerate(materialsandlengths) if x == m]
            names += [f"{','.join([str(i) for i in indexes])}-{m[0]}_{m[1]}"]

        self.name = " - ".join(names)

    def get_name_in_lines(self, separator: Layer = None) -> str:
        """Return the generated design name with one material per line.

        Args:
            separator: Optional separator layer to exclude from the generated
                name.

        Returns:
            Multi-line design name.
        """
        names = []
        materialsandlengths = list(
            zip(
                [layer.material.symbol for layer in self.layers],
                [layer.thickness for layer in self.layers],
            )
        )

        if separator is not None:
            while (separator.material.symbol, separator.thickness) in materialsandlengths:
                materialsandlengths.remove(
                    (separator.material.symbol, separator.thickness)
                )

        materialsandlengths_unique = list(set(materialsandlengths))
        materialsandlengths_unique = sorted(
            materialsandlengths_unique,
            key=lambda x: materialsandlengths.index(x),
        )

        for m in materialsandlengths_unique:
            indexes = [i for i, x in enumerate(materialsandlengths) if x == m]
            names += [f"{','.join([str(i) for i in indexes])}-{m[0]}_{m[1]}"]

        return "\n".join(names)

    def __call__(self, *args: Any, **kwds: Any) -> np.ndarray:
        """Delegate a call to :meth:`measure`.

        Args:
            *args: Positional arguments forwarded to :meth:`measure`.
            **kwds: Keyword arguments forwarded to :meth:`measure`.

        Returns:
            Result returned by :meth:`measure`.
        """
        return self.measure(*args, **kwds)

    def __getitem__(self, index: int) -> Layer:
        """Return a layer by index.

        Args:
            index: Index into :attr:`layers`.

        Returns:
            Layer at ``index``.
        """
        return self.layers[index]