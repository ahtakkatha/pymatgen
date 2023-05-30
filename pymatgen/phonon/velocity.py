"""
This module contains classes to handle and to plot phonon
group velocities (both on regular grid and along special symmetry paths).
"""

from __future__ import annotations

import numpy as np
from monty.json import MSONable

from pymatgen.core import Lattice, Structure
from pymatgen.core.lattice import Lattice
from pymatgen.phonon.bandstructure import (
    PhononBandStructure,
    PhononBandStructureSymmLine,
)
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, freq_units
from pymatgen.util.plotting import pretty_plot

try:
    import phonopy
    from phonopy.phonon.dos import TotalDos
except ImportError as ex:
    print(ex)
    phonopy = None


class Velocity(MSONable):
    """
    Class for group velocities on a regular grid.
    """

    def __init__(
        self,
        qpoints,
        velocities,
        frequencies,
        multiplicities=None,
        structure=None,
        lattice=None,
    ):
        """
        Args:
            qpoints: List of qpoints as numpy arrays, in frac_coords of the given lattice by default.
            velocities: List of absolute group velocities as numpy arrays, shape: (3*len(structure), len(qpoints)).
            frequencies: List of phonon frequencies in THz as a numpy array with shape (3*len(structure), len(qpoints)).
            multiplicities: List of multiplicities.
            structure: The crystal structure (as a pymatgen Structure object) associated with the velocities.
            lattice: The reciprocal lattice as a pymatgen Lattice object. Pymatgen uses the physics convention of
                     reciprocal lattice vectors WITH a 2*pi coefficient.
        """
        self.qpoints = qpoints
        self.velocities = velocities
        self.frequencies = frequencies
        self.multiplicities = multiplicities
        self.lattice = lattice
        self.structure = structure

    @property  # type: ignore
    def tdos(self):
        """
        The total DOS (re)constructed from the mesh.yaml file
        """

        # Here, we will reuse phonopy classes
        class TempMesh:
            """
            Temporary Class
            """

        a = TempMesh()
        a.frequencies = np.transpose(self.frequencies)
        a.weights = self.multiplicities

        b = TotalDos(a)
        b.run()

        return b

    @property
    def phdos(self):
        """
        Returns: PhononDos object
        """
        return PhononDos(self.tdos.frequency_points, self.tdos.dos)


class VelocityPlotter:
    """
    Class to plot Velocity object (group velocities on regular grid)
    with matplotlib.
    """

    def __init__(self, velocity):
        """
        Args:
            velocity: Velocity Object.
        """
        self._velocity = velocity

    def get_plot(self, marker="o", markersize=6, color_q_point=True, units="thz", vel_mode="amount"):
        """
        Get a matplotlib plot showing velocity vs. frequency.
        Color code refers to.
        Args:
            marker: Marker for the depiction.
            markersize: Size of the marker.
            color_q_point: Whether to color-code the irreducible q-points.
            units: Unit for the plots, accepted units: thz, ev, mev, ha, cm-1, cm^-1.
            vel_mode: String argument whether to plot the amount or single components
                of velocity vector. Accepted: "amount" (default), "a", "b", "c".

        Returns: A matplotlib.pyplot plot object.
        """
        allowed_modes = {
            "amount": lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2),
            "a": lambda x: x[0],
            "b": lambda x: x[1],
            "c": lambda x: x[2],
        }
        if vel_mode not in allowed_modes.keys():
            raise ValueError(f"\nParameter vel_mode must be in {allowed_modes}.\n")

        u = freq_units(units)

        # Retranspose for q point-wise plotting
        xs = self._velocity.frequencies.transpose()
        ys = np.apply_along_axis(allowed_modes[vel_mode], 2, self._velocity.velocities)
        ys = ys.transpose()
        plt = pretty_plot(12, 8)

        plt.xlabel(rf"$\mathrm{{Frequency\ ({u.label})}}$")
        plt.ylabel(r"$\mathrm{Velocity}$")

        for x, y in zip(xs, ys):
            x = x * u.factor
            if color_q_point:
                # TODO: better to implement colors fr. (0, 0, 1) to (1, 0, 0)? remove yellow from palette?
                plt.plot(x, y, marker, markersize=markersize)
            else:
                plt.plot(x, y, marker, color="#1f77b4", markersize=markersize)

        plt.tight_layout()

        return plt

    def show(self, units="thz", vel_mode="amount"):
        """
        Show the plot using matplotlib.
        Args:
            units: Units for the plot, accepted units: thz, ev, mev, ha, cm-1, cm^-1.
            vel_mode: String argument whether to plot the amount or single components
                of velocity vector. Accepted: "amount" (default), "a", "b", "c".
        """
        plt = self.get_plot(units=units, vel_mode=vel_mode)
        plt.show()

    def save_plot(self, filename, img_format="pdf", units="thz", vel_mode="amount"):
        """
        Saves the plot to a file.
        Args:
            filename: Name of the filename.
            img_format: Format of the saved plot.
            units: Accepted units: thz, ev, mev, ha, cm-1, cm^-1.
            vel_mode: String argument whether to plot the amount or single components
                of velocity vector. Accepted: "amount" (default), "a", "b", "c".
        """
        plt = self.get_plot(units=units, vel_mode=vel_mode)
        plt.savefig(filename, format=img_format)
        plt.close()


class VelocityPhononBandStructure(PhononBandStructure):
    """
    As the generic PhononBandStructure, this class is defined by
    a list of qpoints and frequencies for each of them
    extended by group velocity parameters for each of them.
    Additional information may be given for frequencies at Gamma, where
    non-analytical contribution may be taken into account.
    """

    def __init__(
        self,
        qpoints,
        frequencies,
        velocities,
        lattice,
        eigendisplacements=None,
        labels_dict=None,
        coords_are_cartesian=False,
        structure=None,
    ):
        """
        Args:
            qpoints: list of qpoint as numpy arrays, in frac_coords of the
                given lattice by default
            frequencies: list of phonon frequencies in THz as a numpy array with shape
                (3*len(structure), len(qpoints)). The First index of the array
                refers to the band and the second to the index of the qpoint.
            velocities: list of group velocity parameters with the same structure
                frequencies.
            lattice: The reciprocal lattice as a pymatgen Lattice object.
                Pymatgen uses the physics convention of reciprocal lattice vectors
                WITH a 2*pi coefficient.
            eigendisplacements: the phonon eigendisplacements associated to the
                frequencies in Cartesian coordinates. A numpy array of complex
                numbers with shape (3*len(structure), len(qpoints), len(structure), 3).
                The first index of the array refers to the band, the second to the index
                of the qpoint, the third to the atom in the structure and the fourth
                to the Cartesian coordinates.
            labels_dict: (dict) of {} this links a qpoint (in frac coords or
                Cartesian coordinates depending on the coords) to a label.
            coords_are_cartesian: Whether the qpoint coordinates are Cartesian.
            structure: The crystal structure (as a pymatgen Structure object)
                associated with the band structure. This is needed if we
                provide projections to the band structure
        """
        PhononBandStructure.__init__(
            self,
            qpoints,
            frequencies,
            lattice,
            nac_frequencies=None,
            eigendisplacements=eigendisplacements,
            nac_eigendisplacements=None,
            labels_dict=labels_dict,
            coords_are_cartesian=coords_are_cartesian,
            structure=structure,
        )
        self.velocities = velocities

    def as_dict(self):
        """
        Returns:
            MSONable (dict)
        """
        d = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "lattice_rec": self.lattice_rec.as_dict(),
            "qpoints": [],
        }
        # qpoints are not Kpoint objects dicts but are frac coords. This makes
        # the dict smaller and avoids the repetition of the lattice
        for q in self.qpoints:
            d["qpoints"].append(q.as_dict()["fcoords"])
        d["bands"] = self.bands.tolist()
        d["labels_dict"] = {}
        for kpoint_letter, kpoint_object in self.labels_dict.items():
            d["labels_dict"][kpoint_letter] = kpoint_object.as_dict()["fcoords"]
        # split the eigendisplacements to real and imaginary part for serialization
        d["eigendisplacements"] = {
            "real": np.real(self.eigendisplacements).tolist(),
            "imag": np.imag(self.eigendisplacements).tolist(),
        }
        d["velocities"] = self.velocities.tolist()
        if self.structure:
            d["structure"] = self.structure.as_dict()

        return d

    @classmethod
    def from_dict(cls, d):
        """
        Args:
            d (dict): Dict representation
        Returns:
            A VelocityPhononBandStructure object: Phonon band structure with Velocity parameters.
        """
        lattice_rec = Lattice(d["lattice_rec"]["matrix"])
        eigendisplacements = np.array(d["eigendisplacements"]["real"]) + np.array(d["eigendisplacements"]["imag"]) * 1j
        structure = Structure.from_dict(d["structure"]) if "structure" in d else None
        return cls(
            qpoints=d["qpoints"],
            frequencies=np.array(d["bands"]),
            velocities=np.array(d["velocities"]),
            lattice=lattice_rec,
            eigendisplacements=eigendisplacements,
            labels_dict=d["labels_dict"],
            structure=structure,
        )


class VelocityPhononBandStructureSymmLine(VelocityPhononBandStructure, PhononBandStructureSymmLine):
    """
    This object stores a VelocityPhononBandStructureSymmLine together with group velocity
    for every frequency.
    """

    def __init__(
        self,
        qpoints,
        frequencies,
        velocities,
        lattice,
        eigendisplacements=None,
        labels_dict=None,
        coords_are_cartesian=False,
        structure=None,
    ):
        """
        Args:
            qpoints: list of qpoints as numpy arrays, in frac_coords of the
                given lattice by default
            frequencies: list of phonon frequencies in eV as a numpy array with shape
                (3*len(structure), len(qpoints))
            velocities: list of absolute velocities as a numpy array with the
                shape (3*len(structure), len(qpoints))
            lattice: The reciprocal lattice as a pymatgen Lattice object.
                Pymatgen uses the physics convention of reciprocal lattice vectors
                WITH a 2*pi coefficient
            eigendisplacements: the phonon eigendisplacements associated to the
                frequencies in Cartesian coordinates. A numpy array of complex
                numbers with shape (3*len(structure), len(qpoints), len(structure), 3).
                The first index of the array refers to the band, the second to the index
                of the qpoint, the third to the atom in the structure and the fourth
                to the Cartesian coordinates.
            labels_dict: (dict) of {} this links a qpoint (in frac coords or
                Cartesian coordinates depending on the coords) to a label.
            coords_are_cartesian: Whether the qpoint coordinates are cartesian.
            structure: The crystal structure (as a pymatgen Structure object)
                associated with the band structure. This is needed if we
                provide projections to the band structure
        """
        VelocityPhononBandStructure.__init__(
            self,
            qpoints=qpoints,
            frequencies=frequencies,
            velocities=velocities,
            lattice=lattice,
            eigendisplacements=eigendisplacements,
            labels_dict=labels_dict,
            coords_are_cartesian=coords_are_cartesian,
            structure=structure,
        )

        PhononBandStructureSymmLine._reuse_init(
            self, eigendisplacements=eigendisplacements, frequencies=frequencies, has_nac=False, qpoints=qpoints
        )

    @classmethod
    def from_dict(cls, d):
        """
        Args:
            d: Dict representation
        Returns: A VelocityPhononBandStructureSymmLine object.
        """
        lattice_rec = Lattice(d["lattice_rec"]["matrix"])
        eigendisplacements = np.array(d["eigendisplacements"]["real"]) + np.array(d["eigendisplacements"]["imag"]) * 1j
        structure = Structure.from_dict(d["structure"]) if "structure" in d else None
        return cls(
            qpoints=d["qpoints"],
            frequencies=np.array(d["bands"]),
            velocities=np.array(d["velocities"]),
            lattice=lattice_rec,
            eigendisplacements=eigendisplacements,
            labels_dict=d["labels_dict"],
            structure=structure,
        )


class VelocityPhononBSPlotter(PhononBSPlotter):
    """
    Class to plot or get data to facilitate the plot of VelocityPhononBandStructureSymmLine objects.
    """

    def __init__(self, bs):
        """
        Args:
            bs: A VelocityPhononBandStructureSymmLine object.
        """
        if not isinstance(bs, VelocityPhononBandStructureSymmLine):
            raise ValueError(
                "VelocityPhononBSPlotter only works with VelocityPhononBandStructureSymmLine objects. "
                "A VelocityPhononBandStructure object (on a uniform grid for instance and "
                "not along symmetry lines won't work)"
            )
        super().__init__(bs)

    def bs_plot_data(self):
        """
        Get the data nicely formatted for a plot.
        Returns:
            A dict of the following format:
            ticks: A dict with the 'distances' at which there is a qpoint (the
            x axis) and the labels (None if no label)
            frequencies: A list (one element for each branch) of frequencies for
            each qpoint: [branch][qpoint][mode]. The data is
            stored by branch to facilitate the plotting
            velocity: VelocityPhononBandStructureSymmLine
            lattice: The reciprocal lattice.
        """
        distance, frequency, velocity = ([] for _ in range(3))

        ticks = self.get_ticks()

        for b in self._bs.branches:
            frequency.append([])
            velocity.append([])
            # TODO compare band.yaml!
            # TODO do band reorder to match modes
            distance.append([self._bs.distance[j] for j in range(b["start_index"], b["end_index"] + 1)])

            for i in range(self._nb_bands):
                frequency[-1].append([self._bs.bands[i][j] for j in range(b["start_index"], b["end_index"] + 1)])
                velocity[-1].append([self._bs.velocities[i][j] for j in range(b["start_index"], b["end_index"] + 1)])

        return {
            "ticks": ticks,
            "distances": distance,
            "frequency": frequency,
            "velocity": velocity,
            "lattice": self._bs.lattice_rec.as_dict(),
        }

    def get_plot_velocity_bs(self, ylim=None, only_bands=None):
        """
        Get a matplotlib.pyplot object for the velocity bandstructure plot.
        Args:
            ylim: Specify the y-axis (velocity) limits. Defaults to None
                for automatic determination.
            only_bands: List to specify which bands to plot, starts at 0.
        """
        plt = pretty_plot(12, 8)
        if only_bands is None:
            only_bands = range(self._nb_bands)

        data = self.bs_plot_data()
        for d in range(len(data["distances"])):
            for i in only_bands:
                plt.plot(
                    data["distances"][d],
                    [data["velocity"][d][i][j] for j in range(len(data["distances"][d]))],
                    "-",
                    marker="o",
                    markersize=1,
                    linewidth=1,
                )

        self._maketicks(plt)

        # plot y=0 line
        plt.axhline(0, linewidth=1, color="k")

        # Main X and Y Labels
        plt.xlabel(r"$\mathrm{Wave\ Vector}$", fontsize=30)
        plt.ylabel(r"$\mathrm{Group\ Velocity}$", fontsize=30)

        # X range (K)
        # last distance point
        x_max = data["distances"][-1][-1]
        plt.xlim(0, x_max)

        if ylim:
            plt.ylim(ylim)
        # TODO add automatic ylim determination?

        plt.tight_layout()

        return plt

    def show_velocity_bs(self, ylim=None, only_bands=None):
        """
        Show the plot using matplotlib.
        Args:
            ylim: Specify the y-axis (velocity) limits. Defaults to None
                for automatic determination.
            only_bands: List to specify which bands to plot, starts at 0.
        """
        plt = self.get_plot_velocity_bs(ylim, only_bands=only_bands)
        plt.show()

    def save_plot_velocity_bs(self, filename, img_format="eps", ylim=None, only_bands=None):
        """
        Save matplotlib plot to a file.
        Args:
            filename: Filename to write to.
            img_format: Image format to use. Defaults to EPS.
            ylim: Specify the y-axis (velocity) limits. Defaults to None
                for automatic determination.
            only_bands: List to specify which bands to plot, starts at 0.
        """
        plt = self.get_plot_velocity_bs(ylim=ylim, only_bands=only_bands)
        plt.savefig(filename, format=img_format)
        plt.close()
