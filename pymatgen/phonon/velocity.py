"""
This module contains classes to handle and to plot phonon
group velocities (both on regular grid and along special symmetry paths).
"""

from __future__ import annotations

import numpy as np
from monty.json import MSONable

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.phonon.bandstructure import (
    PhononBandStructure,
    PhononBandStructureSymmLine,
)
from pymatgen.phonon.dos import PhononDos

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
        # self.band_reorder()

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
