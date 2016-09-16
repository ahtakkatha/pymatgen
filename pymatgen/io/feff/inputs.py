# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals

import re
import warnings
from operator import itemgetter
from six import string_types
from tabulate import tabulate

from monty.io import zopen
from monty.json import MSONable

from pymatgen import Structure, Lattice, Element
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string_utils import str_delimited

"""
This module defines classes for reading/manipulating/writing the main sections
of FEFF input file(feff.inp), namely HEADER, ATOMS, POTENTIAL and the program
control tags.

XANES and EXAFS input files, are available, for non-spin case at this time.
"""

__author__ = "Alan Dozier"
__credits__ = "Anubhav Jain, Shyue Ping Ong"
__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "1.0.3"
__maintainer__ = "Alan Dozier"
__email__ = "adozier@uky.edu"
__status__ = "Beta"
__date__ = "April 7, 2013"


# **Non-exhaustive** list of valid Feff.inp tags
VALID_FEFF_TAGS = ("CONTROL", "PRINT", "ATOMS", "POTENTIALS", "RECIPROCAL",
                   "REAL", "MARKER", "LATTICE", "TITLE", "RMULTIPLIER",
                   "SGROUP", "COORDINATES", "EQUIVALENCE", "CIF", "CGRID",
                   "CFAVERAGE", "OVERLAP", "EXAFS", "XANES", "ELNES", "EXELFS",
                   "LDOS", "ELLIPTICITY", "MULTIPOLE", "POLARIZATION",
                   "RHOZZP", "DANES", "FPRIME", "NRIXS", "XES", "XNCD",
                   "XMCD", "XNCDCONTROL", "END", "KMESH", "PRINT", "EGRID",
                   "DIMS", "AFOLP", "EDGE", "COMPTON", "DANES",
                   "FPRIME" "MDFF", "HOLE", "COREHOLE", "S02", "CHBROAD",
                   "EXCHANGE", "FOLP", "NOHOLE", "RGRID", "SCF",
                   "UNFREEZEF", "CHSHIFT", "DEBYE",
                   "INTERSTITIAL", "CHWIDTH", "EGAP", "EPS0", "EXTPOT",
                   "ION", "JUMPRM", "EXPOT", "SPIN", "LJMAX", "LDEC", "MPSE",
                   "PLASMON", "RPHASES", "RSIGMA", "PMBSE", "TDLDA", "FMS",
                   "DEBYA", "OPCONS", "PREP", "RESTART", "SCREEN", "SETE",
                   "STRFACTORS", "BANDSTRUCTURE", "RPATH", "NLEG", "PCRITERIA",
                   "SYMMETRY", "SS", "CRITERIA", "IORDER", "NSTAR", "ABSOLUTE",
                   "CORRECTIONS", "SIG2", "SIG3", "MBCONV", "SFCONV", "RCONV",
                   "SELF", "SFSE", "MAGIC")


class Header(MSONable):
    """
    Creates Header for the FEFF input file.

    Has the following format::

        * This feff.inp file generated by pymatgen, www.materialsproject.org
        TITLE comment:
        TITLE Source: CoO19128.cif
        TITLE Structure Summary: (Co2 O2)
        TITLE Reduced formula: CoO
        TITLE space group: P1,   space number: 1
        TITLE abc: 3.297078 3.297078 5.254213
        TITLE angles: 90.0 90.0 120.0
        TITLE sites: 4
        * 1 Co     0.666666     0.333332     0.496324
        * 2 Co     0.333333     0.666667     0.996324
        * 3 O     0.666666     0.333332     0.878676
        * 4 O     0.333333     0.666667     0.378675

    Args:
        struct: Structure object, See pymatgen.core.structure.Structure.
        source: User supplied identifier, i.e. for Materials Project this
            would be the material ID number
        comment: Comment for first header line
    """

    def __init__(self, struct, source='', comment=''):
        if struct.is_ordered:
            self.struct = struct
            self.source = source
            sym = SpacegroupAnalyzer(struct)
            data = sym.get_symmetry_dataset()
            self.space_number = data["number"]
            self.space_group = data["international"]
            self.comment = comment or "None given"
        else:
            raise ValueError("Structure with partial occupancies cannot be "
                             "converted into atomic coordinates!")

    @staticmethod
    def from_cif_file(cif_file, source='', comment=''):
        """
        Static method to create Header object from cif_file

        Args:
            cif_file: cif_file path and name
            source: User supplied identifier, i.e. for Materials Project this
                would be the material ID number
            comment: User comment that goes in header

        Returns:
            Header Object
        """
        r = CifParser(cif_file)
        structure = r.get_structures()[0]
        return Header(structure, source, comment)

    @property
    def structure_symmetry(self):
        """
        Returns space number and space group

        Returns:
            Space number and space group list
        """
        return self.space_group, self.space_number

    @property
    def formula(self):
        """
        Formula of structure
        """
        return self.struct.composition.formula

    @staticmethod
    def from_file(filename):
        """
        Returns Header object from file
        """
        hs = Header.header_string_from_file(filename)
        return Header.from_string(hs)

    @staticmethod
    def header_string_from_file(filename='feff.inp'):
        """
        Reads Header string from either a HEADER file or feff.inp file
        Will also read a header from a non-pymatgen generated feff.inp file

        Args:
            filename: File name containing the Header data.

        Returns:
            Reads header string.
        """
        with zopen(filename, "r") as fobject:
            f = fobject.readlines()
            feff_header_str = []
            ln = 0

            # Checks to see if generated by pymatgen
            try:
                feffpmg = f[0].find("pymatgen")
            except IndexError:
                feffpmg = False

            # Reads pymatgen generated header or feff.inp file
            if feffpmg:
                nsites = int(f[8].split()[2])
                for line in f:
                    ln += 1
                    if ln <= nsites + 9:
                        feff_header_str.append(line)
            else:
                # Reads header from header from feff.inp file from unknown
                # source
                end = 0
                for line in f:
                    if (line[0] == "*" or line[0] == "T") and end == 0:
                        feff_header_str.append(line.replace("\r", ""))
                    else:
                        end = 1

        return ''.join(feff_header_str)

    @staticmethod
    def from_string(header_str):
        """
        Reads Header string and returns Header object if header was
        generated by pymatgen.
        Note: Checks to see if generated by pymatgen, if not it is impossible
            to generate structure object so it is not possible to generate
            header object and routine ends

        Args:
            header_str: pymatgen generated feff.inp header

        Returns:
            Structure object.
        """
        lines = tuple(clean_lines(header_str.split("\n"), False))
        comment1 = lines[0]
        feffpmg = comment1.find("pymatgen")

        if feffpmg:
            comment2 = ' '.join(lines[1].split()[2:])

            source = ' '.join(lines[2].split()[2:])
            basis_vec = lines[6].split(":")[-1].split()
            # a, b, c
            a = float(basis_vec[0])
            b = float(basis_vec[1])
            c = float(basis_vec[2])
            lengths = [a, b, c]
            # alpha, beta, gamma
            basis_ang = lines[7].split(":")[-1].split()
            alpha = float(basis_ang[0])
            beta = float(basis_ang[1])
            gamma = float(basis_ang[2])
            angles = [alpha, beta, gamma]

            lattice = Lattice.from_lengths_and_angles(lengths, angles)

            natoms = int(lines[8].split(":")[-1].split()[0])

            atomic_symbols = []
            for i in range(9, 9 + natoms):
                atomic_symbols.append(lines[i].split()[2])

            # read the atomic coordinates
            coords = []
            for i in range(natoms):
                toks = lines[i + 9].split()
                coords.append([float(s) for s in toks[3:]])

            struct = Structure(lattice, atomic_symbols, coords, False,
                                        False, False)

            h = Header(struct, source, comment2)

            return h
        else:
            return "Header not generated by pymatgen, cannot return header object"

    def __str__(self):
        """
        String representation of Header.
        """
        to_s = lambda x: "%0.6f" % x
        output = ["* This FEFF.inp file generated by pymatgen",
                  ''.join(["TITLE comment: ", self.comment]),
                  ''.join(["TITLE Source:  ", self.source]),
                  "TITLE Structure Summary:  {}"
                  .format(self.struct.composition.formula),
                  "TITLE Reduced formula:  {}"
                  .format(self.struct.composition.reduced_formula),
                  "TITLE space group: ({}), space number:  ({})"
                  .format(self.space_group, self.space_number),
                  "TITLE abc:{}".format(" ".join(
                      [to_s(i).rjust(10) for i in self.struct.lattice.abc])),
                  "TITLE angles:{}".format(" ".join(
                      [to_s(i).rjust(10) for i in self.struct.lattice.angles])),
                  "TITLE sites: {}".format(self.struct.num_sites)]
        for i, site in enumerate(self.struct):
            output.append(" ".join(["*", str(i + 1), site.species_string,
                                    " ".join([to_s(j).rjust(12)
                                              for j in site.frac_coords])]))
        return "\n".join(output)

    def write_file(self, filename='HEADER'):
        """
        Writes Header into filename on disk.

        Args:
            filename: Filename and path for file to be written to disk
        """
        with open(filename, "w") as f:
            f.write(str(self) + "\n")


class Atoms(MSONable):
    """
    Atomic cluster centered around the absorbing atom.
    """

    def __init__(self, struct, central_atom, radius):
        """
        Args:
            struct (Structure): input structure
            central_atom (str): Symbol for absorbing atom
            radius (float): radius of the atom cluster in Angstroms.
        """
        self.central_atom = central_atom
        self.radius = radius
        if struct.is_ordered:
            self.struct = struct
            self.pot_dict = get_atom_map(struct)
        else:
            raise ValueError("Structure with partial occupancies cannot be "
                             "converted into atomic coordinates!")

    @staticmethod
    def atoms_string_from_file(filename):
        """
        Reads atomic shells from file such as feff.inp or ATOMS file
        The lines are arranged as follows:

        x y z   ipot    Atom Symbol   Distance   Number

        with distance being the shell radius and ipot an integer identifying
        the potential used.

        Args:
            filename: File name containing atomic coord data.

        Returns:
            Atoms string.
        """
        with zopen(filename, "rt") as fobject:
            f = fobject.readlines()
            coords = 0
            atoms_str = []

            for line in f:
                if coords == 0:
                    find_atoms = line.find("ATOMS")
                    if find_atoms >= 0:
                        coords = 1
                if coords == 1:
                    atoms_str.append(line.replace("\r", ""))

        return ''.join(atoms_str)

    @staticmethod
    def from_string(data):
        """
        At the moment does nothing.

        From atoms string data generates atoms object
        """
        return data

    def get_string(self):
        """
        Returns a string representation of atomic shell coordinates.

        Returns:
            String representation of Atomic Coordinate Shells.
        """
        center_index = self.struct.indices_from_symbol(self.central_atom)[0]
        center = self.struct[center_index].coords
        sphere = self.struct.get_neighbors(self.struct[center_index], self.radius)

        row = [[str(0), str(0), str(0),  str(0), self.central_atom, str(0), str(0)]]
        for i, site_dist in enumerate(sphere):
            site_symbol = re.sub(r"[^aA-zZ]+", "", site_dist[0].species_string)
            ipot = self.pot_dict[site_symbol]
            coords = site_dist[0].coords - center
            row.append(["{:f}".format(coords[0]), "{:f}".format(coords[1]),
                        "{:f}".format(coords[2]), ipot, site_symbol,
                        "{:f}".format(site_dist[1]), i+1])

        row_sorted = str(tabulate(sorted(row, key=itemgetter(5)),
                                  headers=["*       x", "y", "z", "ipot",
                                           "Atom", "Distance", "Number"]))
        atom_list = row_sorted.replace("--", "**")

        return ''.join(["ATOMS\n", atom_list, "\nEND\n"])

    def __str__(self):
        """
        String representation of Atoms file.
        """
        return self.get_string()

    def write_file(self, filename='ATOMS'):
        """
        Write Atoms list to file.

        Args:
           filename: path for file to be written
        """
        with zopen(filename, "wt") as f:
            f.write(str(self) + "\n")


class Tags(dict):
    """
    FEFF control parameters.
    """

    def __init__(self, params=None):
        """
        Args:
            params: A set of input parameters as a dictionary.
        """
        super(Tags, self).__init__()
        if params:
            self.update(params)

    def __setitem__(self, key, val):
        """
        Add parameter-val pair.  Warns if parameter is not in list of valid
        Feff tags. Also cleans the parameter and val by stripping leading and
        trailing white spaces.

        Arg:
            key: dict key value
            value: value associated with key in dictionary
        """
        if key.strip().upper() not in VALID_FEFF_TAGS:
            warnings.warn(key.strip() + " not in VALID_FEFF_TAGS list")
        super(Tags, self).__setitem__(key.strip(),
                                      Tags.proc_val(key.strip(), val.strip())
                                      if isinstance(val, string_types) else val)

    def as_dict(self):
        """
        Dict representation.

        Returns:
            Dictionary of parameters from fefftags object
        """
        tags_dict = dict(self)
        tags_dict['@module'] = self.__class__.__module__
        tags_dict['@class'] = self.__class__.__name__
        return tags_dict

    @staticmethod
    def from_dict(d):
        """
        Creates Tags object from a dictionary.

        Args:
            d: Dict of feff parameters and values.

        Returns:
            Tags object
        """
        i = Tags()
        for k, v in d.items():
            if k not in ("@module", "@class"):
                i[k] = v
        return i

    def get_string(self, sort_keys=False, pretty=False):
        """
        Returns a string representation of the Tags.  The reason why this
        method is different from the __str__ method is to provide options
        for pretty printing.

        Args:
            sort_keys: Set to True to sort the Feff parameters alphabetically.
                Defaults to False.
            pretty: Set to True for pretty aligned output. Defaults to False.

        Returns:
            String representation of Tags.
        """
        keys = self.keys()
        if sort_keys:
            keys = sorted(keys)
        lines = []
        for k in keys:
            if isinstance(self[k], list):
                lines.append([k, " ".join([str(i) for i in self[k]])])
            else:
                lines.append([k, self[k]])

        if pretty:
            return tabulate(lines)
        else:
            return str_delimited(lines, None, "  ")

    def __str__(self):
        return self.get_string()

    def write_file(self, filename='PARAMETERS'):
        """
        Write Tags to a Feff parameter tag file.

        Args:
            filename: filename and path to write to.
        """
        with zopen(filename, "wt") as f:
            f.write(self.__str__() + "\n")

    @staticmethod
    def from_file(filename="feff.inp"):
        """
        Creates a Feff_tag dictionary from a PARAMETER or feff.inp file.

        Args:
            filename: Filename for either PARAMETER or feff.inp file

        Returns:
            Feff_tag object
        """
        with zopen(filename, "rt") as f:
            lines = list(clean_lines(f.readlines()))
        params = {}
        for line in lines:
            m = re.match("([A-Z]+\d*\d*)\s*(.*)", line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                val = Tags.proc_val(key, val)
                if key not in ("ATOMS", "POTENTIALS", "END", "TITLE"):
                    params[key] = val
        return Tags(params)

    @staticmethod
    def proc_val(key, val):
        """
        Static helper method to convert Feff parameters to proper types, e.g.
        integers, floats, lists, etc.

        Args:
            key: Feff parameter key
            val: Actual value of Feff parameter.
        """
        list_type_keys = VALID_FEFF_TAGS
        boolean_type_keys = ()
        float_type_keys = ("SCF", "EXCHANGE", "S02", "FMS", "XANES", "EXAFS",
                           "RPATH", "LDOS")
        int_type_keys = ("PRINT", "CONTROL")

        def smart_int_or_float(numstr):
            if numstr.find(".") != -1 or numstr.lower().find("e") != -1:
                return float(numstr)
            else:
                return int(numstr)

        try:
            if key in list_type_keys:
                output = list()
                toks = re.split("\s+", val)

                for tok in toks:
                    m = re.match("(\d+)\*([\d\.\-\+]+)", tok)
                    if m:
                        output.extend([smart_int_or_float(m.group(2))] *
                                      int(m.group(1)))
                    else:
                        output.append(smart_int_or_float(tok))
                return output
            if key in boolean_type_keys:
                m = re.search("^\W+([TtFf])", val)
                if m:
                    if m.group(1) == "T" or m.group(1) == "t":
                        return True
                    else:
                        return False
                raise ValueError(key + " should be a boolean type!")

            if key in float_type_keys:
                return float(val)

            if key in int_type_keys:
                return int(val)

        except ValueError:
            return val.capitalize()

        return val.capitalize()

    def diff(self, other):
        """
        Diff function.  Compares two PARAMETER files and indicates which
        parameters are the same and which are not. Useful for checking whether
        two runs were done using the same parameters.

        Args:
            other: The other PARAMETER dictionary to compare to.

        Returns:
            Dict of the format {"Same" : parameters_that_are_the_same,
            "Different": parameters_that_are_different} Note that the
            parameters are return as full dictionaries of values.
        """
        similar_param = {}
        different_param = {}
        for k1, v1 in self.items():
            if k1 not in other:
                different_param[k1] = {"FEFF_TAGS1": v1,
                                       "FEFF_TAGS2": "Default"}
            elif v1 != other[k1]:
                different_param[k1] = {"FEFF_TAGS1": v1,
                                       "FEFF_TAGS2": other[k1]}
            else:
                similar_param[k1] = v1
        for k2, v2 in other.items():
            if k2 not in similar_param and k2 not in different_param:
                if k2 not in self:
                    different_param[k2] = {"FEFF_TAGS1": "Default",
                                           "FEFF_TAGS2": v2}
        return {"Same": similar_param, "Different": different_param}

    def __add__(self, other):
        """
        Add all the values of another Tags object to this object
        Facilitates the use of "standard" Tags
        """
        params = dict(self)
        for k, v in other.items():
            if k in self and v != self[k]:
                raise ValueError("Tags have conflicting values!")
            else:
                params[k] = v
        return Tags(params)


class Potential(MSONable):
    """
    FEFF atomic potential.
    """

    def __init__(self, struct, central_atom):
        """
        Args:
            struct (Structure): Structure object.
            central_atom (str): Absorbing atom symbol
        """
        self.central_atom = central_atom
        if struct.is_ordered:
            self.struct = struct
            self.pot_dict = get_atom_map(struct)
        else:
            raise ValueError("Structure with partial occupancies cannot be "
                             "converted into atomic coordinates!")

    @staticmethod
    def pot_string_from_file(filename='feff.inp'):
        """
        Reads Potential parameters from a feff.inp or FEFFPOT file.
        The lines are arranged as follows:

          ipot   Z   element   lmax1   lmax2   stoichometry   spinph

        Args:
            filename: file name containing potential data.

        Returns:
            FEFFPOT string.
        """
        with zopen(filename, "rt") as f_object:
            f = f_object.readlines()
            ln = -1
            pot_str = ["POTENTIALS\n"]
            pot_tag = -1
            pot_data = 0
            pot_data_over = 1

            for line in f:
                if pot_data_over == 1:
                    ln += 1
                    if pot_tag == -1:
                        pot_tag = line.find("POTENTIALS")
                        ln = 0
                    if pot_tag >= 0 and ln > 0 and pot_data_over > 0:
                        try:
                            if int(line.split()[0]) == pot_data:
                                pot_data += 1
                                pot_str.append(line.replace("\r", ""))
                        except (ValueError, IndexError):
                            if pot_data > 0:
                                pot_data_over = 0
        return ''.join(pot_str)

    @staticmethod
    def pot_dict_from_string(pot_data):
        """
        Creates atomic symbol/potential number dictionary
        forward and reverse

        Arg:
            pot_data: potential data in string format

        Returns:
            forward and reverse atom symbol and potential number dictionaries.
        """

        pot_dict = {}
        pot_dict_reverse = {}
        begin = 0
        ln = -1

        for line in pot_data.split("\n"):
            try:
                if begin == 0 and line.split()[0] == "0":
                    begin += 1
                    ln = 0
                if begin == 1:
                    ln += 1
                if ln > 0:
                    atom = line.split()[2]
                    index = int(line.split()[0])
                    pot_dict[atom] = index
                    pot_dict_reverse[index] = atom
            except (ValueError, IndexError):
                pass
        return pot_dict, pot_dict_reverse

    def __str__(self):
        """
        Returns a string representation of potential parameters to be used in
        the feff.inp file,
        determined from structure object.

                The lines are arranged as follows:

          ipot   Z   element   lmax1   lmax2   stoichiometry   spinph

        Returns:
            String representation of Atomic Coordinate Shells.
        """
        central_element = Element(self.central_atom)
        ipotrow = [[0, central_element.Z, central_element.symbol, -1, -1, .0001, 0]]
        for el, amt in self.struct.composition.items():
            ipot = self.pot_dict[el.symbol]
            ipotrow.append([ipot, el.Z, el.symbol, -1, -1, amt, 0])
        ipot_sorted = sorted(ipotrow, key=itemgetter(0))
        ipotrow = str(tabulate(ipot_sorted,
                               headers=["*ipot", "Z", "tag", "lmax1",
                                        "lmax2", "xnatph(stoichometry)",
                                        "spinph"]))
        ipotlist = ipotrow.replace("--", "**")
        ipotlist = ''.join(["POTENTIALS\n", ipotlist])

        return ipotlist

    def write_file(self, filename='POTENTIALS'):
        """
        Write to file.

        Args:
            filename: filename and path to write potential file to.
        """
        with zopen(filename, "wt") as f:
            f.write(str(self) + "\n")


class FeffParserError(Exception):
    """
    Exception class for Structure.
    Raised when the structure has problems, e.g., atoms that are too close.
    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "FeffParserError : " + self.msg


def get_atom_map(structure):
    """
    Returns a dict that maps each atomic symbol to a unique integer starting
    from 1.

    Args:
        structure (Structure)

    Returns:
        dict
    """
    syms = [site.specie.symbol for site in structure]
    unique_pot_atoms = []
    [unique_pot_atoms.append(i) for i in syms if not unique_pot_atoms.count(i)]
    atom_map = {}
    for i, atom in enumerate(unique_pot_atoms):
        atom_map[atom] = i + 1
    return atom_map
