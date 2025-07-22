"""
Classes and functions for trajectory-based analysis

Copyright Schrodinger, LLC. All rights reserved.
"""
import math
import warnings
from collections import Counter
from collections import defaultdict
from collections import namedtuple
from enum import Enum
from functools import lru_cache
from functools import partial
from itertools import chain
from itertools import product
from typing import List
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Callable

import numpy
from scipy.spatial.distance import cdist

from schrodinger import adapter
from schrodinger import structure
from schrodinger.comparison.atom_mapper import align_pos
from schrodinger.application.desmond import cms
from schrodinger.application.desmond import constants
from schrodinger.application.desmond.packages import msys
from schrodinger.application.desmond.packages import pfx
from schrodinger.application.desmond.packages import staf
from schrodinger.application.desmond.packages import topo
from schrodinger.infra import mm
from schrodinger.infra import mmbitset
from schrodinger.rdkit import rdkit_adapter
from schrodinger.structutils.analyze import calculate_sasa
from schrodinger.structutils.analyze import calculate_sasa_by_atom
from schrodinger.structutils.analyze import calculate_sasa_by_residue
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structutils.interactions.hbond import get_halogen_bonds
from schrodinger.structutils.interactions.hbond import get_hydrogen_bonds
from schrodinger.structutils.interactions.pi import find_pi_cation_interactions
from schrodinger.structutils.interactions.pi import find_pi_pi_interactions
from schrodinger.structutils.interactions.salt_bridge import get_salt_bridges
from schrodinger.structutils.rmsd import calculate_in_place_rmsd
from schrodinger.structutils.rmsd import superimpose

BACKBONE_ASL = '(protein and backbone) OR nucleic_backbone'


class GroupType(Enum):
    """
    Enum representing different methods to grouping method to group atoms.

    Currently supported types are:
    - MOLECULE: Atoms are grouped by the molecule they belong to.
    - RESIDUE: Atoms are grouped by the residue they belong to.
    """
    MOLECULE = "molecule"
    RESIDUE = "residue"

    def __str__(self) -> str:
        """
        Return the string representation of the enum value
        """
        return str(self.value)


class _Const:
    CONTACT_CUTOFF = 6.0
    HBOND_CUTOFF = 2.8
    HBOND_MIN_A_ANGLE = 90.0
    HBOND_MIN_D_ANGLE = 120.0
    HBOND_MAX_A_ANGLE = 180.0

    # FIXME: Maestro has two input parameters for Donor mininum angle, one for
    #        the halogen atom as donor, one for the halogen atom as acceptor.
    #        However, the structutils.interactions.hbond API only exposes one.
    #        Adjust code here once SHARED-6040 is closed.
    HALOGEN_BOND_CUTOFF = 3.5
    HALOGEN_BOND_MIN_A_ANGLE = 90.0
    HALOGEN_BOND_MIN_D_ANGLE = 140.0  # there are 2 options in Maestro
    HALOGEN_BOND_MAX_A_ANGLE = 170.0

    HYDROPHOBIC_SEARCH_CUTOFF = 3.2
    HYDROPHOBIC_CUTOFF = 3.6
    HYDROPHOBIC_TYPES = ' '.join(
        ['PHE', 'LEU', 'ILE', 'TYR', 'TRP', 'VAL', 'MET', 'PRO', 'CYS', 'ALA'])
    METAL_ASL = '((ions) or (metals) or (metalloids))'
    METAL_CUTOFF = 3.4
    RESNAME_MAP = {
        'HIE': 'HIS',
        'HID': 'HIS',
        'HIP': 'HIS',
        'HSD': 'HIS',
        'HSP': 'HIS',
        'HSE': 'HIS',
        'CYX': 'CYS',
        'LYN': 'LYS',
        'ARN': 'ARG',
        'ASH': 'ASP',
        'GLH': 'GLU'
    }
    SALT_BRIDGE_CUTOFF = 5.0

    # values used in HydrohobicInterFinder, specified in Maestro
    GOOD_CONTACTS_CUTOFF_RATIO = 1.3
    BAD_CONTACTS_CUTOFF_RATIO = 0.89


def is_small_struc(atoms):
    """
    A simple API to determine whether a molecular structure is small.

    :type  atoms: `list`
    :param atoms: A list of atoms in the structure. The atoms can be atom IDs
                  or atom-class instances.
    """
    return len(atoms) <= 250


class Pbc:

    def __init__(self, box):
        """
        This implementation supports triclinic cell.

        :type  box: `numpy.ndarray`
        :param box: 3x3 matrix whose ROWS are primitive cell vectors.
                    For a `msys.System` instance, call `msys_model.cell`
                    to get this matrix. For a `traj.Frame` instance,
                    call `fr.box` to get it. For a `Cms` instance, call
                    `numpy.reshape(cms_model.box, [3, 3])` to get it.
        """
        self._box = box
        self._box_col_major = self._box.flatten('F')
        self._inv = numpy.linalg.inv(box)
        self._proj = self._inv.flatten('F')
        self._volume = numpy.linalg.det(box)
        assert isinstance(self._box[0][0], numpy.float64)
        assert isinstance(self._inv[0][0], numpy.float64)

    @property
    def box(self):
        return self._box

    @property
    def volume(self):
        return self._volume

    @property
    def inv_box(self):
        return self._inv

    def calcMinimumImage(self, ref_pos, pos):
        """
        Calculates the minimum image of a position vector `pos` relative to
        another position vector `ref_pos`.
        `pos` and `ref_pos` can also be arrays of 3D vectors. In this case,
        they must be of the same size, and minimum images will be calculated for
        each element in `pos` and `ref_pos`.

        :type  ref_pos: `numpy.ndarray`. Either 1x3 or Nx3.
        :param ref_pos: Reference position vector(s)
        :type      pos: `numpy.ndarray`. Either 1x3 or Nx3.
        :param     pos: Position vector(s) of which we will calculate the
                        minimum image.

        :rtype: `numpy.ndarray` with `numpy.float64` elements
        :return: The position vector(s) of the mininum image. This function does
                 NOT mutate any of the input vectors.
        """
        diff = (pos - ref_pos).astype(numpy.float64)
        return pos + msys._msys.calc_wrap_shift_array(
            self._box_col_major, self._proj, diff, out=diff)

    def calcMinimumDiff(self, from_pos, to_pos):
        """
        Calculates the difference vector from `from_pos` to the minimum image
        of `to_pos`.
        `pos` and `ref_pos` can also be arrays of 3D vectors. In this case,
        they must be of the same size, and minimum image difference will be
        calculated for each element in `pos` and `ref_pos`.

        :type  from_pos: `numpy.ndarray`. Either 1x3 or Nx3
        :param from_pos: Reference position vector(s)
        :type    to_pos: `numpy.ndarray`. Either 1x3 or Nx3
        :param   to_pos: Position vector(s) of which we will calculate the
                         minimum image.

        :rtype: `numpy.ndarray` with `numpy.float64` elements
        :return: The difference vector(s). This function does NOT mutate any of
                 the input vectors.
        """
        diff = (to_pos - from_pos).astype(numpy.float64)
        return msys._msys.wrap_vector_array(self._box_col_major,
                                            self._proj,
                                            diff,
                                            out=diff)

    def wrap(self, pos):
        """
        Puts a coordinate back into the box. If the coordinate is already in the
        box, this function will return a new position vector that equals the
        original vector.

        :type pos: `numpy.ndarray`

        :rtype: `numpy.ndarray` with `numpy.float64` elements
        :return: A new position vector which is within the box. This function
                 does NOT mutate and return the input vector `pos`.
        """
        return self.calcMinimumImage(numpy.zeros(3), pos)

    def isWithinCutoff(self, pos0, pos1, cutoff_sq):
        """
        Return True if any of pos0 and pos1 are within the cutoff distance.

        :type       pos0: 1x3 or Nx3 numpy.ndarray
        :type       pos1: 1x3 or Nx3 numpy.ndarray
        :type  cutoff_sq: `float`
        :param cutoff_sq: = cutoff x cutoff
        """
        d = self.calcMinimumDiff(pos0, pos1)
        d_sq = numpy.einsum('ij,ij->i', d, d)
        return numpy.any(d_sq <= cutoff_sq)


def _pos2circ(data, pbc, fr, *_):
    """
    Convert a 3D vector (x, y, z) into a circular coordinate:

      (array([cos_x', cos_y', cos_z',]), array([sin_x', sin_y', sin_z']),)

    . This is needed for the center of mass (or charge, or centroid, for that
    matter) calculations.

    :type data: `dict`. Key = GID, value = circular coordinate of the atom
    """
    gids = list(data.keys())
    pos = fr.pos(gids)
    ang = 2 * numpy.pi * pos.dot(pbc.inv_box)
    return dict(zip(gids, zip(numpy.cos(ang), numpy.sin(ang))))


class CenterOf(staf.GeomAnalyzerBase):
    """
    Base class for computing averaged center of a group of atoms, with optional
    weights. Periodic boundary condition is taken into account.

    N.B.: The calculated center is an unwrapped coordinate.
    """

    def __init__(self,
                 gids: List[int],
                 weights: Optional[List[float]] = None,
                 return_unwrapped_atompos=False):
        """
        :param return_unwrapped_atompos: if `False`, return the unwrapped
                                         center. Otherwise return both unwrapped
                                         center and the unwrapped positions of
                                         the selected atoms.
        """
        self._gids = gids
        self._weights = None
        if weights is not None:
            self._weights = numpy.asarray(weights)
        self._return_atompos = return_unwrapped_atompos
        if self._weights is not None and numpy.isclose(self._weights.sum(), 0):
            raise ValueError('{} sums to 0.'.format(weights))
        if not self._gids:
            raise ValueError('No atom selected')

    def _precalc(self, calc):
        """
        :type calc: `GeomCalc`
        """
        list(map(lambda gid: calc.addCustom(_pos2circ, gid), self._gids))

    # in numpy 1.20, you can do numpy.typing.ndarray[float]
    def _calcCircMean(self, calc, pbc) -> numpy.ndarray:
        """
        :return: Circular mean as 1x3 array

        coordinate transformation::

            cartesian = triclinic * pbc.box
            (x, y, z) = (a, b, c) * pbc.box

        The calculation involves three steps:

        1. Compute the approximate centroid using the circular mean.
        2. Unwrap all points with respect to the approximate centroid.
        3. Compute the weighted geometric center the usual way.

        Details of the circular mean algorithm can be found at U{wiki page<https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions>}.
        """
        # Gets circular coordinates for all involved atoms.
        circ = calc.getCustom(_pos2circ)
        # circ[i]: ((cos(x), cos(y), cos(z),), (sin(x), sin(y), sin(z),),)

        # Calculates the circular mean.
        cos, sin = list(zip(*[circ[i] for i in self._gids]))
        _x = numpy.average(cos, axis=0)
        _y = numpy.average(sin, axis=0)
        _a = numpy.arctan2(_y, _x)

        # In case the atoms are uniformly distributed along one axis,
        # the circular mean becomes erroneous thus we set the coordinate of the
        # center along that axis to be 0, i.e., the middle of the box
        uniform = numpy.linalg.norm(numpy.asarray([_x, _y]), axis=0) < 0.3
        if numpy.any(uniform):
            _a[uniform] = 0

        return _a.dot(pbc.box) / (2 * numpy.pi)

    def _postcalc(self, calc, pbc, fr):
        """
        Result is (center, unwrapped-positions), where the first element is the
        unwrapped center, and the second element is a list of unwrapped
        positions of the selected atoms.
        """
        circ_mean = self._calcCircMean(calc, pbc)

        # Uses the circular mean as the center to unwrap coordinates.
        unwrap = partial(pbc.calcMinimumImage, circ_mean)
        unwrapped_pos = numpy.asarray(list(map(unwrap, fr.pos(self._gids))))

        # Computes the weighted geometric center the usual way.
        center = numpy.average(unwrapped_pos, 0, self._weights)
        self._result = (center,
                        unwrapped_pos) if self._return_atompos else center


class Com(CenterOf):
    """
    Class for computing averaged position weighted by atomic mass under
    periodic boundary condition.

    Basic usage:
      ana = Com(msys_model, cms_model, gids=[1, 23, 34, 5, 6])
      results = analyze(tr, ana)

    where `tr` is a trajectory, and `results` contain a `list` of unwrapped
    centers of mass as `floats`, one `float` for each frame. If
    return_unwrapped_atompos is `True`, `results` contain a list of 2-tuples:
    (unwrapped-center-of-mass, [unwrapped-positions-of-involved-atoms]), and
    each 2-tuple in the list corresponds to a trajectory frame.
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 asl=None,
                 gids=None,
                 return_unwrapped_atompos=False):
        """
        :type      msys_model: `msys.System`
        :type       cms_model: `cms.Cms`
        :type             asl: `str`
        :param            asl: ASL expression to specify the atom selection
        :type            gids: `list` of `int`
        :param           gids: GIDs of atoms
        :param return_unwrapped_atompos: if `False`, return the unwrapped
                                         center. Otherwise return both unwrapped
                                         center and the unwrapped positions of
                                         the selected atoms.

        Both `msys_model` and `cms_model` must be previously obtained through
        the `read_cms` function. They both should have the same atom
        coordinates and the same simulation box matrix. `cms_model` is used to
        obtain atom GIDs from ASL selection. `msys_model` is used to retrieve
        atom attribute from GIDs.

        Either `asl` or `gids` must be specified, but not both.
        """
        assert bool(asl) ^ bool(gids)

        # pseudoatoms need to be included here for L{Dipole} to work
        if cms_model.need_msys:
            gids = gids or topo.asl2gids(cms_model, asl)
            weights = [msys_model.atom(gid).mass for gid in gids]
        else:
            if asl:
                aids = cms_model.select_atom(asl)
            gids = gids or cms_model.convert_to_gids(aids, with_pseudo=True)
            weights = cms_model.get_mass(gids)
        CenterOf.__init__(self, gids, weights, return_unwrapped_atompos)


class Coc(Com):
    '''
    Class for computing center of charge under periodic boundary condition.
    Pseudo atoms are included.

    For each frame, the results will be the unwrapped-center-of-charge. If
    return_unwrapped_atompos is `True`, the results will be a 2-tuple:
    (unwrapped-center-of-charge, [unwrapped-positions-of-involved-atoms]).
    '''

    def __init__(self,
                 msys_model,
                 cms_model,
                 asl=None,
                 gids=None,
                 return_unwrapped_atompos=False):
        """
        Refer to the docstring of `Com.__init__`.
        """
        assert bool(asl) ^ bool(gids)

        if cms_model.need_msys:
            gids = gids or topo.asl2gids(
                cms_model, asl, include_pseudoatoms=True)
            weights = [msys_model.atom(gid).charge for gid in gids]
        else:
            if asl:
                aids = cms_model.select_atom(asl)
            gids = gids or cms_model.convert_to_gids(aids, with_pseudo=True)
            weights = cms_model.get_charge(gids)
        CenterOf.__init__(self, gids, weights, return_unwrapped_atompos)


class Centroid(CenterOf):
    '''
    Class for computing centroid under periodic boundary condition.

    For each frame, the results will be the unwrapped centroid. If
    return_unwrapped_atompos is `True`, the results will be a 2-tuple:
    (unwrapped-centroid, [unwrapped-positions-of-involved-atoms]).
    '''

    def __init__(self,
                 msys_model,
                 cms_model,
                 asl=None,
                 gids=None,
                 return_unwrapped_atompos=False):
        """
        Refer to the docstring of `Com.__init__`.
        """
        assert bool(asl) ^ bool(gids)

        if cms_model.need_msys:
            gids = gids or topo.asl2gids(
                cms_model, asl, include_pseudoatoms=False)
        else:
            if asl:
                aids = cms_model.select_atom(asl)
            gids = gids or cms_model.convert_to_gids(aids, with_pseudo=False)
        CenterOf.__init__(self, gids, None, return_unwrapped_atompos)


class Vector(staf.GeomAnalyzerBase):
    """
    Calculate the vector between two xids. Result is a vector for each
    trajectory frame.
    """

    def __init__(self, msys_model, cms_model, from_xid, to_xid):
        '''
        :type from_xid: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        :type   to_xid: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        '''
        self._sites = self._get_sites(cms_model, [from_xid, to_xid])
        if cms_model.need_msys:
            self._noffset = msys_model.natoms
        else:
            self._noffset = cms_model.particle_total

    def _precalc(self, calc):
        self._gids = self._sites2gids(calc, self._sites, self._noffset)
        calc.addVector(*self._gids)

    def _postcalc(self, calc, *_):
        self._result = calc.getVector(*self._gids)


class Distance(staf.GeomAnalyzerBase):
    """
    Calculate the distance between two xids. Result is a scalar (distance in
    Angstroms) for each trajectory frame.
    """

    def __init__(self, msys_model, cms_model, xid0, xid1):
        '''
        :type xid0: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        :type xid1: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        '''
        self._sites = self._get_sites(cms_model, [xid0, xid1])
        if cms_model.need_msys:
            self._noffset = msys_model.natoms
        else:
            self._noffset = cms_model.particle_total

    def _precalc(self, calc):
        self._gids = self._sites2gids(calc, self._sites, self._noffset)
        calc.addDistance(*self._gids)

    def _postcalc(self, calc, *_):
        self._result = calc.getDistance(*self._gids)


class Angle(staf.GeomAnalyzerBase):
    """
    Calculate the angle formed between three xids. Result is a scalar (angle in
    degrees) for each trajectory frame.
    """

    def __init__(self, msys_model, cms_model, xid0, xid1, xid2):
        """
        The angle is formed by the vectors `xid1`==>`xid0` and
        `xid1`==>`xid2`.

        :type xid0: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        :type xid1: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        :type xid2: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        """
        self._sites = self._get_sites(cms_model, [xid0, xid1, xid2])
        if cms_model.need_msys:
            self._noffset = msys_model.natoms
        else:
            self._noffset = cms_model.particle_total

    def _precalc(self, calc):
        self._gids = self._sites2gids(calc, self._sites, self._noffset)
        calc.addAngle(*self._gids)

    def _postcalc(self, calc, *_):
        self._result = math.degrees(calc.getAngle(*self._gids))


class Torsion(staf.GeomAnalyzerBase):
    """
    Calculate the torsion formed between four xids. Result is a scalar
    (dihedral angle in degrees) for each trajectory frame.
    """

    def __init__(self, msys_model, cms_model, xid0, xid1, xid2, xid3):
        r"""
        The torsion is defined by the four atoms::

              0 o           o 3
                 \         /
                  \       /
                 1 o-----o 2

        :type xid0: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        :type xid1: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        :type xid2: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        :type xid3: aid or `CenterOf` type -- `Com`, `Coc`, `Centroid`
        """
        self._sites = self._get_sites(cms_model, [xid0, xid1, xid2, xid3])
        if cms_model.need_msys:
            self._noffset = msys_model.natoms
        else:
            self._noffset = cms_model.particle_total

    def _precalc(self, calc):
        self._gids = self._sites2gids(calc, self._sites, self._noffset)
        calc.addTorsion(*self._gids)

    def _postcalc(self, calc, *_):
        self._result = math.degrees(calc.getTorsion(*self._gids))


class PlanarAngle(staf.GeomAnalyzerBase):
    """
    Calculate planar angle formed among six xids. The first three xids define
    the first plane and the latter three xids define the second plane.  Result
    is a list of planar angles in degrees for the trajectory frames.
    """

    def __init__(
            self,
            msys_model: msys.System,
            cms_model: "cms.Cms",  # noqa: F821
            xid0: Union[int, CenterOf],
            xid1: Union[int, CenterOf],
            xid2: Union[int, CenterOf],
            xid3: Union[int, CenterOf],
            xid4: Union[int, CenterOf],
            xid5: Union[int, CenterOf],
            minangle: bool = True):
        '''
        :param msys_model: defines the system structure and atomic mapping
        :param cms_model: defines the system structure and atomic mapping
        :param xid*: integer representing an aid or `CenterOf` type (`Com`,
                     `Coc`, `Centroid`)
        :param minangle: `True` to restrict the returned angle to the range [0,
                         90] degrees, treating the order of atoms defining the
                         plane as unimportant and ignoring the directionality
                         of the plane normals. `False` to return the angle in
                         the range [0, 180] degrees.
        '''
        self._sites = self._get_sites(cms_model,
                                      [xid0, xid1, xid2, xid3, xid4, xid5])
        if cms_model.need_msys:
            self._noffset = msys_model.natoms
        else:
            self._noffset = cms_model.particle_total
        self._minangle = minangle

    def _precalc(self, calc):
        self._gids = self._sites2gids(calc, self._sites, self._noffset)
        calc.addPlanarAngle(*self._gids)

    def _postcalc(self, calc, *_):
        angle_degrees = math.degrees(calc.getPlanarAngle(*self._gids))
        if self._minangle and angle_degrees > 90:
            angle_degrees = 180 - angle_degrees
        self._result = angle_degrees


class FittedPlanarAngle(staf.GeomAnalyzerBase):
    """
    Calculate planar angle formed among two groups of atoms, each of containing
    3 or more atoms.  The first list contains xids of the first group and the
    second list contains xids of the second group. A best-fitting plane is
    calculated for each group of xids. Result is a list of planar angles in
    degrees for the trajectory frames.

    This analyzer is useful for such cases as calculating angles between two
    rings.
    """

    def __init__(
            self,
            msys_model: msys.System,
            cms_model: "cms.Cms",  # noqa: F821
            xids0: List[Union[int, CenterOf]],
            xids1: List[Union[int, CenterOf]],
            minangle: bool = True):
        '''
        :param msys_model: defines the system structure and atomic mapping
        :param cms_model: defines the system structure and atomic mapping
        :param xids*: list of integers representing aid or `CenterOf` types
                      (`Com`, `Coc`, `Centroid`)
        :param minangle: `True` to restrict the returned angle to the range [0,
                         90] degrees, treating the order of atoms defining the
                         plane as unimportant and ignoring the directionality
                         of the plane normals. `False` to return the angle in
                         the range [0, 180] degrees.
        '''
        if len(xids0) < 3 or len(xids1) < 3:
            raise ValueError('Both groups must contain at least 3 atoms.')
        self._sites0 = self._get_sites(cms_model, xids0)
        self._sites1 = self._get_sites(cms_model, xids1)
        if cms_model.need_msys:
            self._noffset = msys_model.natoms
        else:
            self._noffset = cms_model.particle_total
        self._minangle = minangle

    def _precalc(self, calc):
        self._gids0 = self._sites2gids(calc, self._sites0, self._noffset)
        self._gids1 = self._sites2gids(calc, self._sites1, self._noffset)
        calc.addFittedPlanarAngle(self._gids0, self._gids1)

    def _postcalc(self, calc, *_):
        angle_degrees = math.degrees(
            calc.getFittedPlanarAngle(self._gids0, self._gids1))
        if self._minangle and angle_degrees > 90:
            angle_degrees = 180 - angle_degrees
        self._result = angle_degrees


class Gyradius(staf.CompositeAnalyzer):
    """
    Class for computing radius of gyration under periodic boundary condition.

    For each frame, the result is the radius of gyration as `float`
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 asl=None,
                 gids=None,
                 mass_weighted: bool = False):
        self._analyzers = [
            Com(msys_model, cms_model, asl, gids, return_unwrapped_atompos=True)
        ]
        if mass_weighted:
            self._weights = self._analyzers[0]._weights
        else:
            self._weights = [1] * len(self._analyzers[0]._weights)

    def _postcalc(self, *args):
        super()._postcalc(*args)
        com_pos, pos = self._analyzers[0]()
        r2 = numpy.square(pos - com_pos).sum(axis=1)
        self._result = numpy.average(r2, weights=self._weights)**0.5


class MassAvgVel(Com):
    """
    Class for computing mass-averaged velocity.
    The trajectory should contain velocities data.

    For each frame, the result is `numpy.ndarray` of `float`
    """

    def _precalc(self, calc):
        pass

    def _postcalc(self, calc, pbc, fr):
        vel = fr.vel()[self._gids]
        self._result = numpy.average(vel, axis=0, weights=self._weights)


def _unwrap_wrt_prevpos(data, pbc, fr, *_):
    '''
    Unwrap every point with respect to its coordinate in the prev frame, if available.

    :type data: `dict`. Key = GID, value = `None` for the first frame,
                otherwise value = the previous frame with unwrapped atom
                coordinates. All GIDs map to the same frame instance.

    :rtype: `dict`. Key = GID, value = a copy of the input frame instance
            `fr`. If previous frame is available in `data`, the atom
            coordinates are unwrapped with respect to their coordinates in the
            previous frame.
    '''
    gids = list(data)
    prev_fr = data[gids[0]]
    unwrapped_fr = fr.copy()
    if prev_fr is not None:
        # Not first frame
        for i in gids:
            unwrapped_pos = pbc.calcMinimumImage(prev_fr.pos(i), fr.pos(i))
            numpy.copyto(unwrapped_fr.pos(i), unwrapped_pos)
    return dict.fromkeys(gids, unwrapped_fr)


class PosTrack(staf.GeomAnalyzerBase):
    '''
    Class for tracking positions of selected atoms in a trajectory.
    Pseudo atoms are included.

    Since periodic boundary condition is assumed in the MD simulation, the
    atom positions are wrapped back into the simulation box when they move out
    of the box. The PosTrack class unwraps the atom positions with respect to
    their positions in the previous frame. It can be used when atom positions
    need to be tracked over time, such as diffusion.
    '''

    def __init__(self, msys_model, cms_model, asl=None, gids=None):
        """
        Refer to the docstring of `Com.__init__`.
        """
        assert bool(asl) ^ bool(gids)

        if cms_model.need_msys:
            gids = gids or topo.asl2gids(
                cms_model, asl, include_pseudoatoms=True)
        else:
            gids = gids or cms_model.convert_to_gids(cms_model.select_atom(asl),
                                                     with_pseudo=True)
        if not gids:
            raise ValueError('No atom selected')
        self._gids = gids

    def _precalc(self, calc):
        list(
            map(lambda gid: calc.addCustom(_unwrap_wrt_prevpos, gid),
                self._gids))

    def _postcalc(self, calc, *_):
        """
        Result is a new frame with selected atoms unwrapped with respect to
        previous frame, if previous frame exists. Otherwise it is a copy of
        the input frame.
        """
        data = calc.getCustom(_unwrap_wrt_prevpos)
        self._result = data[self._gids[0]]


# Aliases for backward compatibility
RadiusOfGyration = Gyradius
CenterOfMotion = MassAvgVel
Position = PosTrack


class _Ramachandran(staf.GeomAnalyzerBase):
    """
    Calculate the Phi and Psi torsions for selected atoms, with GIDs as input.

    Usage example:

      ana = Ramachandran([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])
      # The phi-psi torsions are defined by 1-2-3-4 (phi_0), 2-3-4-5 (psi_0),
      # 3-4-5-6 (phi_1), and 4-5-6-7 (psi_1).

      results = analyze(tr, ana)

    where `tr` is a trajectory, and `results` is a list, and each element in
    the list is a list: [(phi_0, psi_0), (phi_1, psi_1),] for the corresponding
    trajectory frame.
    """

    def __init__(self, phipsi_gids):
        """
        :type  phipsi_gids: `list` of `list` (or `tuple`) of `int`
        :param phipsi_gids: Each element is a 5-element `list` or `tuple`,
                specifying the GIDs of the C0, N0, CA, C1, N1 backbone atoms
        """
        self._phi_psi = []
        for c0, n0, ca, c1, n1 in phipsi_gids:
            self._phi_psi.append((
                (c0, n0, ca, c1),
                (n0, ca, c1, n1),
            ))

    def _precalc(self, calc):
        list(
            map(
                lambda phi_psi2: calc.addTorsion(*phi_psi2[0]) or calc.
                addTorsion(*phi_psi2[1]), self._phi_psi))

    def _postcalc(self, calc, *_):
        self._result = [(
            calc.getTorsion(*phi_psi1[0]),
            calc.getTorsion(*phi_psi1[1]),
        ) for phi_psi1 in self._phi_psi]


class Ramachandran(_Ramachandran):
    """
    Calculate the Phi and Psi torsions for selected atoms.

    Usage example:

      ana = Ramachandran(msys_model, cms_model, 'protein and res.num 20-30')
      results = analyze(tr, ana)

    where `tr` is a trajectory, and `results` is a `list`, and each element
    in the `list` is a `list`: [(phi_0, psi_0), (phi_1, psi_1),] for the
    corresponding trajectory frame.
    """

    def __init__(self, msys_model, cms_model, asl):
        """
        :type msys_model: `msys.System`
        :type  cms_model: `cms.Cms`
        :type        asl: `str`
        :param       asl: ASL expression to specify the residues
        """
        rama_list = []
        self._res_tags = []
        aids = cms_model.select_atom(asl)
        for res in structure.get_residues_by_connectivity(cms_model.fsys_ct):
            if res.atom[1].index not in aids:
                continue
            try:
                phi = res.getDihedralAtoms('Phi')
                psi = res.getDihedralAtoms('Psi')
            except ValueError:  # missing atoms
                continue
            phi_psi = phi + psi[-1:]
            if cms_model.need_msys:
                rama_list.append(
                    topo.aids2gids(cms_model, [a.index for a in phi_psi],
                                   include_pseudoatoms=False))
            else:
                rama_list.append(
                    cms_model.convert_to_gids([a.index for a in phi_psi],
                                              with_pseudo=False))
            tag = "%s:%s_%d" % (
                res.chain, _get_common_resname(res.pdbres.strip()), res.resnum)
            self._res_tags.append(tag)

        _Ramachandran.__init__(self, rama_list)

    def reduce(self, results, *_, **__):
        return self._res_tags, results


class PosAlign(staf.CenteredSoluteAnalysis):
    """
    This analyzer first centers the solute atoms. If `fit_aids` and `fit_ref_pos`
    are provided, it further aligns the given trajectory frames: first calculate
    the rotation / translation transformation to fit the sub-structure
    defined by `fit_aids` to the given geometry (`fit_ref_pos`), and then apply
    the transformation to the coordinates of the selected atoms (`aids`). The
    returned value is the transformed coordinates for `aids`.
    """

    def __init__(self, msys_model, cms_model, aids, fit_aids, fit_ref_pos):
        """
        :type   msys_model: `msys.System`
        :type    cms_model: `cms.Cms`
        :type         aids: `list` of `int`
        :type     fit_aids: `list` of `int` or `None`
        :param fit_ref_pos: positions of reference conformer structure for
                            translation/rotation calculation
        :type  fit_ref_pos: Mx3 `numpy.ndarray` or `None`

        Both `msys_model` and `cms_model` must be previously obtained through
        the `read_cms` function.
        """
        assert aids
        assert not (bool(fit_aids) ^ (fit_ref_pos is not None)), (
            'fit_aids and fit_ref_pos must be both set or unset.')
        if fit_aids:
            assert len(fit_aids) == len(fit_ref_pos)
            if cms_model.need_msys:
                self._fit_gids = topo.aids2gids(cms_model, fit_aids, False)
            else:
                self._fit_gids = cms_model.convert_to_gids(fit_aids,
                                                           with_pseudo=False)
        if cms_model.need_msys:
            self._gids = topo.aids2gids(cms_model,
                                        aids,
                                        include_pseudoatoms=False)
        else:
            self._gids = cms_model.convert_to_gids(aids, with_pseudo=False)
        self._fit_ref_pos = fit_ref_pos

        super().__init__(msys_model, cms_model)

    def _postcalc(self, calc, *_):
        centered_fr = self._getCenteredFrame(calc)
        pos = centered_fr.pos(self._gids)
        if self._fit_ref_pos is not None:
            fit_pos = centered_fr.pos(self._fit_gids)
            pos = align_pos(pos, fit_pos, self._fit_ref_pos)
        self._result = pos


class RMSD(PosAlign):
    """
    Root Mean Square Deviation with respect to reference positions, with optional
    alignment fitting.

    See `RMSF` docstring for a detailed example (replace "RMSF" with "RMSD").

    If spikes are seen, call `topo.make_glued_topology` first.
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 aids,
                 ref_pos,
                 fit_aids=None,
                 fit_ref_pos=None,
                 in_place=False):
        """
        See `PosAlign` for parameters.

        :param     ref_pos: positions of reference conformer structure
        :type      ref_pos: Nx3 `numpy.ndarray`
        :type     fit_aids: `list` of `int`s or `None`. If `None`, it is set
                            to aids, and fit_ref_pos is set to ref_pos.
        :param    in_place: if `True`, calculate RMSD without applying
                            transformations on `ref_pos`

        Typically, `aids` and `fit_aids` come from a common source whereas
        `ref_pos` and `fit_ref_pos` come from another common source.
        """
        assert len(aids) == len(ref_pos)
        if in_place:
            assert not fit_aids
            if cms_model.need_msys:
                self._gids = topo.aids2gids(cms_model, aids, False)
            else:
                self._gids = cms_model.convert_to_gids(aids, with_pseudo=False)
        else:
            if not fit_aids:
                fit_aids, fit_ref_pos = aids, ref_pos
            super().__init__(msys_model, cms_model, aids, fit_aids, fit_ref_pos)
        self._in_place = in_place
        self._ref_pos = ref_pos

    def _precalc(self, calc):
        if not self._in_place:
            super()._precalc(calc)

    def _postcalc(self, calc, _, fr):
        if self._in_place:
            self._result = fr.pos(self._gids)
        else:
            super()._postcalc(calc)
        self._result = numpy.sqrt(
            ((self._result - self._ref_pos)**2).sum(axis=1).mean())


class RMSF(PosAlign):
    """
    Per-atom Root Mean Square Fluctuation with respect to averaged position
    over the trajectory, with optional alignment fitting.

    Example: calculate ligand RMSF with protein backbone aligned

    >>> backbone_asl = 'backbone and not (atom.ele H) and not (m.n 4)'
    >>> backbone_aids = cms_model.select_atom(backbone_asl)
    >>> ligand_aids = cms_model.select_atom('ligand')

    >>> # suppose the backbone reference position comes from a trajectory frame
    >>> backbone_gids = topo.aids2gids(cms_model, backbone_aids, include_pseudoatoms=False)
    >>> backbone_ref_pos = a_frame.pos(backbone_gids)

    >>> ana = RMSF(msys_model, cms_model, ligand_aids, backbone_aids, backbone_ref_pos)
    >>> result = analysis.analyze(a_trajectory, ana)

    Here result is a length N numpy array where N is the number of ligand atoms.
    If spikes are seen, call `topo.make_glued_topology` before any analysis:

    >>> topo.make_glued_topology(msys_model, cms_model)

    This call will change the topology of msys_model, i.e., add 'bonds' for
    atoms that are close and belong to different molecules, using the positions
    in cms_model as gold standard. This change only affects position unwrapping
    for certain trajectory APIs such as topo.make_whole(), topo.center().
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 aids,
                 fit_aids,
                 fit_ref_pos,
                 in_place=False):
        """
        :type   msys_model: `msys.System`
        :type    cms_model: `cms.Cms`
        :type         aids: `list` of `int`
        :type     fit_aids: `list` of `int`
        :param fit_ref_pos: positions of reference conformer structure for
                            translation/rotation calculation
        :type  fit_ref_pos: Mx3 `numpy.ndarray`
        :param    in_place: if `True`, calculate RMSF without applying
                            alignment transformations

        Both `msys_model` and `cms_model` must be previously obtained through
        the `read_cms` function.
        """
        if in_place:
            if cms_model.need_msys:
                self._gids = topo.aids2gids(cms_model, aids, False)
            else:
                self._gids = cms_model.convert_to_gids(aids, with_pseudo=False)
        else:
            super().__init__(msys_model, cms_model, aids, fit_aids, fit_ref_pos)
        self._in_place = in_place
        self._aids = aids

    def _precalc(self, calc):
        if not self._in_place:
            super()._precalc(calc)

    def _postcalc(self, calc, _, fr):
        if self._in_place:
            self._result = fr.pos(self._gids)
        else:
            super()._postcalc(calc)

    def reduce(self, pos_t, *_, **__):
        """
        Temporal average of the RMSF over the trajectory

        :type pos_t: List of Nx3 `numpy.ndarray` where N is the number of
                     atoms. The length of the list is the number of frames.

        :rtype: length N `numpy.ndarray`
        """
        pos_t_array = numpy.asarray(pos_t, dtype=numpy.float64)
        pos_avg_sq = ((pos_t_array.mean(axis=0))**2).sum(axis=1)
        pos_sq_avg = (pos_t_array**2).mean(axis=0).sum(axis=1)
        diff = numpy.maximum(pos_sq_avg - pos_avg_sq, 0.0)
        return numpy.sqrt(diff)


class LigandRMSD(PosAlign):
    """
    Ligand Root Mean Square Deviation from reference positions, with optional
    alignment fitting. Taking conformational symmetry into account.
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 aids,
                 ref_pos,
                 fit_aids=None,
                 fit_ref_pos=None):
        """
        see `RMSD.__init__` for parameters
        """
        assert len(aids) == len(ref_pos)
        super().__init__(msys_model, cms_model, aids, fit_aids, fit_ref_pos)
        self._aids = aids
        if cms_model.need_msys:
            self._gids = topo.aids2gids(cms_model,
                                        self._aids,
                                        include_pseudoatoms=False)
        else:
            self._gids = cms_model.convert_to_gids(self._aids,
                                                   with_pseudo=False)
        self._ref_st = cms_model.extract(aids)
        self._ref_st.setXYZ(ref_pos)
        self._cached_st = self._ref_st.copy()  # to be updated per frame

        # remove non-polar H, simplify bond order and atom charge, then create
        # all atom orders up to conformational symmetry using SMARTS matching
        # TODO? chirality can be used to further lessen the atom orders
        st = cms_model.fsys_ct.extract(aids)
        heavy_atom_w_polar_H = evaluate_asl(st, 'not (non_polar_hydrogens)')
        st_w_polarH = st.extract(heavy_atom_w_polar_H)
        for bond in st_w_polarH.bond:
            bond.order = 1
        for atom in st_w_polarH.atom:
            atom.formal_charge = 0
        implicit_polar_H = adapter.Hydrogens.IMPLICIT
        for a in st_w_polarH.atom:
            if a.element == 'H':
                implicit_polar_H = adapter.Hydrogens.AS_INPUT
                break
        rdkit_mol = rdkit_adapter.to_rdkit(st_w_polarH,
                                           implicitH=implicit_polar_H,
                                           sanitize=adapter.Sanitize_Disable)
        atom_orders = rdkit_mol.GetSubstructMatches(rdkit_mol, uniquify=False)
        # retain only heavy atoms for RMSD
        heavy_atom_filter = partial(
            filter, lambda aid: st_w_polarH.atom[aid].element != 'H')
        get_schro_idx = partial(
            map, lambda atom_ind: rdkit_mol.GetAtomWithIdx(atom_ind).GetIntProp(
                rdkit_adapter.SDGR_INDEX))
        heavy_atom_orders = list(
            map(heavy_atom_filter, map(get_schro_idx, atom_orders)))
        self._heavy_atom_orders = set(tuple(x) for x in heavy_atom_orders)
        self._order0 = next(iter(self._heavy_atom_orders))

    def _postcalc(self, calc, *_):
        centered_fr = self._getCenteredFrame(calc)
        self._cached_st.setXYZ(centered_fr.pos(self._gids))

        if self._fit_ref_pos is None:
            rmsd_call = partial(superimpose, self._ref_st, self._order0,
                                self._cached_st)
        else:
            fit_pos = centered_fr.pos(self._fit_gids)
            pos = align_pos(self._cached_st.getXYZ(copy=False), fit_pos,
                            self._fit_ref_pos)
            self._cached_st.setXYZ(pos)
            rmsd_call = partial(calculate_in_place_rmsd, self._ref_st,
                                self._order0, self._cached_st)
        self._result = min(list(map(rmsd_call, self._heavy_atom_orders)))


def _get_common_resname(resname, mapping=_Const.RESNAME_MAP):
    """
    :param resname: residue name
    :type  mapping: `dict` from string to string
    """
    return mapping[resname] if resname in mapping else resname


def _extract_with_original_id(ct: 'structure.Structure',
                              aids: List) -> 'structure.Structure':
    """
    Set property `constants.ORIGINAL_INDEX` before structure extraction
    """
    tmp_ct = ct.copy()
    for a in tmp_ct.atom:
        a.property[constants.ORIGINAL_INDEX] = a.index
    # Set copy_props=True to copy the PBC properties:
    new_ct = tmp_ct.extract(aids, copy_props=True)
    return new_ct


def get_pdb_protein_bfactor(fsys_ct, aids):
    """
    Calculate per-residue b-factor from pdb data for the selected atoms.

    :type            aids: `list` of `int`
    :param           aids: Atom selections

    :rtype: `numpy.ndarray` of `float`
    """
    st = _extract_with_original_id(fsys_ct, aids)
    bfactors = numpy.zeros(len(st.residue))
    for i, res in enumerate(structure.get_residues_by_connectivity(st)):
        selected = [
            atom.property[constants.ORIGINAL_INDEX] for atom in res.atom
        ]
        res_bfactors = numpy.array(
            list(
                filter(None, [
                    fsys_ct.atom[i].property.get('r_m_pdb_tfactor', 0)
                    for i in selected
                ])))
        if res_bfactors.size:
            bfactors[i] = res_bfactors.mean()
    return bfactors


class ProteinSF(RMSF):
    """
    Per-frame per-residue Square Fluctuation (SF) with respect to averaged
    positions over the trajectory, with optional alignment fitting.

    It returns a tuple of (residue labels, N_frame x N_residue matrix). Each
    matrix entry is the mass-weighted SF averaged over the residue's atoms.
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 aids,
                 fit_aids,
                 fit_ref_pos,
                 in_place=False):
        """
        see `RMSF.__init__` for parameters
        """
        super().__init__(msys_model, cms_model, aids, fit_aids, fit_ref_pos,
                         in_place)

        # find gid indices for the selected aids, grouped by residue
        self._gid_indices, self._residue_labels, self._residue_masses = [], [], []
        st_selection = _extract_with_original_id(cms_model.fsys_ct, aids)
        for res in structure.get_residues_by_connectivity(st_selection):
            selected = [
                atom.property[constants.ORIGINAL_INDEX] for atom in res.atom
            ]
            self._residue_labels.append(
                _prot_atom_label(cms_model, selected[0], res_only=True))
            if cms_model.need_msys:
                res_gids = topo.aids2gids(cms_model,
                                          selected,
                                          include_pseudoatoms=False)
                self._residue_masses.append(
                    numpy.asarray(
                        [msys_model.atom(gid).mass for gid in res_gids]))
            else:
                res_gids = cms_model.convert_to_gids(selected,
                                                     with_pseudo=False)
                self._residue_masses.append(cms_model.get_mass(res_gids))
            self._gid_indices.append(list(map(self._gids.index, res_gids)))
        self._cms_model = cms_model
        self._aids = aids

    def reduce(self, pos_t, *_, **__):
        """
        :type pos_t: List of Nx3 `numpy.ndarray` where N is the number of
                     atoms. The length of the list is the number of frames.

        :rtype : list[string], list[numpy.ndarray]
        :return: residue tags and per-frame per-residue SF matrix
        """
        pos_t_array = numpy.asarray(pos_t, dtype=numpy.float64)
        pos_avg_sq = ((pos_t_array.mean(axis=0))**2).sum(axis=1)  # Nx1
        pos_sq = (pos_t_array**2).sum(axis=2)  # Nt x N
        diff = pos_sq - pos_avg_sq

        # reduce atom-wise SF to mass weighted SF for each residue
        sf_matrix = numpy.empty((len(pos_t), len(self._residue_labels)))
        for i, (indices, masses) in enumerate(
                zip(self._gid_indices, self._residue_masses)):
            sf_matrix[:, i] = numpy.average(diff[:, indices],
                                            weights=masses,
                                            axis=1)
        return self._residue_labels, sf_matrix


class ProteinRMSF(ProteinSF):
    """
    Per-residue Root Mean Square Fluctuation with respect to averaged positions
    over the trajectory, with optional alignment fitting.
    """

    def reduce(self, pos_t, *_, **__):
        """
        :type pos_t: List of Nx3 `numpy.ndarray` where N is the number of
                     atoms. The length of the list is the number of frames.

        :rtype : list[string], list[float]
        :return: residue tags and RMSF for each residue
        """
        atom_rmsf_sq = RMSF.reduce(self, pos_t)**2
        # reduce atom-wise RMSF to mass weighted RMSF for each residue
        rmsf = [
            numpy.sqrt(numpy.average(atom_rmsf_sq[idx], weights=masses))
            for idx, masses in zip(self._gid_indices, self._residue_masses)
        ]
        return self._residue_labels, rmsf


class Dipole(staf.CompositeAnalyzer):
    """
    Electric dipole moment of the selected atoms, in unit of debye.

    The result may not be reliable when the structure of the selected atoms are
    large compared to the simulation box. The unwrapping with respect to
    periodic boundary condition provided by `CenterOf` is based on circular
    mean and may not be adequate.
    """
    EA2DEBYE = 4.802813198  # from (electron charge x angstrom) to debye

    def __init__(self, msys_model, cms_model, aids):
        assert aids, "No atom selected for Dipole calculation."
        if cms_model.need_msys:
            gids = topo.aids2gids(cms_model, aids)
            self._charges = numpy.asarray(
                [msys_model.atom(i).charge for i in gids])
        else:
            gids = cms_model.convert_to_gids(aids)
            self._charges = cms_model.get_charge(gids)
        self._analyzers = [
            Com(msys_model, cms_model, gids=gids, return_unwrapped_atompos=True)
        ]

    def _postcalc(self, *args):
        super()._postcalc(*args)
        com_pos, pos = self._analyzers[0]()
        dipole = numpy.einsum('i, ij->j', self._charges, pos - com_pos)
        self._result = dipole * self.EA2DEBYE


class AxisDirector(staf.GeomAnalyzerBase):
    """
    Basis vector of 3D axis
    """

    def __init__(self, axis):
        """
        :param axis: axis name, 'X', 'Y' or 'Z'
        :type  axis: `str`
        """
        self._result = (numpy.array([['X', 'Y', 'Z']]) == axis).astype(float)


class MomentOfInertia(staf.CompositeAnalyzer):
    """
    Moment of inertia tensor

    Result is 3x3 `numpy.ndarray`
    """

    def __init__(self, msys_model, cms_model, aids):
        """
        :type aids: `list[int]`
        """
        assert aids
        if cms_model.need_msys:
            self._gids = topo.aids2gids(cms_model, aids)
        else:
            self._gids = cms_model.convert_to_gids(aids)
        self._analyzers = [
            Com(msys_model,
                cms_model,
                gids=self._gids,
                return_unwrapped_atompos=True)
        ]

    def _postcalc(self, *args):
        super()._postcalc(*args)
        com_pos, pos = self._analyzers[0]()
        r = pos - com_pos
        mass = self._analyzers[0]._weights
        diag = numpy.einsum('i, ij->', mass, r**2)
        offdiag = numpy.einsum('ij, jk->ik', r.T, mass[:, None] * r)
        self._result = diag * numpy.eye(3) - offdiag


class MomentOfInertiaDirector(staf.CompositeDynamicAslAnalyzer):
    """
    This class calculates the principal moment-of-inertia for each of the
    selected molecules.

    Result: A list of vectors
    """

    def _dyninit(self):
        mol2aids = defaultdict(list)
        for i in self._aids:
            mol2aids[self._cms_model.atom[i].molecule_number].append(i)
        self._analyzers = [
            MomentOfInertia(self._msys_model, self._cms_model, aids)
            for _, aids in sorted(mol2aids.items())
        ]

    def __call__(self):
        return [numpy.linalg.eigh(a())[1][:, -1] for a in self._analyzers]


def _direction(v):
    """
    Normalize vector(s) to unit vector(s).

    :param v: vectors
    :type  v: Nx3 `numpy.array`
    """
    return v / numpy.linalg.norm(v, axis=-1, keepdims=True)


class SmartsDirector(staf.CompositeDynamicAslAnalyzer):
    """
    Direction of atom pairs from SMARTS pattern. The SMARTS pattern should pick
    bonds, i.e., atom pairs, e.g., `smarts='CC'`.

    Convention: The vector is pointing from the first atom to the second.
    """

    def __init__(self, msys_model, cms_model, asl, smarts):
        """
        :type msys_model: `msys.System`
        :type  cms_model: `cms.Cms`
        :type        asl: `str`
        :type     smarts: `str`
        """
        self._smarts = smarts
        self._atom_pairs = adapter.evaluate_smarts(cms_model, smarts,
                                                   adapter.UniqueFilter_Enable)
        super().__init__(msys_model, cms_model, asl)

    def _dyninit(self):
        self._analyzers = []
        self._mol_numbers = []
        aids = set(self._aids)
        for from_aid, to_aid in self._atom_pairs:
            if (from_aid in aids) and (to_aid in aids):
                self._analyzers.append(
                    Vector(self._msys_model, self._cms_model, from_aid, to_aid))
                self._mol_numbers.append(
                    self._cms_model.atom[from_aid].getMolecule().number)

    def __call__(self):
        return _direction(numpy.asarray([a() for a in self._analyzers]))

    def reduce_vec(self, n, m):
        """
        Calculate Legendre polynomial P2 using the inner product of n and m as
        the input.

        :type  n: Nx3 `numpy.array` where N is the number of molecules and the
                  ordering reflects the molecule number.
        :type  m: N'x3 `numpy.array` where N' is the number of chemical bonds.
        :param m: Output of `SmartsDirector` for one frame
        """
        n = [n[mol_num - 1] for mol_num in self._mol_numbers]
        return reduce_vec(n, m)


# FIXME: Looking at the name of the class, I thought SystemDipoleDirector as
#        the dipole vector of the whole
#        simulation system, but it's actually just that of all selected atoms.
#        It would perhaps be more natual to just call it `DipoleDirector`, and
#        people will understand that it's for all selected atoms.
class SystemDipoleDirector(staf.CompositeDynamicAslAnalyzer):
    """
    Direction of electric dipole moment of all the selected atoms
    """

    def _dyninit(self):
        self._analyzers = [
            Dipole(self._msys_model, self._cms_model, self._aids)
        ]

    def __call__(self):
        return _direction(numpy.asarray([a() for a in self._analyzers]))


# FIXME: Call this `MolecularDipoleDirector`?
class DipoleDirector(SystemDipoleDirector):
    """
    Dipole direction for each molecule in the selection
    """

    def _dyninit(self):
        mol2aids = defaultdict(list)
        for i in self._aids:
            mol2aids[self._cms_model.atom[i].molecule_number].append(i)
        self._analyzers = [
            Dipole(self._msys_model, self._cms_model, atoms)
            for atoms in sorted(mol2aids.values())
        ]


class LipidDirector(staf.CompositeAnalyzer):
    """
    Direction of CH bond for carbon atoms on lipid tail
    """

    def __init__(self, msys_model, cms_model, asl, tail_type):
        """
        :type  msys_model: `msys.System`
        :type   cms_model: `cms.Cms`
        :type         asl: `str`
        :param  tail_type: 'sn1', 'sn2', 'all'
        """
        to_atom = lambda c_prefix: (c_prefix + str(s) for s in range(2, 24))
        tail_type_dict = {'sn1': 'C3', 'sn2': 'C2'}
        pdb_prefix = tail_type_dict.get(tail_type, ['C2', 'C3'])
        if tail_type in tail_type_dict:
            pdb_carbon_atoms = to_atom(pdb_prefix)
            _carbon_asl = " and atom.ptype %s"
        else:
            pdb_carbon_atoms = zip(*list(map(to_atom, pdb_prefix)))
            _carbon_asl = " and atom.ptype %s %s"

        def get_CH_pair(c_aid):
            c_atom = cms_model.fsys_ct.atom[c_aid]
            return [(c_aid, a.index)
                    for a in c_atom.bonded_atoms
                    if a.element == 'H']

        self._analyzers = []
        self._CH_counts = [0]
        carbon_asl = asl + _carbon_asl
        for c in pdb_carbon_atoms:
            c_aids = cms_model.select_atom(carbon_asl % c)
            atom_pairs = list(map(get_CH_pair, c_aids))
            for c_gid, h_gid in chain.from_iterable(atom_pairs):
                self._analyzers.append(
                    Vector(msys_model, cms_model, c_gid, h_gid))
            if atom_pairs:
                self._CH_counts.append(len(self._analyzers))

    def __call__(self):
        """
        :rtype: `list` of Nix3 `numpy.array`, where Ni is the number of
                C-H bonds for the i'th lipid C atom over all lipid molecules
        """
        result = []
        for i, j in zip(self._CH_counts, self._CH_counts[1:]):
            result.append(
                _direction(numpy.asarray([a() for a in self._analyzers[i:j]])))
        return result


def reduce_vec(n, m):
    """
    Calculate Legendre polynomial P2 using the inner product of n and m as the
    input.

    :type n: 1x3 or Nx3 `numpy.array`
    :type m: 1x3 or Nx3 `numpy.array`
    """
    return numpy.mean(numpy.einsum('ij,ij->i', n, m)**2) * 1.5 - 0.5


def reduce_vec_list(n, m):
    """
    Calculate Legendre polynomial P2 using the inner product of n and m as the
    input.

    :type n: 1x3 `numpy.array`
    :type m: A `list` of Ni x3 `numpy.array` objects, where the row number
             Ni of each element may not agree
    """
    result = numpy.asarray(
        [numpy.mean(numpy.einsum('ij,ij->i', n, mi)**2) for mi in m])
    return result * 1.5 - 0.5


def reduce_lipid_vec_list(n, m):
    """
    Calculate Legendre polynomial P2 using the inner product of n and m as the
    input. Return its absolute value.

    :type n: 1x3 `numpy.array`
    :type m: A `list` of Ni x3 `numpy.array` objects, where the row number
             Ni of each element may not agree
    """
    return numpy.absolute(reduce_vec_list(n, m))


class OrderParameter(staf.GeomAnalyzerBase):
    """
    Given the director (local or global), and the descriptor (local or global),
    calculate the order parameter <P2> for each frame::

        S = 1/N sum_i ((3 * (n dot m_i)^2 -1) / 2)

    where n is the director vector and m is the descriptor vector.
    For example, n is the z axis and m is the electric dipole moment.

    Typical usage includes::

        Director            Descriptor          result
        Axis                Lipid               avg over carbon type
        Axis                Smarts              avg over bond type
        Axis                Dipole              avg over molecule
        SystemDipole        Dipole              avg over molecule
        Dipole              Smarts              avg over bond type

    To extend its functionality, implement to the `GeomAnalyzerBase` interface
    and provide the reduction rule as callable.
    """

    def __init__(self, vec1, vec2, reducer):
        """
        :param   vec1: a `GeomAnalyzerBase` that computes director
        :param   vec2: a `GeomAnalyzerBase` that computes descriptor
        :type reducer: a callable that reduces the results of `vec1` and
                       `vec2` to order parameter for each frame

        Typically both director and descriptor return Nx3 vectors for each
        frame, where N depends on the context. In this case, one should make
        sure that the orders of these vectors match. For example, if both
        director and descriptor give one vector per molecule, then the
        implementation should guarantee the molecule orders are the same in
        vec1() and vec2().

        For axis director which returns 1x3 vector, reduction with descriptor
        is taken care of by numpy broadcasting. For more complicated cases
        where director and descriptor have incompatible dimensions, the user
        needs to provide special-purpose reduce function, see
        `SmartsDirector.reduce_vec` for example.
        """
        assert isinstance(vec1, staf.GeomAnalyzerBase)
        assert isinstance(vec2, staf.GeomAnalyzerBase)
        assert callable(reducer)
        self._vec1, self._vec2 = vec1, vec2
        self._reduce = reducer

    def _precalc(self, calc):
        self._vec1._precalc(calc)
        self._vec2._precalc(calc)

    def _postcalc(self, *args):
        self._vec1._postcalc(*args)
        self._vec2._postcalc(*args)
        self._result = self._reduce(self._vec1(), self._vec2())


class MoleculeWiseCom(staf.CompositeDynamicAslAnalyzer):
    """
    Calculate the center-of-mass for each of the selected molecules.

    Result: A list of Nx3 numpy arrays, where N is the number of molecules. Note
    that the array size N may vary from frame to frame if the ASL is dynamic.
    """

    def _dyninit(self):
        mol2aids = defaultdict(list)
        for i in self._aids:
            # use fsys_ct due to inconsistency in DESMOND-8687. When that case
            # is resolved, this can be changed back to self._cms_model.atom
            mol2aids[self._cms_model.fsys_ct.atom[i].molecule_number].append(i)
        if self._cms_model.need_msys:
            self._analyzers = [
                Com(self._msys_model,
                    self._cms_model,
                    gids=topo.aids2gids(self._cms_model, aids, False))
                for _, aids in sorted(mol2aids.items())
            ]
        else:
            self._analyzers = [
                Com(self._msys_model,
                    self._cms_model,
                    gids=self._cms_model.convert_to_gids(aids,
                                                         with_pseudo=False))
                for _, aids in sorted(mol2aids.items())
            ]

    def __call__(self):
        result = numpy.empty([len(self._analyzers), 3])
        for i, ana in enumerate(self._analyzers):
            result[i] = ana()
        return result


class AtomicPosition(staf.DynamicAslAnalyzer):
    """
    Extract the positions of the selected atoms.

    Result: A list of Nx3 numpy arrays, where N is the number of atoms. Note
    that the array size N may vary from frame to frame if the ASL is dynamic.
    """

    def __init__(self, msys_model, cms_model, asl=None):
        staf.DynamicAslAnalyzer.__init__(self, msys_model, cms_model, asl)
        if cms_model.need_msys:
            self._gids = (None if self.isDynamic() else sorted(
                topo.aids2gids(cms_model, self._aids,
                               include_pseudoatoms=False)))
        else:
            self._gids = (None if self.isDynamic() else sorted(
                cms_model.convert_to_gids(self._aids, with_pseudo=False)))

    def _postcalc(self, _, __, fr):
        if self._cms_model.need_msys:
            gids = self._gids or sorted(
                topo.aids2gids(
                    self._cms_model, self._aids, include_pseudoatoms=False))
        else:
            gids = self._gids or sorted(
                self._cms_model.convert_to_gids(self._aids, with_pseudo=False))
        self._result = fr.pos(gids)


class SecondaryStructure(staf.MaestroAnalysis):
    """
    Calculate the secondary-structure property for selected atoms. The result is
    a list of `int` numbers, each of which corresponds to a selected atoms and
    is one of the following values:

      SecondaryStructure.NONE
      SecondaryStructure.LOOP
      SecondaryStructure.HELIX
      SecondaryStructure.STRAND
      SecondaryStructure.TURN

    The selected atoms can be obtained by calling the `aids` method.
    """

    NONE = mm.MMCT_SS_NONE
    LOOP = mm.MMCT_SS_LOOP
    HELIX = mm.MMCT_SS_HELIX
    STRAND = mm.MMCT_SS_STRAND
    TURN = mm.MMCT_SS_TURN

    def __init__(self, msys_model, cms_model, aids: List[int]):
        """
        :param aids: IDs of atoms to calculate the secondary-structure property for
        """
        super().__init__(msys_model, cms_model)
        self._aids = aids
        # order residue from N->C
        st = _extract_with_original_id(cms_model.fsys_ct, aids)
        self._ordered_aids, self._labels = [], []
        for r in structure.get_residues_unsorted(st):
            if r.isStandardResidue():
                a1_aid = r.atom[1].property[constants.ORIGINAL_INDEX]
                self._ordered_aids.append(a1_aid)
                self._labels.append(_prot_atom_label(cms_model, a1_aid, True))

    def _postcalc(self, calc, *_):
        fsys_ct = self._getCenteredCt(calc)
        mm.mmss_initialize(mm.error_handler)
        mm.mmss_assign(fsys_ct.handle, 1)
        mm.mmss_terminate()
        self._result = [
            fsys_ct.atom[i].secondary_structure for i in self._ordered_aids
        ]

    def reduce(self, results, *_, **__):
        return self._labels, results


class SolventAccessibleSurfaceAreaByResidue(staf.MaestroAnalysis):
    """
    Calculate the relative SASA broken down by residues. The values are relative
    to the average SASAs as given by
    `SolventAccessibleSurfaceAreaByResidue.DIPEPTIDE_SASA`.

    The result is a 2-tuple: ([residue-names], [relative-SASAs]), where
    relative-SASAs has the structure of [[relative-SASA for each residue] for each frame]
    """

    # Average SASA and stdev of all common residue types
    DIPEPTIDE_SASA = {
        "ACE": (115.4897, 3.5972),
        "NMA": (97.3748, 4.0446),
        "ALA": (128.7874, 4.7150),
        "ARG": (271.5978, 9.5583),
        "ASH": (175.7041, 5.1167),
        "ASN": (179.5393, 4.6320),
        "ASP": (173.4664, 6.9882),
        "CYS": (158.1909, 5.3923),
        "CYX": (99.3829, 10.7089),
        "GLH": (203.2443, 6.2765),
        "GLN": (208.6171, 6.5794),
        "GLU": (201.4660, 6.9328),
        "GLY": (94.1021, 5.1977),
        "HIE": (218.7990, 5.6097),
        "HIS": (208.8269, 5.9202),
        "HID": (208.8269, 5.9202),
        "HIP": (221.1223, 8.3364),
        "ILE": (207.2248, 5.0012),
        "LEU": (211.8823, 5.1490),
        "LYN": (235.5351, 6.8589),
        "LYS": (242.8734, 9.3510),
        "MET": (218.5396, 6.9879),
        "PHE": (243.4793, 5.9699),
        "PRO": (168.7830, 5.5848),
        "SER": (140.6706, 4.9089),
        "THR": (169.0046, 4.9049),
        "TRP": (287.0895, 6.8920),
        "TYR": (256.8637, 6.2782),
        "VAL": (181.2543, 4.8640),
        "UNK": (189.9610, 6.3732)  # This is just an average of the knowns.
    }

    def __init__(self, msys_model, cms_model, asl, resolution=None):
        super().__init__(msys_model, cms_model)
        self._aids = cms_model.select_atom(asl)
        self._resolution = (resolution or
                            (0.1 if is_small_struc(self._aids) else 0.2))
        self._residue_sasa = []
        self._residue_name = []

        ct = _extract_with_original_id(cms_model.fsys_ct, self._aids)
        for res in structure.get_residues_by_connectivity(ct):
            aid = res.atom[1].property[constants.ORIGINAL_INDEX]
            label = _prot_atom_label(cms_model, aid, res_only=True)
            resname = label.split(':')[1].split('_')[0]
            self._residue_sasa.append(self.DIPEPTIDE_SASA[resname][0])
            self._residue_name.append(label)
        self._residue_sasa = numpy.asarray(self._residue_sasa)

    def _postcalc(self, calc, *_):
        fsys_ct = self._getCenteredCt(calc)
        sasa = calculate_sasa_by_residue(fsys_ct,
                                         resolution=self._resolution,
                                         atoms=self._aids,
                                         exclude_water=True)
        self._result = (numpy.asarray(sasa) / self._residue_sasa).tolist()

    def reduce(self, results, *_, **__):
        return self._residue_name, results


class MolecularSurfaceArea(staf.CenteredSoluteAnalysis):
    """
    Calculate the molecular surface area. The result is a single scalar number
    per frame.
    """

    def __init__(self, msys_model, cms_model, asl, grid_spacing=None):
        """
        :type  asl: `str`
        :param asl: ASL expression to select atoms whose secondary-structure
                property is of interest.
        """
        super().__init__(msys_model, cms_model)
        self._aids = cms_model.select_atom(asl)
        if cms_model.need_msys:
            self._gids = topo.aids2gids(cms_model,
                                        self._aids,
                                        include_pseudoatoms=False)
        else:
            self._gids = cms_model.convert_to_gids(self._aids,
                                                   with_pseudo=False)
        self._grid_spacing = (grid_spacing or
                              (0.25 if is_small_struc(self._aids) else 0.6))

        # `mmsurf' module is only used here, reducing the overall dependency.
        import schrodinger.infra.mmsurf as mmsurf
        self._mmsurf = mmsurf
        self._molsurf = mmsurf.mmsurf_molsurf
        self._surfarea = mmsurf.mmsurf_get_surface_area

        # FIXME: Do we allow customizations for the following?
        self._surface_type = mmsurf.MOLSURF_MOLECULAR
        self._vdw_scaling = 1.0
        self._probe_radius = 1.4
        self._area_only = False  # False means to also calculate the normals.
        self._bs = mmbitset.Bitset(size=len(self._aids))
        self._bs.fill()
        self._cached_st = _extract_with_original_id(cms_model.fsys_ct,
                                                    self._aids)

    def _postcalc(self, calc, *_):
        centered_fr = self._getCenteredFrame(calc)
        self._cached_st.setXYZ(centered_fr.pos(self._gids))
        surf = self._mmsurf.mmsurf_molsurf(self._cached_st, self._bs,
                                           self._grid_spacing,
                                           self._probe_radius,
                                           self._vdw_scaling,
                                           self._surface_type, self._area_only)
        self._result = self._mmsurf.mmsurf_get_surface_area(surf)
        self._mmsurf.mmsurf_delete(surf)


class SolventAccessibleSurfaceArea(staf.MaestroAnalysis):
    """
    Calculate solvent accessible surface area for selected atoms.
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 asl,
                 exclude_asl=None,
                 resolution=None):
        super().__init__(msys_model, cms_model)
        self._aids = cms_model.select_atom(asl)
        self._exclude_aids = (exclude_asl and
                              (cms_model.select_atom(exclude_asl) or None))
        self._resolution = (resolution or
                            (0.1 if is_small_struc(self._aids) else 0.2))

    def _postcalc(self, calc, *_):
        fsys_ct = self._getCenteredCt(calc)
        self._result = calculate_sasa(fsys_ct,
                                      atoms=self._aids,
                                      exclude_water=True,
                                      exclude_atoms=self._exclude_aids,
                                      resolution=self._resolution)


class PolarSurfaceArea(staf.CenteredSoluteAnalysis):
    """
    Calculate polar surface area for selected atoms.

    N.B.: Only O and N atoms are considered as polar atoms in this
    implementation.
    """

    def __init__(self, msys_model, cms_model, asl, resolution=None):
        super().__init__(msys_model, cms_model)

        # Strips off H atoms.
        asl = "%s and (not atom.ele H)" % asl
        self._aids = cms_model.select_atom(asl)
        if cms_model.need_msys:
            self._gids = topo.aids2gids(cms_model,
                                        self._aids,
                                        include_pseudoatoms=False)
        else:
            self._gids = cms_model.convert_to_gids(self._aids,
                                                   with_pseudo=False)

        # Selects polar atoms.
        # FIXME: This code basically copies the original algorithm that only
        #        considers O and N for polar atoms. Shall we include S as well?
        #        What about halogens?
        self._polar_aids = [
            i for i, aid in enumerate(self._aids, start=1)
            if (cms_model.atom[aid].element in [
                "O",
                "N",
            ])
        ]
        self._resolution = (resolution or
                            (0.1 if is_small_struc(self._aids) else 0.2))
        self._cached_st = _extract_with_original_id(cms_model.fsys_ct,
                                                    self._aids)
        self._cached_st.retype()

    def _postcalc(self, calc, *_):
        centered_fr = self._getCenteredFrame(calc)
        self._cached_st.setXYZ(centered_fr.pos(self._gids))
        self._result = sum(
            calculate_sasa_by_atom(self._cached_st,
                                   atoms=self._polar_aids,
                                   resolution=self._resolution))


def _select_asl(asl, data, custom):
    """
    This function is auxiliary to functions that work with
    `staf.CustomMaestroAnalysis` to provide ASL selection for each frame. It
    uses the centered full system CT from `staf.center_ct` to evaluate ASL.

    :type    data: `dict`
                   key = `(msys_model, cms_model, center_gids)`
                   value = `list` of `int`, i.e., AIDs of the selected atoms
    :type  custom: `_CustomCalc`
    :param custom: It contains the `MaestroAnalysis` result.

    :return: Updated `data`, where values are updated for the given frame.
    """
    for key in data:
        fsys_ct = custom[staf.center_ct][key]
        data[key] = evaluate_asl(fsys_ct, asl)
    return data


def _res_near_lig(lig_asl, prot_asl, cutoff, data, _, __, custom):
    """
    Select residue atoms near ligand according to distance cutoff. It works
    with `CustomMaestroAnalysis`. Also see `_select_asl`.
    """
    asl = ('not (%s) and (fillres within %f (%s)) and (%s)' %
           (lig_asl, cutoff, lig_asl, prot_asl))
    return _select_asl(asl, data, custom)


def _wat_near_lig(lig_asl, cutoff, data, _, __, custom):
    """
    Select water atoms near ligand according to distance cutoff. It works with
    `CustomMaestroAnalysis`. Also see `_select_asl`.
    """
    asl = 'fillres (water and within %f (%s))' % (cutoff, lig_asl)
    return _select_asl(asl, data, custom)


def _ion_near_lig(ion_asl, lig_not_CH, lig_asl, cutoff, data, _, __, custom):
    """
    Select ions near ligand according to distance cutoff. It works with
    `CustomMaestroAnalysis`. Also see `_select_asl`.

    :type  lig_not_CH: `list` of `int`
    :param lig_not_CH: AIDs of ligand atoms that are not carbon or hydrogen
    """
    asl = ('((%s) and within %f (%s)) and not (%s)' %
           (ion_asl, cutoff, _aids2asl(lig_not_CH), lig_asl))
    return _select_asl(asl, data, custom)


def _hydrophobic_res_near_lig(lig_asl, cutoff, prot_cid, data, _, __, custom):
    """
    Select hydrophobic residues near ligand according to distance cutoff. It
    works with `CustomMaestroAnalysis`. Also see `_select_asl`.

    :type  prot_cid: `_KeyPartial`
    :param prot_cid: it is essentially the function `_res_near_lig`. Here it
                     is used as dictionary key to retrieve the result of
                     `_res_near_lig` from the `_CustomCalc` dictionary custom.
    """
    for key in data:
        res_near_lig_asl = _aids2asl(custom[prot_cid][key])
        asl = ('not (%s) and ((fillres ((within %f (%s)) and '
               '((res. %s) and (sidechain)))) and (sidechain) and (%s))' %
               (lig_asl, cutoff, lig_asl, _Const.HYDROPHOBIC_TYPES,
                res_near_lig_asl))
        fsys_ct = custom[staf.center_ct][key]
        data[key] = evaluate_asl(fsys_ct, asl)
    return data


def _prot_near_ion(cutoff, ion_cid, prot_cid, data, _, __, custom):
    """
    Select protein atoms near ions according to distance cutoff. It works with
    `CustomMaestroAnalysis`. Also see `_select_asl`.

    :type   ion_cid: `_KeyPartial`
    :param  ion_cid: it is essentially the function `_ion_near_lig`. Here it
                     is used as dictionary key to retrieve the result of
                     `_ion_near_lig` from the `_CustomCalc` dictionary custom.
    :type  prot_cid: `_KeyPartial`
    :param prot_cid: it is essentially the function `_res_near_lig`. Here it
                     is used as dictionary key to retrieve the result of
                     `_res_near_lig` from the `_CustomCalc` dictionary custom.
    """
    for key in data:
        ion_aids = custom[ion_cid][key]
        if not ion_aids:
            data[key] = []
            continue
        ion_asl = _aids2asl(ion_aids)
        res_near_lig_asl = _aids2asl(custom[prot_cid][key])
        asl = ('((%s) and not a.e C,H) and within %f (%s)' %
               (res_near_lig_asl, cutoff, ion_asl))
        fsys_ct = custom[staf.center_ct][key]

        data[key] = evaluate_asl(fsys_ct, asl)
    return data


class HydrogenBondFinder(staf.MaestroAnalysis):
    """
    Find hydrogen bonds between two sets of atoms. The result has the structure
    of [[(acceptor atom ID, donor atom ID) for each H bond] for each frame].

    Basic usage:

      ana = HydrogenBondFinder(msys_model, cms_model, aids1, aids2)
      results = analyze(tr, ana)
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 aids1,
                 aids2,
                 max_dist=_Const.HBOND_CUTOFF,
                 min_donor_angle=_Const.HBOND_MIN_D_ANGLE,
                 min_acceptor_angle=_Const.HBOND_MIN_A_ANGLE,
                 max_acceptor_angle=_Const.HBOND_MAX_A_ANGLE):
        """
        :type msys_model: `msys.System`
        :type cms_model: `cms.Cms`

        :param aids1: A list of atom indices for the first atom set. All atoms
            will be used if it is None.
        :type aids1: list or None

        :param aids2: A list of atom indices for the second atom set. All atoms
            will be used if it is None.
        :type aids2: list or None
        """
        super().__init__(msys_model, cms_model)
        self._aids1 = aids1
        self._aids2 = aids2
        self._get_hbonds = partial(get_hydrogen_bonds,
                                   atoms1=aids1,
                                   atoms2=aids2,
                                   max_dist=max_dist,
                                   min_donor_angle=min_donor_angle,
                                   min_acceptor_angle=min_acceptor_angle,
                                   max_acceptor_angle=max_acceptor_angle)

    def _postcalc(self, calc, *_):

        # If any set of atoms are missing, there are no hydrogen bonds. All
        # atoms will be used if any set of atoms is None and do not return
        # empty results in that case.
        if self._aids1 == [] or self._aids2 == []:
            self._result = []
            return

        fsys_ct = self._getCenteredCt(calc)
        # `_get_hbonds` returns a list of (donor, acceptor)'s, while we want
        # them to be (acceptor, donor)'s.
        self._result = [(int(a), int(d)) for d, a in self._get_hbonds(fsys_ct)]


class HalogenBondFinder(staf.MaestroAnalysis):
    """
    Find halogen bonds between two sets of atoms. The result has the structure
    of [[(acceptor atom ID, donor atom ID) for each bond] for each frame].

    Basic usage:

      ana = HalogenBondFinder(msys_model, cms_model, protein_aids, ligand_aids)
      results = analyze(tr, ana)
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 aids1,
                 aids2,
                 max_dist=_Const.HALOGEN_BOND_CUTOFF,
                 min_donor_angle=_Const.HALOGEN_BOND_MIN_D_ANGLE,
                 min_acceptor_angle=_Const.HALOGEN_BOND_MIN_A_ANGLE,
                 max_acceptor_angle=_Const.HALOGEN_BOND_MAX_A_ANGLE):
        """
        :type msys_model: `msys.System`
        :type  cms_model: `cms.Cms`
        :type      aids1: `list` of atom AIDs
        :type      aids2: `list` of atom AIDs
        """
        super().__init__(msys_model, cms_model)
        self._get_xbonds = partial(get_halogen_bonds,
                                   atoms1=aids1,
                                   atoms2=aids2,
                                   max_dist=max_dist,
                                   min_donor_angle=min_donor_angle,
                                   min_acceptor_angle=min_acceptor_angle,
                                   max_acceptor_angle=max_acceptor_angle)

    def _postcalc(self, calc, *_):
        fsys_ct = self._getCenteredCt(calc)
        # `_get_xbonds` returns a list of (donor, acceptor)'s, while we want
        # them to be (acceptor, donor)'s.
        self._result = [(int(a), int(d)) for d, a in self._get_xbonds(fsys_ct)]


# FIXME: maybe this is available somewhere in mmshare
def _aids2asl(aids: List[int]) -> str:
    """
    Convert a list of AIDs to an ASL string.
    """
    if not aids:
        return 'not all'
    return '(atom.num %s)' % ','.join(map(str, aids))


def get_ligand_fragments(lig_ct):
    """
    Decompose the ligand into several fragments using the murcko rules.

    :type lig_ct: `schrodinger.structure.Structure`

    :return: ligand fragments
    :rtype:  `list`. Each element is a `list` of `int`.
    """
    from schrodinger.structutils import createfragments
    frag_creator = createfragments.FragmentCreator(atoms=2,
                                                   bonds=100,
                                                   carbon_hetero=True,
                                                   maxatoms=350,
                                                   removedup=False,
                                                   murcko=False,
                                                   recap=True,
                                                   complete=False,
                                                   add_h=False,
                                                   verbose=False)
    fragments = frag_creator.fragmentStructureForEA(lig_ct) or [lig_ct]
    return [[a.property[constants.ORIGINAL_INDEX]
             for a in frag.atom]
            for frag in fragments]


# FIXME: Similar logic exists in mmshare as well. Maybe move it to mmshare and
#        expose it.
def _prot_atom_label(fsys_ct, aid, res_only=False):
    """
    Return the protein atom or residue label.

    :param res_only: set `True` to label up to the residue
    """
    atom = fsys_ct.atom[aid]
    chain = atom.chain
    if chain == ' ':
        chain = '_'
    resname = atom.pdbres.strip()
    if resname == '':
        resname = 'UNK'
    label = '%s:%s_%d' % (chain, _get_common_resname(resname), atom.resnum)
    if res_only:
        return label
    return label + ':' + atom.pdbname.strip()


def _has_unique_PDB_names(ct):
    """
    Return True if all atoms have unique s_m_pdb_atom_name

    :type ct: `schrodinger.structure.Structure`
    """
    pdbnames = set(a.pdbname.strip() for a in ct.atom)
    return len(pdbnames) == ct.atom_total


def _lig_atom_label(fsys_ct, aid, frag_idx, is_name_unique):
    """
    Return ligand atom label with fragmentation information.

    :param       frag_idx: mapping AID to ligand fragment index
    :type        frag_idx: `int`
    :param is_name_unique: `True` if all atoms have different PDB names
    """
    atom = fsys_ct.atom[aid]
    atomname = atom.pdbname.strip()
    if not is_name_unique or atomname == '':
        atomname = atom.element.strip() + str(aid)
    label = 'L-FRAG_%d:%s' % (frag_idx, atomname)
    return label


def _ligand_h_to_heavy_map(st: structure.Structure,
                           lig_aids: List[int]) -> Dict[int, int]:
    """
    Generate a map of ligand hydrogen atom AIDs to their heavy atom AIDs.
    """
    h_to_heavy = {}
    for aid in lig_aids:
        atom = st.atom[aid]
        if atom.element == 'H':
            # hydrogen atom is bonded to only one heavy atom
            heavy_atom = next(atom.bonded_atoms)
            h_to_heavy[aid] = heavy_atom.index
    return h_to_heavy


def _cleanup_water_ligand_distance(result):
    """
    Remove `WaterBridges` results from `_WatLigFragDistance` results.

    :param result: The result to be cleaned up. It must contain the results
                   from the `_WatLigFragDistance`, `WaterBridges` analyzers,
                   as keyed by "WaterBridgeResult" and "LigWatResult",
                   respectively.
    :type  result: `dict`
    """
    exclude = {wb.wat_res_num for wb in result['WaterBridgeResult']}
    f = lambda lw: lw.wat_res_num not in exclude
    result['LigWatResult'] = list(filter(f, result['LigWatResult']))


def _cleanup_polar_inter(result: Dict[str, List[object]], label_res: Callable,
                         h2heavy: Dict[int, int]):
    """
    Remove `ProtLigHbondInter` and `WaterBridges` results from
    `_ProtLigSaltBridges` results.

    Since the `_ProtLigSaltBridges` report results between heavy atoms,
    we have a map of ligand hydrogens to their heavy atoms. This data structure
    is stored in `h2heavy`.

    :param result: The result to be cleaned up. It must contain the results
                   from the `_ProtLigSaltBridges`, `WaterBridges` and
                   `ProtLigHbondInter` analyzers, as keyed by "PolarResult",
                   "WaterBridgeResult", and "HBondResult" respectively.
    :param label_res: label up to the residue
    :type  label_res: callable
    :param h2heavy: mapping ligand hydrogen atom aid to heavy atom aid
    """
    sb_to_exclude = {
        (label_res(inter.prot_aid), h2heavy.get(inter.lig_aid, inter.lig_aid))
        for inter in result['HBondResult'] + result['WaterBridgeResult']
    }
    # keep on polar interactions that are not in sb_to_exclude
    f = lambda sb: ((label_res(sb.prot_aid), sb.lig_aid) not in sb_to_exclude)
    result['PolarResult'] = list(filter(f, result['PolarResult']))


def _cleanup_hydrophobic_inter(result, label_res, aid2frag):
    """
    Remove `ProtLigHbondInter` and `ProtLigPiInter` results from
    `_HydrophobicInter` results.

    :param    result: The result to be cleaned up. It must contain the results
                      from the `_HydrophobicInter`, `ProtLigPiInter` and
                      `ProtLigHbondInter` ananlyzers, as keyed by
                      "HydrophobicResult" "PiPiResult", "PiCatResult", and
                      "HBondResult" respectively.
    :type     result: `dict`
    :param label_res: label up to the residue
    :type  label_res: callable
    :param  aid2frag: mapping ligand atom aid to fragment index
    :type   aid2frag: `dict`
    """
    exclude_hb = {(label_res(hb.prot_aid), aid2frag[hb.lig_aid])
                  for hb in result['HBondResult']}
    exclude_pp = {
        (label_res(pp.ca_aid), pp.frag_idx) for pp in result['PiPiResult']
    }
    exclude_pc = {label_res(pc[0]) for pc in result['PiCatResult']}
    exclude = exclude_hb | exclude_pp
    f = lambda hi: ((label_res(hi.ca_aid), hi.frag_idx) not in exclude and
                    label_res(hi.ca_aid) not in exclude_pc)
    result['HydrophobicResult'] = list(filter(f, result['HydrophobicResult']))


class _KeyPartial(partial):
    """
    Extend partial such that the function instead of the partial instance is
    used as dictionary key.
    """

    def __hash__(self):
        return hash(self.func)

    def __eq__(self, other):
        return self.func == other.func


# This data structure is shared among all protein-ligand interaction analyzers
# to memoize the `_get_lig_properties` function.
_Ligand = namedtuple("_Ligand", "aids frags rings aid2frag aid2label, h2heavy")


@lru_cache()
def _get_lig_properties(cms_model, lig_asl):
    aids = cms_model.select_atom(lig_asl)
    ct = _extract_with_original_id(cms_model.fsys_ct, aids)
    frags = get_ligand_fragments(ct)
    aid2frag = {aid: i for i, frag in enumerate(frags) for aid in frag}
    aid2label = {
        i: _lig_atom_label(cms_model, i, aid2frag[i], _has_unique_PDB_names(ct))
        for i in aids
    }
    h2heavy = _ligand_h_to_heavy_map(cms_model.fsys_ct, aids)
    return _Ligand(aids, frags, [], aid2frag, aid2label, h2heavy)


class _HydrophobicInter(staf.CompositeAnalyzer):
    """
    Compute protein-ligand hydrophobic interaction candidates: protein atoms on
    hydrophobic residues and ligand aromatic/aliphatic atoms within hydrophobic
    distance cutoff. To further exclude hydrogen bonds and pi-pi interactions,
    call `_cleanup_hydrophobic_inter`. Use the class `HydrophobicInter`
    instead for automated cleaning up.

    The result has the structure of::

        [{'HydrophobicResult': [`_HydrophobicInter.Result` for each interaction
            candidate]} for each frame]

    """
    # ca_aid is the AID of alpha carbon, frag_idx is the index of ligand fragment.
    Result = namedtuple("_Hydrophobic", "ca_aid frag_idx")

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 contact_cutoff=_Const.CONTACT_CUTOFF,
                 hydrophobic_search_cutoff=_Const.HYDROPHOBIC_SEARCH_CUTOFF,
                 hydrophobic_cutoff=_Const.HYDROPHOBIC_CUTOFF):
        res_near_lig = _KeyPartial(_res_near_lig, lig_asl, prot_asl,
                                   contact_cutoff)
        hydrophobic_res_near_lig = _KeyPartial(_hydrophobic_res_near_lig,
                                               lig_asl,
                                               hydrophobic_search_cutoff,
                                               res_near_lig)
        sel0 = staf.CustomMaestroAnalysis(msys_model, cms_model, res_near_lig)
        sel1 = staf.CustomMaestroAnalysis(msys_model, cms_model,
                                          hydrophobic_res_near_lig)
        # res_near_lig selection should proceed hydrophobic_res_near_lig
        self._analyzers = [sel0, sel1]
        self._ligand = _get_lig_properties(cms_model, lig_asl)
        self._cutoff_sq = hydrophobic_cutoff**2

        # record the gids of the aromatic/aliphatic carbon in ligand fragments
        is_sp2sp3 = lambda i: cms_model.atom[i].atom_type_name in ['C2', 'C3']
        frags = self._ligand.frags
        self._sp2sp3_C_aids = [list(filter(is_sp2sp3, frag)) for frag in frags]

        # these two public properties are used for cleanup
        self.aid2frag = self._ligand.aid2frag
        self.label_res = partial(_prot_atom_label, cms_model, res_only=True)

    def _postcalc(self, calc, pbc, _):
        super()._postcalc(calc, pbc, _)
        hydrophobic_res_near_lig, fsys_ct = self._analyzers[1]()
        if not (hydrophobic_res_near_lig and self._sp2sp3_C_aids):
            self._result = {'HydrophobicResult': []}
            return
        frags_pos = [
            numpy.asarray([fsys_ct.atom[aid].xyz
                           for aid in aids]) if aids else numpy.empty([0, 3])
            for aids in self._sp2sp3_C_aids
        ]
        # find unique hydrophobic interactions
        # up to (protein residue, ligand fragment)
        hydrophobic, visited = [], set()
        tmp_ct = fsys_ct.copy()
        for a in tmp_ct.atom:
            a.property[constants.ORIGINAL_INDEX] = a.index
        for (frag_idx, frag_pos), aid in product(enumerate(frags_pos),
                                                 hydrophobic_res_near_lig):
            res_atom = tmp_ct.atom[aid]
            res = res_atom.getResidue()
            valid_atom_type = res_atom.atom_type_name in ['C2', 'C3']
            if valid_atom_type and (res.resnum, frag_idx) not in visited:
                pos = res_atom.xyz
                if pbc.isWithinCutoff(pos, frag_pos, self._cutoff_sq):
                    ca = res.getAlphaCarbon()
                    if ca is None:
                        continue
                    ca_aid = ca.property[constants.ORIGINAL_INDEX]
                    hydrophobic.append(self.Result(ca_aid, frag_idx))
                    visited.add((res.resnum, frag_idx))
        self._result = {'HydrophobicResult': hydrophobic}


class HydrophobicInter(staf.CompositeAnalyzer):
    """
    Calculate hydrophobic interactions between protein and ligand, with hbonds
    and pi-pi interactions excluded.

    The result has the structure of::

        [{
        'HydrophobicResult': [`_HydrophobicInter.Result` for each interaction],
        'PiPiResult': [`ProtLigPiInter.Pipi` for each interaction],
        'PiCatResult': [`ProtLigPiInter.PiLCatP` or `ProtLigPiInter.PiPCatL` for each interaction],
        'HBondResult': [`ProtLigHbondInter.Result` for each interaction],
          } for each frame]

    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 contact_cutoff=_Const.CONTACT_CUTOFF,
                 hydrophobic_search_cutoff=_Const.HYDROPHOBIC_SEARCH_CUTOFF,
                 hbond_cutoff=_Const.HBOND_CUTOFF):
        hi = _HydrophobicInter(msys_model, cms_model, prot_asl, lig_asl,
                               contact_cutoff, hydrophobic_search_cutoff)
        pi = ProtLigPiInter(msys_model, cms_model, prot_asl, lig_asl,
                            contact_cutoff)
        hb = ProtLigHbondInter(msys_model, cms_model, prot_asl, lig_asl,
                               contact_cutoff, hbond_cutoff)
        self._analyzers = [hi, pi, hb]

    def _postcalc(self, *args):
        super()._postcalc(*args)
        self._result = {}
        for a in self._analyzers:
            self._result.update(a())
        _cleanup_hydrophobic_inter(self._result, self._analyzers[0].label_res,
                                   self._analyzers[0].aid2frag)


class HydrophobicInterFinder(staf.MaestroAnalysis):
    """
    This method is adopted from the following script:
    mmshare/python/common/display_hydrophobic_interactions.py
    While a similar analyzer `HydrophobicInter` is used specifically for finding
    Protein-Ligand hydrophobic interactions, this method seems more general. We
    may want to transition using this method for detecting Protein-Ligand
    hydrophobic interactions.

    The frames are first centered on `aids1` selection.
    """

    HYDROPHOB_ASL = 'SMARTS.[#6!$([C,c][O,o])&!$([C,c][N,n]),S&^3,P,Cl,Br,I]'

    def __init__(self,
                 msys_model,
                 cms_model,
                 aids1,
                 aids2,
                 good_cutoff_ratio=_Const.GOOD_CONTACTS_CUTOFF_RATIO,
                 bad_cutoff_ratio=_Const.BAD_CONTACTS_CUTOFF_RATIO):

        def filter_hydrophobic_atoms(aids_sel):
            st = _extract_with_original_id(cms_model.fsys_ct, aids_sel)
            return [
                st.atom[i].property[constants.ORIGINAL_INDEX]
                for i in evaluate_asl(st, self.HYDROPHOB_ASL)
            ]

        self.good_cutoff_ratio = good_cutoff_ratio
        self.bad_cutoff_ratio = bad_cutoff_ratio
        # select just the hydrophobic atoms
        self.aids1 = filter_hydrophobic_atoms(aids1)
        self.aids2 = filter_hydrophobic_atoms(aids2)

        if cms_model.need_msys:
            self.gids1 = topo.aids2gids(cms_model,
                                        self.aids1,
                                        include_pseudoatoms=False)
            self.gids2 = topo.aids2gids(cms_model,
                                        self.aids2,
                                        include_pseudoatoms=False)
        else:
            self.gids1 = cms_model.convert_to_gids(self.aids1,
                                                   with_pseudo=False)
            self.gids2 = cms_model.convert_to_gids(self.aids2,
                                                   with_pseudo=False)
        self._gids2aids = dict(zip(self.gids1, self.aids1))
        self._gids2aids.update(dict(zip(self.gids2, self.aids2)))

        radii1 = numpy.array([cms_model.atom[i].radius for i in self.aids1])
        radii2 = numpy.array([cms_model.atom[i].radius for i in self.aids2])
        self.radii_pair_sum = numpy.ravel(radii1[:, None] + radii2[None, :])
        self.aids_pairs = numpy.array(list(product(self.aids1, self.aids2)))

        self._key = (msys_model, cms_model, tuple(self.gids1))

    def _precalc(self, calc):
        # center frame on aids1 selection
        calc.addCustom(staf.center_fr, self._key)

    def _postcalc(self, calc, *_):
        fr = self._getCenteredFrame(calc)

        dist = numpy.ravel(cdist(fr.pos(self.gids1), fr.pos(self.gids2)))
        cont_ratio = dist / self.radii_pair_sum
        pairs = (cont_ratio > self.bad_cutoff_ratio) & (cont_ratio <
                                                        self.good_cutoff_ratio)
        self._result = self.aids_pairs[pairs].tolist()


class SaltBridgeFinder(staf.MaestroAnalysis):
    """
    Find salt bridges present between two sets of atoms. This class wraps
    around the `get_salt_bridges` function.

    The result has the structure of::

        [[(anion atom, cation atom) for each bridge] for each frame]

    where the atoms are `structure._StructureAtom`.
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 aids1,
                 aids2,
                 cutoff=_Const.SALT_BRIDGE_CUTOFF):
        super().__init__(msys_model, cms_model)
        self._get_sb = partial(get_salt_bridges,
                               group1=aids1,
                               group2=aids2,
                               cutoff=cutoff)

    def _postcalc(self, calc, *_):
        self._result = [
            (anion.index, cation.index)
            for anion, cation in self._get_sb(self._getCenteredCt(calc))
        ]


class _ProtLigSaltBridges(staf.CompositeAnalyzer):
    """
    Calculate protein-ligand polar interaction candidates. To further exclude
    hydrogen bonds and water bridges, call `_cleanup_polar_inter`. Use the class
    `ProtLigPolarInter` instead for automated cleaning up.

    The result has the structure of
    [{
    'PolarResult': [`_ProtLigSaltBridges.Result` for each bridge]
    } for each frame]
    """
    # Here prot_aid and lig_aid are the AID of the protein atom and ligand atom,
    # polar_type is a string that denotes the side chain/backbone information,
    # distance is the distance between the protein atom and the ligand atom.
    Result = namedtuple("_PolarInter", "prot_aid polar_type lig_aid distance")

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 contact_cutoff=_Const.CONTACT_CUTOFF,
                 salt_bridge_cutoff=_Const.SALT_BRIDGE_CUTOFF):

        res_near_lig = _KeyPartial(_res_near_lig, lig_asl, prot_asl,
                                   contact_cutoff)
        selection = staf.CustomMaestroAnalysis(msys_model, cms_model,
                                               res_near_lig)
        self._analyzers = [selection]
        self._ligand = _get_lig_properties(cms_model, lig_asl)
        self._get_sb = partial(get_salt_bridges,
                               group1=self._ligand.aids,
                               cutoff=salt_bridge_cutoff)

        self._backbone = cms_model.select_atom(BACKBONE_ASL)
        # this public property is used for cleanup
        self.label_res = partial(_prot_atom_label, cms_model, res_only=True)

    def _postcalc(self, calc, pbc, fr):
        super()._postcalc(calc, pbc, fr)
        prot_aids, fsys_ct = self._analyzers[0]()

        polar = []
        for an, cat in self._get_sb(fsys_ct, group2=prot_aids):
            dist = numpy.linalg.norm(
                pbc.calcMinimumDiff(numpy.array(an.xyz), numpy.array(cat.xyz)))
            p_aid, l_aid = an.index, cat.index
            if an.index in self._ligand.aids:
                p_aid, l_aid = l_aid, p_aid
            polar_type = 'b' if p_aid in self._backbone else 's'
            polar.append(
                self.Result(p_aid, polar_type, l_aid, dist.astype(float)))
        self._result = {'PolarResult': polar}


class ProtLigPolarInter(staf.CompositeAnalyzer):
    """
    Calculate polar interactions between protein and ligand, with hbonds and
    water bridges excluded.

    The result has the structure of
    [{
    'PolarResult': [`_ProtLigSaltBridges.Result` for each bridge],
    'HBondResult': [`ProtLigHbondInter.Result` for each H bond],
    'WaterBridgeResult': [`WaterBridges.Result` for each bridge],
    } for each frame]
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 contact_cutoff=_Const.CONTACT_CUTOFF,
                 salt_bridge_cutoff=_Const.SALT_BRIDGE_CUTOFF,
                 hbond_cutoff=_Const.HBOND_CUTOFF):
        sb = _ProtLigSaltBridges(msys_model, cms_model, prot_asl, lig_asl,
                                 contact_cutoff, salt_bridge_cutoff)
        wb = WaterBridges(msys_model, cms_model, prot_asl, lig_asl,
                          contact_cutoff, hbond_cutoff)
        hb = ProtLigHbondInter(msys_model, cms_model, prot_asl, lig_asl,
                               contact_cutoff, hbond_cutoff)
        self._analyzers = [sb, wb, hb]

    def _postcalc(self, *args):
        super()._postcalc(*args)
        self._result = {}
        for a in self._analyzers:
            self._result.update(a())
        _cleanup_polar_inter(self._result, self._analyzers[0].label_res,
                             self._analyzers[0]._ligand.h2heavy)


class MetalInter(staf.CompositeAnalyzer):
    """
    Interactions between metal elements and protein/ligand atoms.

    The result has the structure of
    [{
    'MetalResult': [`_MetalInter.MetalP` or `_MetalInter.MetalL` for each interaction]
    } for each frame]
    """
    # Here ion_aid, prot_aid and lig_aid are the AID of the ion atom, protein
    # atom and ligand atom, ion_ele is the ion element string, distance is the
    # distance between the ion atom and the protein/ligand atom.
    MetalP = namedtuple("_MetalP", "ion_aid ion_ele prot_aid distance")
    MetalL = namedtuple("_MetalL", "ion_aid ion_ele lig_aid distance")

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 metal_asl=None,
                 contact_cutoff=_Const.CONTACT_CUTOFF,
                 metal_cutoff=_Const.METAL_CUTOFF):
        if not metal_asl:
            metal_asl = _Const.METAL_ASL
        self._ligand = _get_lig_properties(cms_model, lig_asl)
        f = lambda aid: (cms_model.atom[aid].element not in ['C', 'H'] and
                         cms_model.atom[aid].atom_type_name not in ['ST', 'P5'])
        lig_not_CH = list(filter(f, self._ligand.aids))
        ion_near_lig = _KeyPartial(_ion_near_lig, metal_asl, lig_not_CH,
                                   lig_asl, metal_cutoff)
        res_near_lig = _KeyPartial(_res_near_lig, lig_asl, prot_asl,
                                   contact_cutoff)
        prot_near_ion = _KeyPartial(_prot_near_ion, metal_cutoff, ion_near_lig,
                                    res_near_lig)
        sel0 = staf.CustomMaestroAnalysis(msys_model, cms_model, ion_near_lig)
        sel1 = staf.CustomMaestroAnalysis(msys_model, cms_model, res_near_lig)
        sel2 = staf.CustomMaestroAnalysis(msys_model, cms_model, prot_near_ion)
        # ion_near_lig selection should proceed prot_near_ion selection
        self._analyzers = [sel0, sel1, sel2]
        self._lig_not_CH = lig_not_CH
        self._metal_cutoff = metal_cutoff

    def _isWithinCutoff(self, zipped_data):
        return zipped_data[-1] < self._metal_cutoff

    def _postcalc(self, calc, pbc, *_):
        super()._postcalc(calc, pbc, *_)
        ion, fsys_ct = self._analyzers[0]()
        prot_near_ion, _ = self._analyzers[2]()
        if not prot_near_ion:
            self._result = {'MetalResult': []}
            return

        pos = lambda aid: numpy.asarray(fsys_ct.atom[aid].xyz)
        ion_pos = [(aid, pos(aid)) for aid in ion]
        lig_pos = list(map(pos, self._lig_not_CH))
        prot_pos = list(map(pos, prot_near_ion))
        metal_inter = []
        for i_aid, i_pos in ion_pos:
            i_ele = fsys_ct.atom[i_aid].element
            l_dis = numpy.linalg.norm(pbc.calcMinimumDiff(i_pos, lig_pos),
                                      axis=1)
            p_dis = numpy.linalg.norm(pbc.calcMinimumDiff(i_pos, prot_pos),
                                      axis=1)
            ion_prot = list(
                filter(
                    self._isWithinCutoff,
                    zip([i_aid] * len(prot_near_ion),
                        [i_ele] * len(prot_near_ion), prot_near_ion, p_dis)))
            ion_lig = list(
                filter(
                    self._isWithinCutoff,
                    zip([i_aid] * len(self._lig_not_CH),
                        [i_ele] * len(self._lig_not_CH), self._lig_not_CH,
                        l_dis)))
            metal_inter.extend(self.MetalP(*x) for x in ion_prot)
            metal_inter.extend(self.MetalL(*x) for x in ion_lig)
        self._result = {'MetalResult': metal_inter}


class _PiInteractionFinder(staf.MaestroAnalysis):
    """
    Base class for pi-pi and cation-pi interaction finders.
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 asl1=None,
                 asl2=None,
                 aids1=None,
                 aids2=None):
        """
        If two atom selections are provided, find Pi interactions between the
        rings in the two selections. If the second atom selection is omitted,
        find Pi interactions within the first selection.
        """
        assert bool(asl1) ^ bool(aids1)
        assert not (bool(asl2) and bool(aids2))

        if not aids1:
            assert not topo.is_dynamic_asl(cms_model, asl1)
            aids1 = cms_model.select_atom(asl1)
        self._st1 = _extract_with_original_id(cms_model.fsys_ct, aids1)
        self._aids1 = set(aids1)
        if cms_model.need_msys:
            self._gids1 = topo.aids2gids(cms_model,
                                         aids1,
                                         include_pseudoatoms=False)
        else:
            self._gids1 = cms_model.convert_to_gids(aids1, with_pseudo=False)

        self._st2 = self._gids2 = None
        if asl2:
            assert not topo.is_dynamic_asl(cms_model, asl2)
            aids2 = cms_model.select_atom(asl2)
        if aids2:
            self._st2 = _extract_with_original_id(cms_model.fsys_ct, aids2)
            if cms_model.need_msys:
                self._gids2 = topo.aids2gids(cms_model,
                                             aids2,
                                             include_pseudoatoms=False)
            else:
                self._gids2 = cms_model.convert_to_gids(aids2,
                                                        with_pseudo=False)
        super().__init__(msys_model, cms_model)

    def _postcalc(self, calc, *_):
        fr = self._getCenteredFrame(calc)
        self._st1.setXYZ(fr.pos(self._gids1))
        if self._st2:
            self._st2.setXYZ(fr.pos(self._gids2))
        self._result = []


# Here ring1 and ring2 are ring atoms' AIDs, distance is between the centroids
# of the rings, angle is the plane angle between the two rings wrapped in
# [0, 90) range, and the interaction_type is either 'f2f' (face-to-face) or
# 'e2f' (edge-to-face).
PiPiInteraction = namedtuple('_PiPiInteraction',
                             'ring1 ring2 distance angle interaction_type')
# Here ring is the ring atoms' AIDs, cations are the cations' AIDs, distance
# is between the cation and the ring centroid, angle is between ring's normal
# vector and the vector from the ring's centroid to the cation centroid, and
# the direction has 3 possible values:
#    '11': both ring and cations in selection 1 (selection 2 is absent)
#    '12': cations in selection 1, ring in selection 2
#    '21': cations in selection 2, ring in selection 1
CatPiInteraction = namedtuple('_PiCatInteraction',
                              'ring cations distance angle direction')


class PiPiFinder(_PiInteractionFinder):
    """
    Find Pi-Pi interactions present between two sets of atoms, or within one
    set of atoms.

    The result has the structure of
    [[`PiPiInteraction` for each interaction] for each frame]
    """

    def _postcalc(self, calc, *_):
        super()._postcalc(calc, *_)
        for p in find_pi_pi_interactions(self._st1, struct2=self._st2):
            ring1 = tuple(p.struct1.atom[i].property[constants.ORIGINAL_INDEX]
                          for i in p.ring1.atoms)
            ring2 = tuple(p.struct2.atom[i].property[constants.ORIGINAL_INDEX]
                          for i in p.ring2.atoms)
            p_type = 'f2f' if p.face_to_face else 'e2f'
            self._result.append(
                PiPiInteraction(ring1, ring2, p.distance, p.angle, p_type))


class CatPiFinder(_PiInteractionFinder):
    """
    Find Cation-Pi interactions present between two sets of atoms, or within one
    set of atoms. With two sets of atom selections, it computes both
    cations in selection 1 with respect to rings in selection 2,
    cations in selection 2 with respect to rings in selection 1,
    but not within the same selection.

    The result has the structure of
    [[`CatPiInteraction` for each interaction] for each frame]
    """

    def _postcalc(self, calc, *_):
        super()._postcalc(calc, *_)
        for p in find_pi_cation_interactions(self._st1, struct2=self._st2):
            cations = tuple(
                p.cation_structure.atom[i].property[constants.ORIGINAL_INDEX]
                for i in p.cation_centroid.atoms)
            ring = tuple(
                p.pi_structure.atom[i].property[constants.ORIGINAL_INDEX]
                for i in p.pi_centroid.atoms)
            cat_loc = '1' if cations[0] in self._aids1 else '2'
            ring_loc = '1' if ring[0] in self._aids1 else '2'
            self._result.append(
                CatPiInteraction(ring, cations, p.distance, p.angle,
                                 cat_loc + ring_loc))


@lru_cache()
class ProtLigPiInter(staf.CompositeAnalyzer):
    """
    Compute pi-pi and pi-cation interactions between protein and ligand.

    The result has the structure of
    [{
    'PiPiResult': [`ProtLigPiInter.Pipi` for each interaction],
    'PiCatResult': [`ProgLigPiInter.PiLCatP`, or `ProgLigPiInter.PiPCatL` for each interaction]
    } for each frame]
    """
    # Here frag_idx is the index of the ligand fragment, ca_aid is the AID of
    # alpha carbon, prot_aid and lig_aid are the AID of the protein atom and
    # ligand atom, ring_idx is the ligand ring index, type could be 'f2f' or
    # 'e2f', distance and angle describe the geometry of the corresponding interaction.
    Pipi = namedtuple('_Pipi', 'ca_aid frag_idx ring_idx distance angle type')
    PiLCatP = namedtuple('_PiLCatP',
                         'prot_aid frag_idx ring_idx distance angle')
    PiPCatL = namedtuple('_PiPCatL', 'ca_aid lig_aid distance angle')

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 contact_cutoff=_Const.CONTACT_CUTOFF):
        res_near_lig = _KeyPartial(_res_near_lig, lig_asl, prot_asl,
                                   contact_cutoff)
        selection = staf.CustomMaestroAnalysis(msys_model, cms_model,
                                               res_near_lig)
        self._analyzers = [selection]
        self._ligand = _get_lig_properties(cms_model, lig_asl)

    def _postcalc(self, calc, pbc, *_):
        super()._postcalc(calc, pbc, *_)
        prot, fsys_ct = self._analyzers[0]()
        lig, aid2frag = self._ligand.aids, self._ligand.aid2frag
        if not (lig and prot):
            self._result = {'PiPiResult': [], 'PiCatResult': []}
            return
        lig_rings = self._ligand.rings  # utilize side effect to update
        prot_ct = _extract_with_original_id(fsys_ct, prot)
        lig_ct = _extract_with_original_id(fsys_ct, lig)

        # pi-pi result, unique to up (protein C-alpha and ligand fragment index)
        pipi_result, visited = [], set()

        # pi-pi result
        # unique to up (protein C-alpha and ligand fragment index)
        for p in find_pi_pi_interactions(prot_ct, struct2=lig_ct):
            prot_atom = p.struct1.atom[p.ring1.atoms[0]]
            prot_ca_aid = prot_atom.getResidue().getAlphaCarbon().property[
                constants.ORIGINAL_INDEX]
            if prot_ca_aid is None:
                # C-alpha not found in this protein residue
                continue
            lig_aid = p.struct2.atom[p.ring2.atoms[0]].property[
                constants.ORIGINAL_INDEX]
            lig_frag_idx = aid2frag[lig_aid]
            if (lig_frag_idx, prot_ca_aid) not in visited:
                visited.add((lig_frag_idx, prot_ca_aid))
                ring = [
                    p.struct2.atom[idx].property[constants.ORIGINAL_INDEX]
                    for idx in p.ring2.atoms
                ]
                try:
                    ring_idx = lig_rings.index(ring)
                except ValueError:
                    ring_idx = 0
                    lig_rings.append(ring)
                p_type = 'f2f' if p.face_to_face else 'e2f'
                pipi_result.append(
                    self.Pipi(prot_ca_aid, lig_frag_idx, ring_idx, p.distance,
                              p.angle, p_type))
        # pi-cation:
        # record cation on protein and ligand ring
        #     or cation on ligand and protein C-alpha
        pication_result, visited = [], set()
        for p in find_pi_cation_interactions(prot_ct, struct2=lig_ct):
            i = p.cation_centroid.atoms[0]
            cation_centr_aid = p.cation_structure.atom[i].property[
                constants.ORIGINAL_INDEX]
            pi_atom = p.pi_structure.atom[p.pi_centroid.atoms[0]]
            pi_centr_aid = pi_atom.property[constants.ORIGINAL_INDEX]
            if pi_centr_aid in aid2frag:  # ligand has the aromatic ring
                ring = [
                    p.pi_structure.atom[idx].property[constants.ORIGINAL_INDEX]
                    for idx in p.pi_centroid.atoms
                ]
                try:
                    ring_idx = lig_rings.index(ring)
                except ValueError:
                    ring_idx = 0
                    lig_rings.append(ring)
                pication_result.append(
                    self.PiLCatP(cation_centr_aid, aid2frag[pi_centr_aid],
                                 ring_idx, p.distance, p.angle))
            else:  # ligand atom is a cation
                prot_ca_aid = pi_atom.getResidue().getAlphaCarbon().property[
                    constants.ORIGINAL_INDEX]
                if prot_ca_aid is None:
                    # C-alpha not found in this protein residue
                    continue
                if (cation_centr_aid, prot_ca_aid) not in visited:
                    visited.add((cation_centr_aid, prot_ca_aid))
                    pication_result.append(
                        self.PiPCatL(prot_ca_aid, cation_centr_aid, p.distance,
                                     p.angle))
        self._result = {
            'PiPiResult': pipi_result,
            'PiCatResult': pication_result
        }


@lru_cache()
class ProtLigHalogenBondInter(staf.CompositeAnalyzer):
    """
    Find Halogen Bonds for protein ligand interactions.

    The result has the structure of
    [{
    'HalogenBondResult': [`ProtLigHalogenBondInter.Result` for each bond]
    } for each frame]
    """
    Result = namedtuple("_HalogenBond", "prot_aid lig_aid")

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 max_dist=_Const.HALOGEN_BOND_CUTOFF,
                 min_donor_angle=_Const.HALOGEN_BOND_MIN_D_ANGLE,
                 min_acceptor_angle=_Const.HALOGEN_BOND_MIN_A_ANGLE,
                 max_acceptor_angle=_Const.HALOGEN_BOND_MAX_A_ANGLE):
        self.prot_aids = cms_model.select_atom(prot_asl)
        self.lig_aids = cms_model.select_atom(lig_asl)
        self._analyzers = [
            HalogenBondFinder(msys_model,
                              cms_model,
                              self.prot_aids,
                              self.lig_aids,
                              max_dist=max_dist,
                              min_donor_angle=min_donor_angle,
                              min_acceptor_angle=min_acceptor_angle,
                              max_acceptor_angle=max_acceptor_angle)
        ]

    def _postcalc(self, *args):
        super()._postcalc(*args)
        halobond = []
        for prot_aid, lig_aid in self._analyzers[0]():
            if lig_aid not in self.lig_aids:
                prot_aid, lig_aid = lig_aid, prot_aid
            halobond.append(self.Result(prot_aid, lig_aid))
        self._result = {'HalogenBondResult': halobond}


@lru_cache()
class ProtLigHbondInter(staf.CompositeAnalyzer):
    """
    Compute protein-ligand hydrogen bonds.

    The result has the structure of
    [{
    'HBondResult': [`ProtLigHbondInter.Result` for each H bond],
    } for each frame]
    """
    bb_donor_type = [
        'H', 'HA', 'HA2', 'HA3', '1H', '2H', '3H', '1HA', '2HA', '3HA'
    ]
    bb_acceptor_type = ['O']
    # Here prot_aid is the AID of the protein atom, prot_type is a string that
    # denotes the acceptor/donor, backbone/side chain information, lig_aid is
    # the AID of ligand atom.
    Result = namedtuple("_Hbond", "prot_aid prot_type lig_aid")

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 contact_cutoff=_Const.CONTACT_CUTOFF,
                 hbond_cutoff=_Const.HBOND_CUTOFF):
        res_near_lig = _KeyPartial(_res_near_lig, lig_asl, prot_asl,
                                   contact_cutoff)
        selection = staf.CustomMaestroAnalysis(msys_model, cms_model,
                                               res_near_lig)
        self._ligand = _get_lig_properties(cms_model, lig_asl)
        self._analyzers = [selection]
        self._get_hbonds = partial(get_hydrogen_bonds,
                                   max_dist=hbond_cutoff,
                                   honor_pbc=True)

    def _postcalc(self, calc, pbc, fr):
        super()._postcalc(calc, pbc, fr)
        prot, fsys_ct = self._analyzers[0]()
        lig = self._ligand.aids
        if not (lig and prot):
            self._result = {'HBondResult': []}
            return
        hbonds = []
        for d, a in self._get_hbonds(fsys_ct, atoms1=lig, atoms2=prot):
            lig_aid, prot_aid, prot_type = int(d), int(a), 'a'
            if lig_aid in prot:
                lig_aid, prot_aid, prot_type = prot_aid, lig_aid, 'd'
            # label backbone hbond or sidechain hbond
            prot_ptype = fsys_ct.atom[prot_aid].pdbname.strip()
            if (prot_type == 'a' and prot_ptype in self.bb_acceptor_type) or (
                    prot_type == 'd' and prot_ptype in self.bb_donor_type):
                prot_type += '-b'
            else:
                prot_type += '-s'
            hbonds.append(self.Result(prot_aid, prot_type, lig_aid))
        self._result = {'HBondResult': hbonds}


class _WatLigFragDistance(staf.CompositeAnalyzer):
    """
    For all water molecules within the cutoff radius (`hbond_cutoff` + 0.3) to
    the ligand molecule, find the distance between the water oxygen atom and
    the closest ligand fragment's centroid.

    The result has the structure of
    [{
    'LigWatResult': [`_WatLigFragDistance.Result` for each water-ligand-fragment-pair]
    } for each frame]
    """
    # Here `frag_idx` is the index of the ligand fragment, `wat_res_num` is the
    # water residue number, and `distance` is the distance between the water
    # oxygen atom and the fragment's centroid.
    Result = namedtuple('_WatLigFragDis', 'frag_idx wat_res_num distance')

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 hbond_cutoff=_Const.HBOND_CUTOFF):
        wat_near_lig = _KeyPartial(_wat_near_lig, lig_asl, hbond_cutoff + 0.3)
        selection = staf.CustomMaestroAnalysis(msys_model, cms_model,
                                               wat_near_lig)
        self._analyzers = [selection]
        self._ligand = _get_lig_properties(cms_model, lig_asl)
        self._cms_model = cms_model

        # Registers the centroids of the ligand fragments.
        for frag in self._ligand.frags:
            if cms_model.need_msys:
                gids = topo.aids2gids(cms_model, frag, False)
            else:
                gids = cms_model.convert_to_gids(frag, with_pseudo=False)
            self._analyzers.append(Centroid(msys_model, cms_model, gids=gids))

    def _postcalc(self, calc, pbc, fr):
        result = []
        if self._ligand.aids:
            super()._postcalc(calc, pbc, fr)
            near_wat, fsys = self._analyzers[0]()
            if near_wat:
                aids_O = [
                    aid for aid in near_wat if fsys.atom[aid].element == 'O'
                ]
                if self._cms_model.need_msys:
                    gids_O = topo.aids2gids(self._cms_model, aids_O, False)
                else:
                    gids_O = self._cms_model.convert_to_gids(aids_O,
                                                             with_pseudo=False)

                frag_centroids = [a() for a in self._analyzers[1:]]

                frag_O_r2 = []
                for gid in gids_O:
                    # Avoids expensive `sqrt` (`numpy.norm`).
                    diff = pbc.calcMinimumDiff(fr.pos(gid), frag_centroids)
                    frag_O_r2.append((diff**2).sum(axis=1))

                min_dist = numpy.sqrt(numpy.min(frag_O_r2, axis=1))
                closest_frag = numpy.argmin(frag_O_r2, axis=1)
                wat_res_num = [fsys.atom[aid].resnum for aid in aids_O]
                result = list(zip(closest_frag, wat_res_num, min_dist))
        self._result = {'LigWatResult': [self.Result(*x) for x in result]}


class WatLigFragDistance(staf.CompositeAnalyzer):
    """
    Distance between water oxygen atom and its closest ligand fragment, with
    water bridges excluded.

    The result has the structure of
    [{
    'LigWatResult': [`_WatLigFragDistance.Result` for each water-ligand-fragment-pair],
    'WaterBridgeResult': [`WaterBridges.Result` for each bridge]
    } for each frame]
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 contact_cutoff=_Const.CONTACT_CUTOFF,
                 hbond_cutoff=_Const.HBOND_CUTOFF):
        wlfd = _WatLigFragDistance(msys_model, cms_model, prot_asl, lig_asl,
                                   hbond_cutoff)
        wb = WaterBridges(msys_model, cms_model, prot_asl, lig_asl,
                          contact_cutoff, hbond_cutoff)
        self._analyzers = [wlfd, wb]

    def _postcalc(self, *args):
        super()._postcalc(*args)
        self._result = {}
        for a in self._analyzers:
            self._result.update(a())
        _cleanup_water_ligand_distance(self._result)


@lru_cache()
class WaterBridges(staf.CompositeAnalyzer):
    """
    Find water bridges between protein and ligand.

    The result has the structure of
    [{
    'WaterBridgeResult': [`WaterBridges.Result` for each bridge]
    } for each frame]
    """
    # Here prot_aid and lig_aid are the AID of the protein atom and ligand atom,
    # prot_type and lig_type are strings that denotes the acceptor/donor information,
    # wat_res_num is the water residue number.
    Result = namedtuple("_WaterBridge",
                        "prot_aid prot_type lig_aid lig_type wat_res_num")

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 contact_cutoff=_Const.CONTACT_CUTOFF,
                 hbond_cutoff=_Const.HBOND_CUTOFF):
        res_near_lig = _KeyPartial(_res_near_lig, lig_asl, prot_asl,
                                   contact_cutoff)
        wat_near_lig = _KeyPartial(_wat_near_lig, lig_asl, hbond_cutoff + 0.3)
        sel0 = staf.CustomMaestroAnalysis(msys_model, cms_model, res_near_lig)
        sel1 = staf.CustomMaestroAnalysis(msys_model, cms_model, wat_near_lig)
        self._analyzers = [sel0, sel1]
        self._ligand = _get_lig_properties(cms_model, lig_asl)
        self._get_hbonds = partial(get_hydrogen_bonds,
                                   max_dist=hbond_cutoff,
                                   min_donor_angle=_Const.HBOND_MIN_D_ANGLE -
                                   10.0,
                                   min_acceptor_angle=_Const.HBOND_MIN_A_ANGLE,
                                   max_acceptor_angle=_Const.HBOND_MAX_A_ANGLE,
                                   honor_pbc=True)

    def _postcalc(self, calc, pbc, _):
        super()._postcalc(calc, pbc, _)
        prot, fsys_ct = self._analyzers[0]()
        wat, _ = self._analyzers[1]()
        lig = self._ligand.aids
        if not (lig and prot and wat):
            self._result = {'WaterBridgeResult': []}
            return
        # find protein-water hbonds
        prot_hbonds_w_wat = defaultdict(list)
        for d, a in self._get_hbonds(fsys_ct, atoms1=prot, atoms2=wat):
            wat_aid, prot_aid, prot_type = int(d), int(a), 'a'
            if wat_aid in prot:
                wat_aid, prot_aid, prot_type = prot_aid, wat_aid, 'd'
            wat_res_num = fsys_ct.atom[wat_aid].resnum
            prot_hbonds_w_wat[wat_res_num].append((prot_aid, prot_type))

        # find ligand-water hbonds
        lig_hbonds_w_wat = defaultdict(list)
        for d, a in self._get_hbonds(fsys_ct, atoms1=lig, atoms2=wat):
            wat_aid, lig_aid, lig_type = int(d), int(a), 'a'
            if wat_aid in lig:
                wat_aid, lig_aid, lig_type = lig_aid, wat_aid, 'd'
            wat_res_num = fsys_ct.atom[wat_aid].resnum
            lig_hbonds_w_wat[wat_res_num].append((lig_aid, lig_type))

        # find unique water bridges
        # up to (protein residue, water residue, ligand aid)
        water_bridges, visited = [], set()
        for wat_res_num, lig_hbonds in lig_hbonds_w_wat.items():
            if wat_res_num in prot_hbonds_w_wat:
                for lig, prot in product(lig_hbonds,
                                         prot_hbonds_w_wat[wat_res_num]):
                    prot_res_num = fsys_ct.atom[prot[0]].resnum
                    if (prot_res_num, wat_res_num, lig[0]) in visited:
                        continue
                    wb = self.Result(prot[0], prot[1], lig[0], lig[1],
                                     wat_res_num)
                    water_bridges.append(wb)
                    visited.add((prot_res_num, wat_res_num, lig[0]))
        self._result = {'WaterBridgeResult': water_bridges}


class ProtLigInter(staf.CompositeAnalyzer):
    """
    Composition of various protein ligand interactions.

    The result has the structure of
    [{
    'WaterBridgeResult': [`WaterBridges.Result` for each bridge],
    'LigWatResult': [`_WatLigFragDistance.Result` for each water-ligand-fragment-pair],
    'HBondResult': [`ProtLigHbondInter.Result` for each interaction],
    'PiPiResult': [`ProtLigPiInter.Pipi` for each interaction],
    'PiCatResult': [`ProtLigPiInter.PiLCatP` or `ProtLigPiInter.PiPCatL` for each interaction],
    'MetalResult': [`_MetalInter.MetalP` or `_MetalInter.MetalL` for each interaction],
    'PolarResult': [`_ProtLigSaltBridges.Result` for each bridge],
    }]
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 prot_asl,
                 lig_asl,
                 metal_asl=None):
        """
        :type msys_model: `msys.System`
        :type  cms_model: `cms.Cms`
        :type   prot_asl: `str`
        :param  prot_asl: ASL expression to specify protein atoms
        :type    lig_asl: `str`
        :param   lig_asl: ASL expression to specify ligand atoms
        :type  metal_asl: `str` or `None`
        :param metal_asl: ASL expression to specify metal atoms. If `None`, use
                          default values.
        """
        args = (msys_model, cms_model, prot_asl, lig_asl)
        self._analyzers = [
            _HydrophobicInter(*args),
            _ProtLigSaltBridges(*args),
            WaterBridges(*args),
            _WatLigFragDistance(*args),
            ProtLigHbondInter(*args),
            MetalInter(*(args + (metal_asl,))),
            ProtLigPiInter(*args),
            ProtLigHalogenBondInter(*args),
        ]

    def _postcalc(self, *args):
        super()._postcalc(*args)
        self._result = {}
        for a in self._analyzers:
            self._result.update(a())
        label_res = self._analyzers[0].label_res
        h2heavy = self._analyzers[0]._ligand.h2heavy
        aid2frag = self._analyzers[0].aid2frag
        _cleanup_water_ligand_distance(self._result)
        _cleanup_polar_inter(self._result, label_res, h2heavy)
        _cleanup_hydrophobic_inter(self._result, label_res, aid2frag)


class AmorphousCrystalInter(staf.CompositeAnalyzer):
    """
    Composition of various interactions between a specified molecule and its
    crystal-mates.

    The result has the structure of::

        {'HBondResult': [number of `HydrogenBondFinder.Result` per frame],
         'HalBondResult': [number of `HalogenBondFinder.Result` per frame],
         'PiPiResult': [number of `PiPiFinder.Result` per frame],
         'CatPiResult': [number of `CatPiFinder.Result` per frame],
         'PolarResult': [number of `SaltBridgeFinder.Result` per frame]
         'HydrophobResult': [number of `HydrophobicInterFinder.Result` per frame]}

    """

    RESULT_NAMES = [
        'HBondResult', 'HalBondResult', 'PiPiResult', 'CatPiResult',
        'PolarResult', 'HydrophobResult'
    ]

    def __init__(self, msys_model, cms_model, asl):
        """
        Selection of molecule of interest is passed through `asl` variable.
        Environment selection is generated by excluding the `asl` and waters.

        :type  msys_model: `msys.System`
        :type   cms_model: `cms.Cms`
        :type         asl: `str`
        :param        asl: ASL expression to specify selection atoms
        """
        aids1 = cms_model.select_atom(asl)
        aids2 = cms_model.select_atom(f'not ({asl}) and not water')
        args = (msys_model, cms_model)

        self._analyzers = [
            HydrogenBondFinder(*args, aids1, aids2),
            HalogenBondFinder(*args, aids1, aids2),
            PiPiFinder(*args, aids1=aids1, aids2=aids2),
            CatPiFinder(*args, aids1=aids1, aids2=aids2),
            SaltBridgeFinder(*args, aids1, aids2),
            HydrophobicInterFinder(*args, aids1, aids2)
        ]

    def _postcalc(self, *args):
        super()._postcalc(*args)
        self._result = []
        for a in self._analyzers:
            self._result.append(len(a()))

    def reduce(self, results):
        ret = defaultdict(list)
        for fr_result in results:
            for n, v in zip(self.RESULT_NAMES, fr_result):
                ret[n].append(v)
        return ret


class VolumeMapper(staf.GeomAnalyzerBase):
    """
    This class calculates the 3D histogram of selected atoms over a trajectory.

    Note: The trajectory input for this method should already be centered and
          aligned on the atoms of interest. By default, the returned histogram
          has origin in its central bin.

    Basic usage:
        ana = VolumeMapper(cms_model, 'mol.num 1')
        results = analyze(tr, ana)
    """

    def __init__(self,
                 cms_model,
                 asl=None,
                 aids=None,
                 spacing=(1., 1., 1.),
                 length=(10., 10., 10.),
                 center=(0., 0., 0.),
                 normalize=True):
        """
        :type  cms_model: `cms.Cms`
        :type   asl: `str`
        :param  asl: The ASL selection for which volumetric density map will be
                     constructed
        :type  aids: `list` of `int`
        """
        assert (asl is None) ^ (aids is None)

        self._length = numpy.asarray(length)
        self._spacing = numpy.asarray(spacing)
        self._center = numpy.asarray(center)
        if asl:
            aids = cms_model.select_atom(asl)
        if cms_model.need_msys:
            self._gids = topo.aids2gids(cms_model,
                                        aids,
                                        include_pseudoatoms=False)
        else:
            self._gids = cms_model.convert_to_gids(aids, with_pseudo=False)

        assert (len(self._gids) > 2), "The number of selected atoms must be >2"
        assert (self._spacing > 0.0).all()
        assert (self._length > 0.0).all()
        assert (self._spacing <= self._length).all()

        # Determine the number of grid points
        self._bins = numpy.ceil(self._length / self._spacing).astype(int)
        self._density = normalize

    def _postcalc(self, _, pbc, fr):
        self._result = fr.pos(self._gids)

    def reduce(self, pos_t, *_, **__):
        pos_t_array = numpy.asarray(pos_t)
        pos_t_array.shape = -1, 3
        grid_range = [
            [-x / 2 + o, x / 2 + o] for x, o in zip(self._length, self._center)
        ]
        h, _ = numpy.histogramdd(pos_t_array,
                                 bins=self._bins,
                                 range=grid_range,
                                 density=self._density)
        return h


def progress_report_frame_number(i, *_):
    print("analyzing frame# %d..." % i)


def analyze(tr, analyzer, *arg, **kwarg):
    """
    Do analyses on the given trajectory `tr`, and return the results.
    The analyses are specified as one or more positional arguements. Each
    analyzer should satisfy the interface requirements (see the docstring of
    `GeomCalc.addAnalyzer`).

    :type   tr: `list` of `traj.Frame`
    :param  tr: The simulation trajectory to analyze
    :param arg: A list of analyzer objects

    :type kwarg["progress_feedback"]: callable, e.g., func(i, fr, tr), where
            `i` is the current frame index, `fr` the current frame, `tr` the
            whole trajectory.
    :param kwarg["progress_feedback"]: This function will be called at start of
            analysis on the current frame. This function is intended to report the
            progress of the analysis.

    :rtype: `list`
    :return: For a single analyzer, this function will return a list of analysis
             results, and each element in the list corresponds to the result of
             the corresponding frame. For multiple analyzers, this function will
             return a list of lists, and each element is a list of results of
             the corresponding analyzer. If an analyzer has a `reduce` method,
             the reduce method will be called, and its result will be returned.
    """
    calc = staf.GeomCalc()
    analyzers = [analyzer] + list(arg)
    results = []
    for a in analyzers:
        calc.addAnalyzer(a)
        results.append([])
    report_progress = kwarg.get("progress_feedback", None)
    for i, fr in enumerate(tr):
        report_progress and report_progress(i, fr, tr)
        pbc = Pbc(fr.box)
        calc(pbc, fr)
        for a, r in zip(analyzers, results):
            r.append(a())

    for a, (i, r) in zip(analyzers, enumerate(results)):
        if callable(getattr(a, "reduce", None)):
            results[i] = a.reduce(r)

    # FIXME: do we really want this special treatment?
    return results[0] if 1 == len(analyzers) else results


def rmsd_matrix(obj, tr, rmsd_gids, fit_gids=None):
    """
    Return an NxN matrix where N is the number of frames in the trajectory `tr`
    and the (i, j)'th entry is the RMSD between frame i and frame j. The frame-wise
    RMSD values are calculated for atoms specified by `rmsd_gids`. If `fit_gids`
    is provided, the corresponding atoms are used to superimpose the two frames
    first.

    :type        obj: `msys.System` or `cms.Cms.glued_topology`
    :param       obj: connection for trjactory centering
    :type         tr: `list` of `traj.Frame` objects
    :param        tr: Trajectory
    :type  rmsd_gids: `list` of `int`
    :param rmsd_gids: GIDs of atoms for which to calculate the RMSD
    :type   fit_gids: `None` or a `list` of `int`
    :param  fit_gids: GIDs of atoms on which to we align the structures. If
                      `None`, no alignment is performed.

    :rtype: `numpy.ndarray` of `float`
    :return: A symmetric square matrix of RMSDs
    """
    if not rmsd_gids:
        raise ValueError("Empty GID list for RMSD calculation")

    n = len(tr)
    m = numpy.zeros((n, n))
    if fit_gids:
        tr_copy = [fr.copy() for fr in tr]
        if isinstance(obj, msys.System):
            topo.center(obj, fit_gids, tr_copy)
        else:
            pfx.center(obj, fit_gids, tr_copy)
        rmsd_pos_array = [fr.pos(rmsd_gids) for fr in tr_copy]
        fit_pos_array = rmsd_pos_array if rmsd_gids == fit_gids \
                                 else [fr.pos(fit_gids) for fr in tr_copy]
    else:
        rmsd_pos_array = [fr.pos(rmsd_gids) for fr in tr]

    for i in range(n):
        for j in range(i + 1, n):
            if fit_gids:
                pos_j = align_pos(rmsd_pos_array[j],
                                  fit_pos_array[j],
                                  fit_pos_array[i],
                                  is_precentered=True)
            else:
                pos_j = rmsd_pos_array[j]
            m[i, j] = m[j, i] = numpy.sqrt(
                ((pos_j - rmsd_pos_array[i])**2).sum(axis=1).mean())
    return m


def cluster(affinity_matrix) -> Tuple[List[int], List[int]]:
    """
    Use the affinity propagation method to cluster the input matrix, gradually
    increasing the damping factor until the algorithm converges.

    The maximum number of iterations is currently hard-coded to 400. If the
    damping factor reaches 1.0 (which is not a valid value), the function will
    return empty lists for the centers and labels.

    :type  affinity_matrix: `numpy.ndarray` of `float`
    :param affinity_matrix: A square matrix of affinity/similarity values

    :return: The first list is the sample indices of the clusters' centers, the
             second list is a cluster label of all samples.
    """
    from sklearn.cluster import AffinityPropagation
    from sklearn.exceptions import ConvergenceWarning
    max_iterations = 400
    converged = False
    damping = 0.5
    # Damping values are only valid between [.5, 1.0)
    while not converged and damping < 1.0:
        # Skip convergence warning. Can add `verbose=True` to the ap constructor
        # for convergence information.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            ap = AffinityPropagation(max_iter=max_iterations,
                                     affinity="precomputed",
                                     damping=damping,
                                     random_state=0)
            ap.fit(affinity_matrix)

        # Use iteration count to determine convergence.
        converged = ap.n_iter_ < max_iterations
        damping += 0.01

    if converged:
        return list(ap.cluster_centers_indices_), list(ap.labels_)

    # If the loop exited without converging, we should return an empty list to
    # avoid passing degenerate values.
    return [], []


class Rdf(staf.CompositeAnalyzer):
    """
    Calculate radial distribution function (RDF, also known as g(r)) for atom or
    atom group selections.

    In general, we need two groups of positions. The first group are the
    reference positions, whereas the second the distance group. The reference
    and distance groups can be the same.

    The 'pos_type' parameter determines the types of positions used as reference
    and distance groups. When 'pos_type' is set to "atom", each atom is
    considered individually and the 'group_type' parameters are ignored, meaning
    no grouping is performed.

    However, when 'pos_type' is set to values other than "atom", such as "com",
    "coc", or "centroid", the atoms can be grouped into larger units like
    molecules or residues based on the 'group_type' parameters. In these cases,
    the RDF is calculated using these larger group units instead of individual
    atoms.

    For example, if we want to calculate the RDF of the distances of water
    hydrogen atoms with respect to water oxygen atoms, and 'pos_type' is set to
    "atom", each atom is considered individually. Alternatively, if 'pos_type'
    is "com" and 'group_type' is "MOLECULE", atoms are grouped by their
    molecular identity and the center of mass of each molecule is used, thus
    resulting in RDF of water molecules with respect to water molecules.
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 asl0,
                 asl1=None,
                 pos_type0="atom",
                 pos_type1="atom",
                 dr=0.1,
                 rmax=12.0,
                 group_type0=GroupType.MOLECULE,
                 group_type1=GroupType.MOLECULE):
        """
        :type   msys_model: `msys.System`
        :type    cms_model: `cms.Cms`
        :type         asl0: `str`
        :param        asl0: Atom selection for the reference group
        :type         asl1: `str` or `None`
        :param        asl1: Atom selection for the distance group. If it's
                            `None`, it will default to `asl0`.
        :type    pos_type0: `str`
        :param   pos_type0: Type of positions of the reference group:
                            "atom"    : Use atom's position directly
                            "com"     : Use molecular center of mass
                            "coc"     : Use molecular center of charge
                            "centroid": Use molecular centroid
        :type    pos_type1: `str`
        :param   pos_type1: Type of positions of the distance group. Values are
                            the same as those of `pos_type0`
        :type           dr: `float`
        :param          dr: Bin width in the unit of Angstroms
        :type         rmax: `float`
        :param        rmax: Maximum distance in the unit in Angstroms. The RDF
                            will be calculated until `rmax`.
        :type  group_type0: `GroupType`
        :param group_type0: Method to use to group atoms in `asl0` selection.
                            Only used when `pos_type0` is not "atom".
        :type  group_type1: `GroupType`
        :param group_type1: Method to use to group atoms in `asl1` selection.
                            Only used when `pos_type1` is not "atom".
        """
        self._pos_types = (pos_type0, pos_type1)
        self._group_types = (group_type0, group_type1)
        # We need positers for derived positions such as COM, COC, and centroid.
        # We will know their GIDs only during the run time.
        self._analyzers = []
        self._all_gids_list = [None] * 2

        # Intemediate results
        self._all_num_gids = [None] * 2
        self._bins = numpy.linspace(0, rmax, int(rmax / dr) + 1)
        self._slice_volumes = numpy.array(Rdf._get_slice_volumes(self._bins))
        self._volumes = []
        self._histograms = []

        def add_analyzer(asl, pos_type, group_type):  # positer or not
            if pos_type == 'atom':
                self._analyzers.append(
                    staf.DynamicAslAnalyzer(msys_model, cms_model, asl))
            else:
                init = partial(Rdf._aids2gids,
                               pos_type=pos_type,
                               group_type=group_type)
                self._analyzers.append(
                    staf.DynamicPositerAnalyzer(msys_model, cms_model, asl,
                                                init))

        add_analyzer(asl0, pos_type0, group_type0)
        if asl1 is not None:
            add_analyzer(asl1, pos_type1, group_type1)
            # avoid redundant second analyzer
            both_static = not (self._analyzers[0].isDynamic() or
                               self._analyzers[1].isDynamic())
            if both_static and pos_type0 == pos_type1 and \
                self._analyzers[0]._aids == self._analyzers[1]._aids:
                del self._analyzers[1]

        for i, ana in enumerate(self._analyzers):  # dynamic or not
            if ana.isDynamic():
                super().__init__(is_dynamic=True)
            else:  # Track the static types
                ana.disableDyncalc()
                if isinstance(ana, staf.DynamicAslAnalyzer):
                    self._all_gids_list[i] = self._aids2gids(
                        msys_model,
                        cms_model,
                        ana._aids,
                        self._pos_types[i],
                        group_type=self._group_types[i])
                else:
                    self._all_gids_list[i] = ana._positer

    def _dyncalc(self, calc):
        super()._dyncalc(calc)
        for i, ana in enumerate(self._analyzers):
            if ana.isDynamic():
                if isinstance(ana, staf.DynamicAslAnalyzer):
                    self._all_gids_list[i] = self._aids2gids(
                        ana._msys_model,
                        ana._cms_model,
                        ana._aids,
                        self._pos_types[i],
                        group_type=self._group_types[i])
                else:
                    self._all_gids_list[i] = ana._positer

    @staticmethod
    def _getAtomIDsPerMolecule(cms_model: cms.Cms,
                               aids: List[int]) -> List[List[int]]:
        """
        Group the atoms indices by molecule they belong to

        :param cms_model: The structure which contains the atoms

        :param      aids: The list of atom ids to be grouped

        :return         : List of lists of the atom ids belonging to the molecule
        """
        mol_atoms = defaultdict(list)
        for aid in aids:
            mol_atoms[cms_model.atom[aid].molecule_number].append(aid)
        return list(mol_atoms.values())

    @staticmethod
    def _getAtomIDsPerResidue(cms_model: cms.Cms,
                              aids: List[int]) -> List[List[int]]:
        """
        Group the atoms indices by residue they belong to

        :param cms_model: The structure which contains the atoms

        :param      aids: The list of atom ids to be grouped

        :return         : List of lists of the atom ids belonging to the residue
        """
        res_atoms = defaultdict(list)
        for aid in aids:
            residue = cms_model.atom[aid].getResidue()
            # Manual creation of hash key is faster than looping over
            # structure.residue
            hash_key = (residue.molecule_number, residue.chain, residue.pdbres,
                        residue.resnum, residue.inscode)
            res_atoms[hash_key].append(aid)
        return list(res_atoms.values())

    @staticmethod
    def _groupAtoms(cms_model: cms.Cms, aids: List[int],
                    group_type: GroupType) -> List[List[int]]:
        """
        Group/cluster the atoms indices by the given group type.

        :param   cms_model: The structure which contains the atoms

        :param        aids: The list of atom ids to be grouped

        :param  group_type: The type of grouping to be performed.

        :return           : The list of lists where each list contains atom ids
                            belonging to the same group

        :raises ValueError: If the group type is not supported
        """
        group_methods = {
            GroupType.MOLECULE: Rdf._getAtomIDsPerMolecule,
            GroupType.RESIDUE: Rdf._getAtomIDsPerResidue,
        }

        group_method = group_methods.get(group_type, None)
        if group_method is None:
            valid_types = ', '.join([gt.value for gt in GroupType])
            raise ValueError(f'ERROR: Unsupported group type "{group_type}". '
                             f'Acceptable group types are: {valid_types}.')

        return group_method(cms_model, aids)

    @staticmethod
    def _aids2gids(msys_model,
                   cms_model,
                   aids,
                   pos_type,
                   group_type=GroupType.MOLECULE):
        if not aids:
            raise ValueError('ERROR: No atom selected.')

        if pos_type == "atom":
            if cms_model.need_msys:
                return sorted(topo.aids2gids(cms_model, aids, False))
            else:
                return sorted(cms_model.convert_to_gids(aids,
                                                        with_pseudo=False))
        else:
            Analyzer = {"com": Com, "coc": Coc, "centroid": Centroid}[pos_type]
            grouped_atom_ids = Rdf._groupAtoms(cms_model, aids, group_type)
            analyzers = []
            for aids in grouped_atom_ids:
                if cms_model.need_msys:
                    gids = sorted(topo.aids2gids(cms_model, aids, False))
                else:
                    gids = sorted(
                        cms_model.convert_to_gids(aids, with_pseudo=False))
                analyzers.append(Analyzer(msys_model, cms_model, gids=gids))

            return staf.Positer(analyzers, len(analyzers))

    @staticmethod
    def _get_slice_volumes(bins):
        """
        Calculate slice volumes between the adjacent bins
        """
        pi_4_3 = 4. / 3. * numpy.pi
        volumes = [pi_4_3 * r**3 for r in bins]
        return [1] + [
            large - small for small, large in zip(volumes, volumes[1:])
        ]

    def _postcalc(self, _, pbc, fr):
        # If we ever need the positions of some COMs/COCs/centroids, they
        # should be ready in `fr.pos()' now. To get them, all we need to do is
        # to figure out their GIDs.
        # This could be slow, so we should cache the GID lists.
        if self.isDynamic() or self._all_num_gids[0] is None:
            gids0 = self._all_gids_list[0]
            gids1 = self._all_gids_list[1]
            if not isinstance(gids0, list):
                # `gids0' should be an object that allows us to get the GIDs.
                gids0 = gids0.gids()
            if gids1 and not isinstance(gids1, list):
                # `gids1' should be an object that allows us to get the GIDs.
                gids1 = gids1.gids()

            # Ensures gids0 and gids1 are mutually exclusive if they are not
            # identical.
            if gids0 == gids1:
                gids1 = None
            elif gids1:
                if set(gids0) & set(gids1):
                    raise ValueError('ERROR: Two groups of positions should be '
                                     'mutually exclusive if they are not '
                                     'identical')

            # Ensures the CenterOf objects are mutually exclusive, if not identical
            positer_check = self._pos_types[0] == self._pos_types[1] != 'atom'
            if positer_check and len(self._analyzers) == 2:
                aids0 = self._analyzers[0]._aids
                aids1 = self._analyzers[1]._aids
                if aids0 == aids1:
                    gids1 = None
                elif set(aids0) & set(aids1):
                    raise ValueError('ERROR: Two groups of positions should be '
                                     'mutually exclusive if they are not '
                                     'identical')

            self._all_gids_list[0] = gids0
            self._all_gids_list[1] = gids1
            if gids1 is None:
                gids1 = gids0
            self._all_num_gids[0] = len(gids0)
            self._all_num_gids[1] = len(gids1)

        histogram = 0
        if self._all_gids_list[1]:
            # Two groups, say group 1 has N atoms and group 2 has M atoms, then
            # we need to generate all M*N differences.
            pos0 = fr.pos(self._all_gids_list[0])
            pos1 = fr.pos(self._all_gids_list[1])
            for xyz0 in pos0:
                diff_vec = pbc.calcMinimumDiff(xyz0, pos1)
                dist = numpy.sqrt((diff_vec**2).sum(axis=-1))
                histogram += numpy.histogram(dist, self._bins)[0]
        else:
            # Single group selection, say N atoms, then we need all N-choose-2
            # differences, i.e., the upper triangular portion of pairwise matrix.
            pos1 = fr.pos(self._all_gids_list[0])
            pos0 = pos1[:-1]
            for i, xyz0 in enumerate(pos0):
                diff_vec = pbc.calcMinimumDiff(xyz0, pos1[i + 1:])
                dist = numpy.sqrt((diff_vec**2).sum(axis=-1))
                histogram += numpy.histogram(dist, self._bins)[0]

        self._histograms.append(histogram)
        self._volumes.append(pbc.volume)

    def reduce(self, *_, **__):
        """
        Aggregates the frame-based results (histograms) and returns the final
        RDF results.

        :rtype: `(list, list)`
        :return: Returns the RDF (the first list), and the integral (the second
                 list).
        """
        is_autordf = int(self._all_gids_list[1] is None)
        num_center = self._all_num_gids[0]
        num_around = self._all_num_gids[1] - is_autordf
        average_density = num_around / numpy.mean(self._volumes)

        average_histogram = numpy.mean(self._histograms, axis=0)
        average_histogram /= num_center
        average_histogram = numpy.insert(average_histogram, 0, 0)

        if is_autordf:
            average_histogram *= 2
        rdf = average_histogram / average_density
        rdf = rdf / self._slice_volumes
        integral = numpy.cumsum(average_histogram)
        return rdf, integral

    def bins(self):
        return self._bins


# FIXME: The result only takes 1 atom to represent a ring or a cation group.
#        It may be better to use `PiPiInteraction` and `CatPiInteraction`.
class ProtProtPiInter(staf.MaestroAnalysis):
    """
    Protein-protein Pi interaction finder. The result has the structure of
    [{
    'pi-pi': [(an atom from ring1, an atom from ring2) for each interaction],
    'pi-cat': [(an atom from ring, an atom from cation) for each interaction]
    } for each frame]
    """

    def __init__(self, msys_model, cms_model, asl):
        """
        :type msys_model: `msys.System`
        :type  cms_model: `cms.Cms`
        :type        asl: `str`
        :param       asl: ASL expression to select protein atoms
        """
        self._aids = cms_model.select_atom(asl)

        super().__init__(msys_model, cms_model)

    def _postcalc(self, calc, _, fr):
        fsys_ct = self._getCenteredCt(calc)
        self._result = {'pi-pi': [], 'pi-cat': []}

        for p in find_pi_pi_interactions(fsys_ct,
                                         atoms1=self._aids,
                                         atoms2=self._aids):
            pair = sorted([p.ring1.atoms[0], p.ring2.atoms[0]])
            self._result['pi-pi'].append(pair)

        for p in find_pi_cation_interactions(fsys_ct,
                                             atoms1=self._aids,
                                             atoms2=self._aids):
            self._result['pi-cat'].append(
                (p.pi_centroid.atoms[0], p.cation_centroid.atoms[0]))


class ProtProtHbondInter(staf.CompositeAnalyzer):
    """
    Protein-protein hydrogen bond finder.

    The result has the structure of
    [{
    'hbond_bb': [(donor AID, acceptor AID) for each interaction],
    'hbond_ss': [(donor AID, acceptor AID) for each interaction],
    'hbond_sb': [(AID, AID) for each interaction],
    'hbond_bs': [], # deprecated but kept to maintain user scripts compatibility
    'hbond_self': [(donor AID, acceptor AID) for each interaction]
    } for each frame]
    Here 'b' denotes backbone and 's' sidechain.
    The order of donor/acceptor is not guaranteed for `hbond_sb` type
    in order to preserve the (sidechain, backbone) order.
    """

    def __init__(self, msys_model, cms_model, asl):
        self._backbone = cms_model.select_atom(BACKBONE_ASL)
        self._cms_model = cms_model
        aids = cms_model.select_atom(asl)
        self._analyzers = [
            HydrogenBondFinder(msys_model, cms_model, aids, aids)
        ]

    def _get_hbond_type(self, donor: int, acceptor: int) -> str:
        """
        Determine and return the type of hydrogen bond.
        'hbond_self' is a type of hydrogen bond between the sidechain and
        backbone of the same residue.

        :param donor: index of donor atom in the protein structure
        :param acceptor: index of acceptor atom in the protein structure
        :return: hydrogen bond type
        """
        d_tag = 'b' if donor in self._backbone else 's'
        a_tag = 'b' if acceptor in self._backbone else 's'
        if self._cms_model.atom[donor].getResidue(
        ) == self._cms_model.atom[acceptor].getResidue():
            return 'hbond_self'
        return f'hbond_{d_tag}{a_tag}'

    def _postcalc(self, *args):
        super()._postcalc(*args)
        self._result = {
            'hbond_bb': [],
            'hbond_ss': [],
            'hbond_sb': [],
            'hbond_bs': [],  # not used but kept not to break user scripts
            'hbond_self': []
        }
        for a, d in self._analyzers[0]():
            hb_type = self._get_hbond_type(donor=d, acceptor=a)
            # consolidate backbone-sidechain (bs) and sidechain-backbone(bs)
            # entries to improve statistics for this category.
            if hb_type == 'hbond_bs':
                hb_type, a, d = 'hbond_sb', d, a
            self._result[hb_type].append((d, a))


class _ProtProtSaltBridges(staf.CompositeAnalyzer):
    """
    Protein-protein salt bridge finder. Hbond result is not excluded.

    The result has the structure of
    [{
    'salt-bridge': [(anion atom AID, cation atom AID) for each bridge]
    } for each frame]
    """

    def __init__(self, msys_model, cms_model, asl):
        aids = cms_model.select_atom(asl)
        self._analyzers = [SaltBridgeFinder(msys_model, cms_model, aids, aids)]

    def _postcalc(self, *args):
        super()._postcalc(*args)
        self._result = {'salt-bridge': self._analyzers[0]()}


class ProtProtInter(staf.CompositeAnalyzer):
    """
    Protein-protein interactions.

    The result has the structure of:
    [{
    'pi-pi': [(an atom from ring1, an atom from ring2) for each interaction],
    'pi-cat': [(an atom from ring, an atom from cation) for each interaction],
    'salt-bridge': [(anion atom AID, cation atom AID) for each bridge],
    'hbond_bb': [(donor AID, acceptor AID) for each interaction],
    'hbond_ss': [(donor AID, acceptor AID) for each interaction],
    'hbond_sb': [(AID, AID) for each interaction],
    'hbond_self': [(donor AID, acceptor AID) for each interaction],
    'hbond_bs': [] # deprecated} for each frame]

    Here 'b' denotes backbone and 's' sidechain. For the same frame, results are
    unique up to residue level, e.g., even if there are multiple salt-bridges
    between residue A and B, only 1 is recorded.
    NOTES:
    - 'hbond_bs': is deprecated but kept for compatibility of user scripts
    - 'hbond_sb': the order of donor/acceptor is not guaranteed
    """

    def __init__(self, msys_model, cms_model, asl):
        """
        :type msys_model: `msys.System`
        :type  cms_model: `cms.Cms`
        :type        asl: `str`
        :param       asl: ASL expression to select protein atoms
        """
        args = (msys_model, cms_model, asl)
        self._analyzers = [
            ProtProtPiInter(*args),
            ProtProtHbondInter(*args),
            _ProtProtSaltBridges(*args),
        ]
        self._aid2label = partial(_prot_atom_label, cms_model, res_only=True)
        prot_aids = cms_model.select_atom(asl)
        prot_ct = _extract_with_original_id(cms_model.fsys_ct, prot_aids)
        self._tag2ca = self._get_tag2ca(prot_ct)

    def _get_tag2ca(self, prot_ct):
        """
        :param prot_ct: `schrodinger.structure.Structure`

        :rtype: `dict`
        """
        tag2ca = {}
        for res in structure.get_residues_by_connectivity(prot_ct):
            ca = res.getAlphaCarbon()
            if ca is None:
                continue
            ca_aid = ca.property[constants.ORIGINAL_INDEX]
            tag2ca[self._aid2label(ca_aid)] = ca_aid
        return tag2ca

    def _postcalc(self, *args):
        super()._postcalc(*args)
        self._result = {}
        for a in self._analyzers:
            self._result.update(a())
        self._cleanup_hbonds()

    def reduce(self, results, *_, **__):
        """
        Reduces the interaction data from all analyzers and frames into a
        summary of interaction counts.

        :param results: interactions of all frames
        :type  results: `list` of `dict`. Its length is the number of frames.

        :rtype : `dict`
        :return: counts of the various interactions over the whole trajectory
        """
        summary = {
            'pi-cat': [],
            'pi-pi': [],
            'salt-bridge': [],
            'hbond_bb': [],
            'hbond_ss': [],
            'hbond_sb': [],
            'hbond_bs': [],  # not used but kept not to break user scripts
            'hbond_self': []
        }
        for frame_result in results:
            for inter, pairs in frame_result.items():
                data = set(tuple(map(self._aid2label, pair)) for pair in pairs)
                summary[inter].extend(data)
        for inter, atoms in summary.items():
            summary[inter] = dict(Counter(atoms))
        return summary

    def _cleanup_hbonds(self):
        """
        Remove side-chain hbond if it is salt bridge. It is assumed that between
        two side-chains, there is at most 1 salt bridge and hbond interaction in
        total.
        """
        exclude = {
            tuple(sorted(map(self._aid2label, hbond)))
            for hbond in self._result['salt-bridge']
        }
        raw_hbss = self._result['hbond_ss']
        hbss = [(i, tuple(sorted(map(self._aid2label, hb))))
                for i, hb in enumerate(raw_hbss)]
        f = lambda hb: hb[1] not in exclude
        self._result['hbond_ss'] = [raw_hbss[hb[0]] for hb in filter(f, hbss)]


class AreaPerLipid(staf.GeomAnalyzerBase):
    """
    Calculate Area-Per-Lipid in from pure lipid bilayer simulation (no
    protein). This analyzer assumes the bilayer is symmetric -- number of lipids
    in upper and lower leaflets are the same.
    To calculate area-per-lipid, get X-Y area and divides by number of lipids/2
    """

    def __init__(self, msys_model, cms_model, membrane_asl: str = 'membrane'):
        """

        :type    msys_model: `msys.System`
        :type     cms_model: `cms.Cms`
        :type  membrane_asl: `asl`
        :param membrane_asl: selection to center the system along the Z-axis
        """
        self._membrane_asl = membrane_asl
        membrane_aids = cms_model.select_atom(membrane_asl)
        assert membrane_aids
        membrane_ct = cms_model.extract(membrane_aids)
        self.nlipids = membrane_ct.mol_total
        assert self.nlipids % 2 == 0
        self.half_nlipids = self.nlipids / 2.0

    def _postcalc(self, calc, _, fr):
        xy_area = fr.box[0][0] * fr.box[1][1]
        self._result = xy_area / self.half_nlipids


class MembraneDensityProfile(staf.GeomAnalyzerBase):
    """
    Calculate Density Profile along the Z axis. The results are returned in
    g/cm^3.
    """
    AMU_TO_GRAM = 1.66

    def __init__(self,
                 msys_model,
                 cms_model,
                 membrane_asl='membrane',
                 density_sel_asl='lipids',
                 slice_height=1.0,
                 z_min=-40,
                 z_max=40):
        """
        :type       msys_model: `msys.System`
        :type        cms_model: `cms.Cms`
        :type     membrane_asl: `asl`
        :param    membrane_asl: selection to center the system along the Z-axis
        :type  density_sel_asl: `asl`
        :param density_sel_asl: selection for density calculation
        :type     slice_height: `float`
        :param    slice_height: the height of the slices to use
        :type            z_min: int
        :param           z_min: the lower limit for which density will be calculated
        :type            z_max: int
        :param           z_max: the upper limit for which density will be calculated
        """
        self.membrane_asl = membrane_asl
        self.density_sel_asl = density_sel_asl

        if cms_model.need_msys:
            membrane_gids = topo.asl2gids(cms_model,
                                          membrane_asl,
                                          include_pseudoatoms=False)
            self.density_sel_gids = topo.asl2gids(cms_model,
                                                  density_sel_asl,
                                                  include_pseudoatoms=False)
            self.density_sel_mass = numpy.array(
                [msys_model.atom(gid).mass for gid in self.density_sel_gids])
        else:
            membrane_gids = cms_model.convert_to_gids(
                cms_model.select_atom(membrane_asl), with_pseudo=False)
            self.density_sel_gids = cms_model.convert_to_gids(
                cms_model.select_atom(density_sel_asl), with_pseudo=False)
            self.density_sel_mass = cms_model.get_mass(self.density_sel_gids)

        self._key = (msys_model, cms_model, tuple(membrane_gids))
        self.slice_height = slice_height

        # set up limits along the Z
        self.bins = numpy.arange(z_min, z_max + self.slice_height,
                                 self.slice_height)
        self.limits = list(zip(self.bins, self.bins[1:]))

    def _precalc(self, calc):
        # recenter frame along the Z axis
        calc.addCustom(staf.center_fr_along_z, self._key)

    def _postcalc(self, calc, *_):
        centered_fr = calc.getCustom(staf.center_fr_along_z)[self._key]
        # Get Z values for all selected gids
        z_pos = centered_fr.pos(self.density_sel_gids)[:, 2]
        slice_volume = centered_fr.box[0][0] * centered_fr.box[1][
            1] * numpy.abs(self.slice_height)

        self._result = [
            numpy.sum(self.density_sel_mass[numpy.logical_and(
                z_pos > zmin, z_pos < zmax)]) * self.AMU_TO_GRAM / slice_volume
            for zmin, zmax in self.limits
        ]

    def reduce(self, results, *_, **__):
        """
        Return (mean, std) density along the Z-axis
        """
        return numpy.mean(results, axis=0), numpy.std(results, axis=0)


class MembraneThickness(staf.GeomAnalyzerBase):
    """
    Calculate thickness of the membrane bilayer. For phospholipids a phosphate
    atom is often used as a proxy for each leaflet. The system is first
    centered along the Z-axis, and the distance betweent the <phosphates> in each
    leaflet is calculated.
    """

    def __init__(self,
                 msys_model,
                 cms_model,
                 membrane_asl='membrane',
                 proxy_asl='atom.ele P'):
        """
        :type    msys_model: `msys.System`
        :type     cms_model: `cms.Cms`
        :type  membrane_asl: `asl`
        :param membrane_asl: selection to center the system along the Z-axis
        :type     proxy_asl: `asl`
        :param    proxy_asl: selection of atom(s) to be used for thickness
        """
        if cms_model.need_msys:
            membrane_gids = topo.asl2gids(cms_model, membrane_asl, False)
            self.proxy_gids = topo.asl2gids(cms_model, proxy_asl, False)
        else:
            membrane_gids = cms_model.convert_to_gids(
                cms_model.select_atom(membrane_asl), with_pseudo=False)
            self.proxy_gids = cms_model.convert_to_gids(
                cms_model.select_atom(proxy_asl), with_pseudo=False)
        self._key = (msys_model, cms_model, tuple(membrane_gids))

    def _precalc(self, calc):
        # recenter frame along the Z axis
        calc.addCustom(staf.center_fr_along_z, self._key)

    def _postcalc(self, calc, *_):
        centered_fr = calc.getCustom(staf.center_fr_along_z)[self._key]
        # Get Z values for all selected gids
        z_pos = centered_fr.pos(self.proxy_gids)[:, 2]
        top_leaflet_pos = z_pos[z_pos > 0]
        bottom_leaflet_pos = z_pos[z_pos < 0]
        self._result = numpy.mean(top_leaflet_pos) - numpy.mean(
            bottom_leaflet_pos)
