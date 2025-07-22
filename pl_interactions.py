"""
Module with utilities for analyzing protein-ligand interactions from ark format.
"""
import itertools
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
from typing import Self

from schrodinger import structure
from schrodinger.utils import env
from schrodinger.utils import fileutils

with env.prepend_sys_path(fileutils.get_mmshare_scripts_dir()):
    from event_analysis_dir.res_data import ResData

# Either [frame_num, protein_label, lig_label] (pre-21-4 only)
# e.g.   [0 "A:THR_45:O" "L-FRAG_0:I"]
#
# or     [frame_num, protein_label, sub_type, lig_label]
# e.g.   [0 "B:VAL_227:H" b "L-FRAG_0:Cl3812"]
RawHalogenBondInteraction = list[int, str, str] | list[int, str, str, str]


class LigandData(NamedTuple):
    """
    :ivar fragment_name: Fragment name of the ligand e.g. L-FRAG_3
    :ivar atom_label: The atom in the ligand that is interacting with a protein.
        In the format of "Cl3812" where "Cl" is the element and "3812" is the
        atom index. Data from pre-21-4 will not have the atom index.
    """
    fragment_name: str
    atom_label: str

    @classmethod
    def fromFullName(cls, full_name: str) -> Self:
        fragment_name, atom_label = full_name.split(':')
        return cls(fragment_name, atom_label)


@dataclass
class PLResData(ResData):
    """
    Class to store information about a residue in a protein-ligand interaction.

    :ivar interaction_atom: The name of the atom in this residue that
        participates in the protein-ligand interaction.
    """
    interaction_atom: str = None

    @classmethod
    def fromFullName(cls, full_name: str) -> Self:
        """
        Create a PLResData object from a protein label e.g. "A:THR_45:O", which
        is a common format for PL interaction data in .eaf files.
        """
        res_data = super().fromFullName(full_name)
        res_data.interaction_atom = full_name.split(':')[-1]
        return res_data


@dataclass
class HalogenBondInteraction:
    """
    Class storing information about a halogen bond interaction between a residue
    from a protein and a ligand.

    :ivar frame_num: Frame number of the simulation.
    :ivar res_id: Information about the residue interacting with the ligand.
    :ivar sub_type: Halogen bond sub-type: "s" | "b" (sidechain/backbone).
    :ivar lig_label: Ligand atom label.
    """
    frame_num: int
    res_data: ResData
    sub_type: str
    lig_data: LigandData

    @classmethod
    def fromRawInteraction(cls, raw_inter: RawHalogenBondInteraction) -> Self:
        frame_num = raw_inter[0]
        res_data = PLResData.fromFullName(raw_inter[1])  # noqa: F841
        # pre 21-4 we did not store subtype (sidechain/backbone)
        if len(raw_inter) == 3:
            sub_type = _derive_halogen_bond_sub_type(
                protein_res_label=raw_inter[1])
            lig_data = LigandData.fromFullName(raw_inter[2])
        else:
            sub_type = raw_inter[2]
            lig_data = LigandData.fromFullName(raw_inter[3])

        return cls(frame_num, PLResData.fromFullName(raw_inter[1]), sub_type,
                   lig_data)


def parse_halogen_interactions(data: list[list[HalogenBondInteraction]],
                               parsed_data: np.ndarray,
                               protein_res_labels: list[str]) -> None:
    """
    Parse halogen bond interactions from `data` and store it in `parsed_data`.

    :param data: Halogen bond interactions for each frame of the simulation.
    :param parsed_data: 3D numpy array to store the parsed data. Has the shape:
        (num prot residues, num frames, num interaction types)
    :param protein_res_labels: List of protein residue labels in the full SID
        dataset.
    """
    sub_type_idx_map = {'s': 13, 'b': 14}
    for raw_inter in itertools.chain.from_iterable(data):
        inter = HalogenBondInteraction.fromRawInteraction(raw_inter)
        res_idx = protein_res_labels.index(inter.res_data.fullName())
        sub_type_idx = sub_type_idx_map[inter.sub_type]
        parsed_data[res_idx, inter.frame_num, sub_type_idx] += 1


def _derive_halogen_bond_sub_type(protein_res_label: str) -> str:
    """
    Derive the halogen bond sub-type (sidechain/backbone) from a protein residue
    label.
    """
    is_backbone = protein_res_label.split(':')[-1] in ['H', 'O']
    return 'b' if is_backbone else 's'


def get_halogen_bond_statistics(fstart: int,
                                fend: int,
                                hal_bonds: list[list[tuple]] | None,
                                hal_prot_dict: dict[str, int] | None,
                                ligand_atom_dict: dict,
                                pose_st: structure.Structure
                                ) -> list[list]:  # yapf: disable
    """
    Get statistics on Halogen bonds during the simulation.

    :param fstart: Starting frame index
    :param fend: Ending frame index
    :param hal_bonds: List of all halogen bonds present in each frame of the
        simulation.
    :param hal_prot_dict: Dictionary mapping protein atom labels to their
        corresponding indices in the protein structure e.g.
        {'B:VAL_41:O': 469, ...}.
    :param ligand_atom_dict: Dictionary mapping labels for ligands participating
        in halogen bonds to its corresponding atom index in the ligand
        structure e.g. {'L-FRAG_0:C3810': 3810, ...}
    :param pose_st: The pose structure
    """
    if hal_bonds is None or hal_prot_dict is None:
        return []
    nframes = fend + 1 - fstart
    results = {}
    all_hal_bonds = []
    for frame in hal_bonds[fstart:fend + 1]:
        for hal_bonds in frame:
            if len(hal_bonds) == 3:
                # pre 21-4, we did not store subtype (sidechain/backbone)
                subtype = _derive_halogen_bond_sub_type(
                    protein_res_label=hal_bonds[1])
                all_hal_bonds.append((hal_bonds[1], subtype, hal_bonds[2]))
            else:
                all_hal_bonds.append((hal_bonds[1:]))
    unique = set(all_hal_bonds)
    unique_filtered = filter_degen_res_atoms(unique)
    for hal_bond, hal_bond_filtered in zip(unique, unique_filtered):
        lig_aid = ligand_atom_dict[hal_bond[2]]
        count = all_hal_bonds.count(hal_bond)
        aid = hal_prot_dict[hal_bond[0]]
        xyz = pose_st.atom[aid].xyz
        key = (hal_bond_filtered[0], hal_bond[1], lig_aid)
        count += results[key][0] if key in results else 0
        results[key] = [count, hal_bond[0], aid, xyz, lig_aid]

    res = []
    for v in list(results.values()):
        res.append([v[0] / nframes] + v[1:])
    return res


def filter_degen_res_atoms(bonds: tuple[tuple]):
    """
    This function takes a list of bonds and looks at the Protein Residue tag
    (ie: 'A:LYS_33:1HZ'), and determines if there are atoms there are equivalent
    bond donors/acceptors in that residue.  If so, then the atom name is replced
    by a more a more generic name.

    We need this for LID, in the scenarious when equivelent residue
    donors/acceptors interact with the same ligand atom.  Multple equivelent
    interactions are created in LID.
    """
    modify_res = ['LYS', 'ARG', 'ASP', 'ASN', 'GLU', 'GLN', 'HIS']
    LYS_pdb_atm = ['1HZ', '2HZ', '3HZ', 'HZ1', 'HZ2', 'HZ3']
    ARG_pdb_atm = [
        'HE', '1HH1', '2HH1', '1HH2', '2HH2', 'HH11', 'HH12', 'HH21', 'HH22'
    ]
    ARG_pdb_heavy = ['NH1', 'NH2']

    ASP_pdb_atm = ['OD1', 'OD2']
    ASN_pdb_atm = ['1HD2', '2HD2', 'HD21', 'HD22']
    GLU_pdb_atm = ['OE1', 'OE2']
    GLN_pdb_atm = ['1HE2', '2HE2', 'HE21', 'HE22']
    HIS_pdb_atm = ['ND1', 'NE2', 'HD1', 'HE2']

    replace_dict = {
        'LYS': 'HX',
        'ARG': 'HHX',
        'GLN': 'HEX',
        'GLU': 'OEX',
        'ASP': 'ODX',
        'ASN': 'HDX',
        'HIS': 'NX',
        'ARG_heavy': 'NX'
    }

    updated_list = []

    for bond in bonds:
        res_tag = bond[0]
        c, resname, resnum, atom_name = parse_res_name_with_atoms(res_tag)
        if resname in modify_res:
            if resname in ['HIS', 'HIE', 'HIP'] and atom_name in HIS_pdb_atm:
                if atom_name[0] == 'H':
                    atom_name = 'HX'
                else:
                    atom_name = replace_dict['HIS']
            elif resname == 'GLN' and atom_name in GLN_pdb_atm:
                atom_name = replace_dict['GLN']
            elif resname in ['GLU', 'GLH'] and atom_name in GLU_pdb_atm:
                atom_name = replace_dict['GLU']
            elif resname == 'ASN' and atom_name in ASN_pdb_atm:
                atom_name = replace_dict['ASN']
            elif resname in ['ASP', 'ASH'] and atom_name in ASP_pdb_atm:
                atom_name = replace_dict['ASP']
            elif resname in ['ARG', 'ARN']:
                if atom_name in ARG_pdb_atm:
                    atom_name = replace_dict['ARG']
                elif atom_name in ARG_pdb_heavy:
                    atom_name = replace_dict['ARG_heavy']
            elif resname in ['LYS', 'LYN'] and atom_name in LYS_pdb_atm:
                atom_name = replace_dict['LYS']
            bond = list(bond)
            bond[0] = '%s:%s_%s:%s' % (c, resname, resnum, atom_name)
            bond = tuple(bond)
        updated_list.append(bond)
    return updated_list


def parse_res_name_with_atoms(res_tag: str) -> tuple[str, str, int, str]:
    k = res_tag.split(':')
    chain = k[0]
    atom_name = k[2]
    t = k[1].split('_')
    res_name = t[0]
    res_num = t[1]
    return (chain, res_name, res_num, atom_name)
