r"""
Customer-facing script for protein-protein complex MD trajectory analysis using Schrodinger Suite.

This script reads a Desmond MD trajectory and analyzes protein-protein interactions
across the simulation, reporting occupancy and generating summary outputs and plots.

Usage:
    python analyze_simulation.py --cms <structure.cms> --traj <trajectory_dir> \
        --chain-groups A,B C,D --occupancy-threshold 0.2 --output-prefix results

:author: Your Company Name
:date: 2024-06-07
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
from ppi_interaction_diagram import consolidated_ppi_diagram
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches

# Schrodinger imports
from schrodinger.structure import StructureReader
from schrodinger.structutils.analyze import hbond
from schrodinger.application.desmond.packages import traj
try:
    from schrodinger.utils import log as schrod_log
    logger = schrod_log.get_output_logger("PPIAnalysis")
except ImportError:
    import logging
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
    logger = logging.getLogger("PPIAnalysis")

# Schrödinger professional amino acid color scheme
AMINO_ACID_COLORS = {
    'ALA': '#87CEEB', 'VAL': '#4682B4', 'LEU': '#1E90FF', 'ILE': '#0000CD',
    'MET': '#4169E1', 'PHE': '#191970', 'TRP': '#000080', 'PRO': '#483D8B',
    'GLY': '#98FB98', 'SER': '#90EE90', 'THR': '#32CD32', 'CYS': '#228B22',
    'TYR': '#006400', 'ASN': '#20B2AA', 'GLN': '#48D1CC',
    'ASP': '#FFB6C1', 'GLU': '#FF69B4',
    'LYS': '#9370DB', 'ARG': '#8A2BE2', 'HIS': '#4B0082',
    'UNK': '#C0C0C0',
}

# --- Constants (from Schrodinger analysis.py) ---
HBOND_CUTOFF = 2.8
SALT_BRIDGE_CUTOFF = 5.0
PI_STACK_CUTOFF = 6.0
PI_CATION_CUTOFF = 6.0
HYDROPHOBIC_CUTOFF = 3.6
AROMATIC_RES = {'PHE', 'TYR', 'TRP', 'HIS'}
CATIONIC_RES = {'ARG', 'LYS'}
ANIONIC_RES = {'ASP', 'GLU'}
HYDROPHOBIC_RES = {'PHE', 'LEU', 'ILE', 'TYR', 'TRP', 'VAL', 'MET', 'PRO', 'CYS', 'ALA'}
NON_PROTEIN = {'HOH', 'WAT', 'TIP3', 'SOL', 'SPC', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN'}

def parse_chain_groups(chain_groups_list):
    r"""
    Parse chain group arguments from command line.

    :param list[str] chain_groups_list: List of chain group strings (e.g., ['A,B', 'C,D'])
    :return: List of chain group lists (e.g., [['A', 'B'], ['C', 'D']])
    :rtype: list[list[str]]
    """
    return [group.split(',') for group in chain_groups_list]

def get_chain_ids(cms_file):
    """
    Extract all chain IDs from the given CMS file using Schrodinger StructureReader.
    :param cms_file: Path to the CMS file
    :type cms_file: Path
    :return: Sorted list of unique chain IDs
    :rtype: list[str]
    """
    chain_ids = set()
    with StructureReader(str(cms_file)) as reader:
        for st in reader:
            for atom in st.atom:
                res = atom.getResidue()
                if res.chain:
                    chain_ids.add(str(res.chain))
    return sorted(chain_ids)

class ProteinProteinInteractionAnalyzer:
    def __init__(self, structure, chain_groups: List[List[str]], logger):
        self.structure = structure
        self.chain_groups = chain_groups
        self.logger = logger

    def analyze_hydrogen_bonds(self, structure, group1: List[str], group2: List[str]) -> List[Dict[str, Any]]:
        hbonds = hbond.get_hydrogen_bonds(structure, max_dist=HBOND_CUTOFF)
        interactions = []
        for hb in hbonds:
            donor_atom, acceptor_atom = hb[0], hb[1]
            donor_res = donor_atom.getResidue()
            acceptor_res = acceptor_atom.getResidue()
            donor_chain = str(donor_res.chain)
            acceptor_chain = str(acceptor_res.chain)
            if donor_res.pdbres.strip() in NON_PROTEIN or acceptor_res.pdbres.strip() in NON_PROTEIN:
                continue
            if ((donor_chain in group1 and acceptor_chain in group2) or
                (donor_chain in group2 and acceptor_chain in group1)):
                interactions.append({
                    'type': 'hydrogen_bond',
                    'donor': f"{donor_chain}:{donor_res.pdbres.strip()}{donor_res.resnum}",
                    'acceptor': f"{acceptor_chain}:{acceptor_res.pdbres.strip()}{acceptor_res.resnum}",
                    'distance': np.linalg.norm(np.array(donor_atom.xyz) - np.array(acceptor_atom.xyz))
                })
        return interactions

    def analyze_salt_bridges(self, structure, group1: List[str], group2: List[str]) -> List[Dict[str, Any]]:
        interactions = []
        # Vectorized: collect charged atoms by group
        charged_atoms1, charged_atoms2 = [], []
        for atom in structure.atom:
            res = atom.getResidue()
            chain = str(res.chain)
            if res.pdbres.strip() in NON_PROTEIN:
                continue
            if chain in group1 and res.pdbres.strip() in (CATIONIC_RES | ANIONIC_RES):
                charged_atoms1.append(atom)
            elif chain in group2 and res.pdbres.strip() in (CATIONIC_RES | ANIONIC_RES):
                charged_atoms2.append(atom)
        if charged_atoms1 and charged_atoms2:
            coords1 = np.array([a.xyz for a in charged_atoms1])
            coords2 = np.array([a.xyz for a in charged_atoms2])
            dists = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=2)
            idxs = np.where(dists <= SALT_BRIDGE_CUTOFF)
            for i, j in zip(*idxs):
                res1 = charged_atoms1[i].getResidue()
                res2 = charged_atoms2[j].getResidue()
                interactions.append({
                    'type': 'salt_bridge',
                    'res1': f"{res1.chain}:{res1.pdbres.strip()}{res1.resnum}",
                    'res2': f"{res2.chain}:{res2.pdbres.strip()}{res2.resnum}",
                    'distance': dists[i, j]
                })
        return interactions

    def analyze_pi_pi(self, structure, group1: List[str], group2: List[str]) -> List[Dict[str, Any]]:
        interactions = []
        arom1, arom2 = [], []
        for atom in structure.atom:
            res = atom.getResidue()
            chain = str(res.chain)
            if res.pdbres.strip() not in AROMATIC_RES:
                continue
            if chain in group1:
                arom1.append(res)
            elif chain in group2:
                arom2.append(res)
        centroids1 = [self.get_ring_centroid(r) for r in arom1]
        centroids2 = [self.get_ring_centroid(r) for r in arom2]
        for i, c1 in enumerate(centroids1):
            if c1 is None:
                continue
            for j, c2 in enumerate(centroids2):
                if c2 is None:
                    continue
                dist = np.linalg.norm(c1 - c2)
                if dist <= PI_STACK_CUTOFF:
                    interactions.append({
                        'type': 'pi_pi',
                        'res1': f"{arom1[i].chain}:{arom1[i].pdbres.strip()}{arom1[i].resnum}",
                        'res2': f"{arom2[j].chain}:{arom2[j].pdbres.strip()}{arom2[j].resnum}",
                        'distance': dist
                    })
        return interactions

    def analyze_pi_cation(self, structure, group1: List[str], group2: List[str]) -> List[Dict[str, Any]]:
        interactions = []
        aroms, cations = [], []
        for atom in structure.atom:
            res = atom.getResidue()
            chain = str(res.chain)
            if res.pdbres.strip() in AROMATIC_RES and chain in group1:
                aroms.append(res)
            elif res.pdbres.strip() in CATIONIC_RES and chain in group2:
                cations.append(res)
        for arom in aroms:
            c1 = self.get_ring_centroid(arom)
            if c1 is None:
                continue
            for cat in cations:
                c2 = self.get_cation_center(cat)
                if c2 is None:
                    continue
                dist = np.linalg.norm(c1 - c2)
                if dist <= PI_CATION_CUTOFF:
                    interactions.append({
                        'type': 'pi_cation',
                        'aromatic': f"{arom.chain}:{arom.pdbres.strip()}{arom.resnum}",
                        'cation': f"{cat.chain}:{cat.pdbres.strip()}{cat.resnum}",
                        'distance': dist
                    })
        return interactions

    @staticmethod
    def get_ring_centroid(res) -> Any:
        ring_atoms = [atom for atom in res.atom if atom.pdbname.strip() in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH2', 'NE1', 'CE3', 'CZ2', 'CZ3', 'CD', 'CE', 'NZ', 'ND1', 'NE2']]
        if not ring_atoms:
            return None
        coords = np.array([atom.xyz for atom in ring_atoms])
        return np.mean(coords, axis=0)

    @staticmethod
    def get_cation_center(res) -> Any:
        if res.pdbres.strip() == 'ARG':
            atom_names = ['NH1', 'NH2', 'CZ']
        elif res.pdbres.strip() == 'LYS':
            atom_names = ['NZ']
        else:
            return None
        atoms = [atom for atom in res.atom if atom.pdbname.strip() in atom_names]
        if not atoms:
            return None
        coords = np.array([atom.xyz for atom in atoms])
        return np.mean(coords, axis=0)

def analyze_hydrophobic(structure, group1, group2):
    """Detect inter-chain hydrophobic interactions."""
    interactions = []
    hydros1, hydros2 = [], []
    for atom in structure.atom:
        res = atom.getResidue()
        chain = str(res.chain)
        if res.pdbres.strip() in HYDROPHOBIC_RES:
            if chain in group1:
                hydros1.append(atom)
            elif chain in group2:
                hydros2.append(atom)
    for a1 in hydros1:
        for a2 in hydros2:
            dist = np.linalg.norm(np.array(a1.xyz) - np.array(a2.xyz))
            if dist <= HYDROPHOBIC_CUTOFF:
                interactions.append({
                    'type': 'hydrophobic',
                    'atom1': f"{a1.getResidue().chain}:{a1.getResidue().pdbres.strip()}{a1.getResidue().resnum}:{a1.pdbname.strip()}",
                    'atom2': f"{a2.getResidue().chain}:{a2.getResidue().pdbres.strip()}{a2.getResidue().resnum}:{a2.pdbname.strip()}",
                    'distance': dist
                })
    return interactions

def plot_protein_protein_network(interactions, output_file="ppi_network.png"):
    if not interactions:
        print(f"⚠️ No interactions found. Skipping plot: {output_file}")
        return
    G = nx.Graph()
    edge_labels = {}
    node_chain = {}
    node_resname = {}
    node_resnum = {}
    for inter in interactions:
        if 'donor' in inter and 'acceptor' in inter:
            n1, n2 = inter['donor'], inter['acceptor']
        elif 'res1' in inter and 'res2' in inter:
            n1, n2 = inter['res1'], inter['res2']
        else:
            continue
        for n in [n1, n2]:
            if n not in G:
                try:
                    chain, rest = n.split(':')
                    resname = rest[:3]
                    resnum = rest[3:]
                except Exception:
                    chain, resname, resnum = ' ', 'UNK', ''
                node_chain[n] = chain
                node_resname[n] = resname
                node_resnum[n] = resnum
                G.add_node(n)
        G.add_edge(n1, n2, type=inter['type'])
        if 'occupancy' in inter:
            percent = int(round(inter['occupancy'] * 100))
            edge_labels[(n1, n2)] = f"{percent}%"
    # Layout: left/right with vertical spread
    chains = list({node_chain[n] for n in G.nodes})
    pos = {}
    chain_y = {c: 0 for c in chains}
    for node in G.nodes():
        chain = node_chain[node]
        x = -1 if chain == chains[0] else 1
        y = chain_y[chain]
        pos[node] = (x, y)
        chain_y[chain] += 1
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    node_size = 2200
    # Draw nodes as perfect circles with AA color fill and AA color border
    for node in G.nodes():
        x, y = pos[node]
        resname = node_resname[node]
        fill_color = AMINO_ACID_COLORS.get(resname, AMINO_ACID_COLORS['UNK'])
        border_color = AMINO_ACID_COLORS.get(resname, AMINO_ACID_COLORS['UNK'])
        circ = mpatches.Circle((x, y), radius=(node_size/20000), facecolor=fill_color, edgecolor=border_color, linewidth=3, zorder=2)
        ax.add_patch(circ)
    # Custom node labels: all centered and inside the node, with white outline for contrast
    for node in G.nodes():
        x, y = pos[node]
        chain = node_chain[node]
        aa = node_resname[node]
        num = node_resnum[node]
        label_text = f"{chain}\n{aa}\n{num}"
        text = ax.text(x, y, label_text, fontsize=12, fontweight='bold', ha='center', va='center', color='black', zorder=4, linespacing=1.1)
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='white'),
            path_effects.Normal()
        ])
    # Draw edges and occupancy labels
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, ax=ax)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='purple', font_size=13, label_pos=0.5, ax=ax)
    plt.title("Protein-Protein Interaction Network", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✅ Network plot saved to {output_file}")

# Define interaction types after all analyze_* functions are defined
INTERACTION_TYPES = [
    ('hydrogen_bond', 'analyze_hydrogen_bonds'),
    ('salt_bridge', 'analyze_salt_bridges'),
    ('pi_pi', 'analyze_pi_pi'),
    ('pi_cation', 'analyze_pi_cation'),
]

def main():
    r"""
    Main function for customer-facing protein-protein MD trajectory analysis.
    Parses arguments, runs analysis, and outputs results and plots.
    """
    parser = argparse.ArgumentParser(
        description=r"""
        Standalone script for protein-protein MD trajectory analysis using Schrodinger Suite.
        Reports occupancy and generates summary outputs and plots.
        """
    )
    parser.add_argument('--cms', required=True, type=str,
                        help='Path to the Desmond .cms structure file')
    parser.add_argument('--traj', required=True, type=str,
                        help='Path to the trajectory file or directory')
    parser.add_argument('--chain-groups', nargs='+', default=None,
                        help='Chain groups to analyze (e.g., A,B C,D). If not specified, available chains will be listed.')
    parser.add_argument('--occupancy-threshold', type=float, default=0.2,
                        help='Minimum occupancy threshold (default: 0.2)')
    parser.add_argument('--output-prefix', type=str, default='ppi_analysis',
                        help='Prefix for output files (default: ppi_analysis)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debug/progress logging')

    args = parser.parse_args()
    debug = args.debug

    cms_file = Path(args.cms)
    trajectory_file = Path(args.traj)
    occupancy_threshold = args.occupancy_threshold
    output_prefix = args.output_prefix

    # Detect and print available chain IDs
    chain_ids = get_chain_ids(cms_file)
    print(f"\n🔬 Protein-Protein MD Trajectory Analysis")
    print(f"CMS file: {cms_file}")
    print(f"Trajectory: {trajectory_file}")
    print(f"Available chains in structure: {chain_ids}")
    print(f"Occupancy threshold: {occupancy_threshold}")
    print(f"Output prefix: {output_prefix}\n")

    if args.chain_groups is None:
        print("ℹ️  Please specify --chain-groups using the available chain IDs above.")
        print("    Example: --chain-groups {}".format(' '.join([','.join(chain_ids[:2])])))
        return

    chain_groups = parse_chain_groups(args.chain_groups)

    # Validate that all specified chains exist
    missing_chains = set()
    for group in chain_groups:
        for chain in group:
            if chain not in chain_ids:
                missing_chains.add(chain)
    if missing_chains:
        print(f"⚠️ Warning: The following specified chains are not present in the structure: {sorted(missing_chains)}")
        print(f"   Available chains: {chain_ids}")
        print(f"   Please check your --chain-groups argument.")
        return

    print(f"Chain groups for analysis: {chain_groups}")

    # Load structure
    with StructureReader(str(cms_file)) as reader:
        structure = next(reader)

    # Print atom, chain, and residue info for debugging
    print("\n=== Structure Debug Info ===")
    print("First 10 atoms and their chain/residue info:")
    for atom in list(structure.atom)[:10]:
        res = atom.getResidue()
        print(f"Atom {atom.index}: Chain={res.chain}, Residue={res.pdbres}{res.resnum}")

    # Print chain group composition
    for idx, group in enumerate(chain_groups):
        atoms_in_group = [atom for atom in structure.atom if str(atom.getResidue().chain) in group]
        print(f"\nGroup {idx+1} ({group}): {len(atoms_in_group)} atoms")
        residues = []
        for atom in atoms_in_group:
            res = atom.getResidue()
            residues.append((str(res.chain), res.pdbres, res.resnum))
        unique_residues = list(dict.fromkeys(residues))
        print(f"  First 5 residues in group {idx+1}:")
        for r in unique_residues[:5]:
            print(f"    Chain={r[0]}, Residue={r[1]}{r[2]}")

    # Load trajectory
    trj_frames = list(traj.read_traj(str(trajectory_file)))
    print(f"\n📊 Found {len(trj_frames)} trajectory frames\n")

    # Analyze first N frames (for demonstration, use 10 or all)
    max_frames = min(10, len(trj_frames))
    analyzer = ProteinProteinInteractionAnalyzer(structure, chain_groups, logger)
    persistent_by_type = {}
    for interaction_type, method_name in INTERACTION_TYPES:
        interaction_frames = defaultdict(set)
        interaction_details = {}
        if debug:
            logger.info(f"\n[DEBUG] Starting analysis for {interaction_type}...")
        for i in range(max_frames):
            if debug:
                logger.info(f"[DEBUG] Frame {i} - {interaction_type} analysis started.")
            frame = trj_frames[i]
            frame_structure = structure.copy()
            positions = frame.pos()
            for j, atom in enumerate(frame_structure.atom):
                if j < len(positions):
                    atom.xyz = positions[j]
            frame_interactions = []
            for idx, group1 in enumerate(chain_groups):
                for group2 in chain_groups[idx+1:]:
                    if debug:
                        logger.info(f"[DEBUG] Frame {i} - {interaction_type}: group1={group1}, group2={group2}")
                    # Print candidate residue counts for this group pair
                    if interaction_type == 'hydrogen_bond':
                        candidates1 = [atom for atom in frame_structure.atom if str(atom.getResidue().chain) in group1]
                        candidates2 = [atom for atom in frame_structure.atom if str(atom.getResidue().chain) in group2]
                        if debug:
                            logger.info(f"[DEBUG] Frame {i} - {interaction_type}: group1 atoms={len(candidates1)}, group2 atoms={len(candidates2)}")
                    analyze_func = getattr(analyzer, method_name)
                    frame_interactions.extend(analyze_func(frame_structure, group1, group2))
            if debug:
                logger.info(f"[DEBUG] Frame {i} - {interaction_type}: {len(frame_interactions)} interactions found.")
            for interaction in frame_interactions:
                key = (interaction['donor'], interaction['acceptor']) if 'donor' in interaction else (interaction.get('res1', ''), interaction.get('res2', ''))
                interaction_frames[key].add(i)
                if key not in interaction_details:
                    interaction_details[key] = interaction
            if debug:
                logger.info(f"[DEBUG] Frame {i} - {interaction_type} analysis complete.")
        persistent_interactions = []
        for key, frames in interaction_frames.items():
            occupancy = len(frames) / max_frames
            if occupancy >= occupancy_threshold:
                inter = dict(interaction_details[key])
                inter['occupancy'] = occupancy
                persistent_interactions.append(inter)
        persistent_by_type[interaction_type] = persistent_interactions
        logger.info(f"\n=== Consistent {interaction_type.replace('_', ' ').title()}s (occupancy ≥ {occupancy_threshold*100:.0f}%): ===")
        for inter in persistent_interactions:
            logger.info(str(inter))
        logger.info(f"\n✅ Found {len(persistent_interactions)} consistent {interaction_type.replace('_', ' ')} interactions across {max_frames} frames.")
        plot_protein_protein_network(persistent_interactions, output_file=f"ppi_network_{interaction_type}.png")
        if debug:
            logger.info(f"[DEBUG] Finished analysis and plotting for {interaction_type}.")

    consolidated_ppi_diagram(persistent_by_type, output_file="ppi_consolidated.png")

if __name__ == "__main__":
    main()