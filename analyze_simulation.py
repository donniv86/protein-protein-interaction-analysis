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

# Schrodinger imports
from schrodinger.structure import StructureReader
from schrodinger.structutils.analyze import hbond
from schrodinger.application.desmond.packages import traj

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

def analyze_hydrogen_bonds(structure, group1, group2):
    """Detect inter-chain hydrogen bonds between two chain groups."""
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

def analyze_salt_bridges(structure, group1, group2):
    """Detect inter-chain salt bridges between two chain groups."""
    interactions = []
    # Collect charged residues by chain group
    charged1, charged2 = [], []
    for atom in structure.atom:
        res = atom.getResidue()
        chain = str(res.chain)
        if res.pdbres.strip() in NON_PROTEIN:
            continue
        if chain in group1 and res.pdbres.strip() in (CATIONIC_RES | ANIONIC_RES):
            charged1.append(res)
        elif chain in group2 and res.pdbres.strip() in (CATIONIC_RES | ANIONIC_RES):
            charged2.append(res)
    # Check all pairs
    for res1 in charged1:
        for res2 in charged2:
            if ((res1.pdbres.strip() in CATIONIC_RES and res2.pdbres.strip() in ANIONIC_RES) or
                (res1.pdbres.strip() in ANIONIC_RES and res2.pdbres.strip() in CATIONIC_RES)):
                for atom1 in res1.atom:
                    for atom2 in res2.atom:
                        dist = np.linalg.norm(np.array(atom1.xyz) - np.array(atom2.xyz))
                        if dist <= SALT_BRIDGE_CUTOFF:
                            interactions.append({
                                'type': 'salt_bridge',
                                'res1': f"{res1.chain}:{res1.pdbres.strip()}{res1.resnum}",
                                'res2': f"{res2.chain}:{res2.pdbres.strip()}{res2.resnum}",
                                'distance': dist
                            })
    return interactions

def get_ring_centroid(res):
    """Return centroid of aromatic ring atoms for pi interactions."""
    ring_atoms = [atom for atom in res.atom if atom.pdbname.strip() in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH2', 'NE1', 'CE3', 'CZ2', 'CZ3', 'CD', 'CE', 'NZ', 'ND1', 'NE2']]
    if not ring_atoms:
        return None
    coords = np.array([atom.xyz for atom in ring_atoms])
    return np.mean(coords, axis=0)

def analyze_pi_pi(structure, group1, group2):
    """Detect inter-chain pi-pi stacking interactions."""
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
    for res1 in arom1:
        c1 = get_ring_centroid(res1)
        if c1 is None:
            continue
        for res2 in arom2:
            c2 = get_ring_centroid(res2)
            if c2 is None:
                continue
            dist = np.linalg.norm(c1 - c2)
            if dist <= PI_STACK_CUTOFF:
                interactions.append({
                    'type': 'pi_pi',
                    'res1': f"{res1.chain}:{res1.pdbres.strip()}{res1.resnum}",
                    'res2': f"{res2.chain}:{res2.pdbres.strip()}{res2.resnum}",
                    'distance': dist
                })
    return interactions

def get_cation_center(res):
    """Return center of cationic group for pi-cation interactions."""
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

def analyze_pi_cation(structure, group1, group2):
    """Detect inter-chain pi-cation interactions."""
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
        c1 = get_ring_centroid(arom)
        if c1 is None:
            continue
        for cat in cations:
            c2 = get_cation_center(cat)
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

def analyze_frame(structure, chain_groups):
    """Analyze all interaction types for a single frame."""
    all_interactions = []
    for i, group1 in enumerate(chain_groups):
        for group2 in chain_groups[i+1:]:
            all_interactions.extend(analyze_hydrogen_bonds(structure, group1, group2))
    return all_interactions

def plot_protein_protein_network(interactions, output_file="ppi_network.png"):
    """
    Create a 2D network plot of protein-protein interactions.
    :param interactions: List of dicts with keys for residue pairs (e.g., 'donor'/'acceptor' or 'res1'/'res2')
    :param output_file: Output image file name
    """
    G = nx.Graph()
    for inter in interactions:
        if 'donor' in inter and 'acceptor' in inter:
            n1, n2 = inter['donor'], inter['acceptor']
        elif 'res1' in inter and 'res2' in inter:
            n1, n2 = inter['res1'], inter['res2']
        else:
            continue  # skip if keys are missing
        G.add_node(n1, chain=n1.split(':')[0])
        G.add_node(n2, chain=n2.split(':')[0])
        G.add_edge(n1, n2, type=inter['type'])

    # Layout: separate chains on left/right
    chains = list({G.nodes[n]['chain'] for n in G.nodes})
    pos = {}
    chain_y = {c: 0 for c in chains}
    for node in G.nodes():
        chain = G.nodes[node]['chain']
        x = -1 if chain == chains[0] else 1
        y = chain_y[chain]
        pos[node] = (x, y)
        chain_y[chain] += 1

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
    plt.title("Protein-Protein Interaction Network")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✅ Network plot saved to {output_file}")

# Define interaction types after all analyze_* functions are defined
INTERACTION_TYPES = [
    ('hydrogen_bond', analyze_hydrogen_bonds),
    ('salt_bridge', analyze_salt_bridges),
    ('pi_pi', analyze_pi_pi),
    ('pi_cation', analyze_pi_cation),
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

    args = parser.parse_args()

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
    persistent_by_type = {}
    for interaction_type, analyze_func in INTERACTION_TYPES:
        interaction_frames = defaultdict(set)  # (donor, acceptor) -> set of frame indices
        interaction_details = {}  # (donor, acceptor) -> example interaction dict
        for i in range(max_frames):
            frame = trj_frames[i]
            frame_structure = structure.copy()
            positions = frame.pos()
            for j, atom in enumerate(frame_structure.atom):
                if j < len(positions):
                    atom.xyz = positions[j]
            # Analyze this interaction type
            frame_interactions = []
            for idx, group1 in enumerate(chain_groups):
                for group2 in chain_groups[idx+1:]:
                    frame_interactions.extend(analyze_func(frame_structure, group1, group2))
            # Track which frames each interaction appears in
            for interaction in frame_interactions:
                key = (interaction['donor'], interaction['acceptor']) if 'donor' in interaction else (interaction.get('res1', ''), interaction.get('res2', ''))
                interaction_frames[key].add(i)
                if key not in interaction_details:
                    interaction_details[key] = interaction
        # Calculate occupancy and filter for persistent interactions
        persistent_interactions = []
        for key, frames in interaction_frames.items():
            occupancy = len(frames) / max_frames
            if occupancy >= occupancy_threshold:
                inter = dict(interaction_details[key])
                inter['occupancy'] = occupancy
                persistent_interactions.append(inter)
        persistent_by_type[interaction_type] = persistent_interactions
        print(f"\n=== Consistent {interaction_type.replace('_', ' ').title()}s (occupancy ≥ {occupancy_threshold*100:.0f}%): ===")
        for inter in persistent_interactions:
            print(inter)
        print(f"\n✅ Found {len(persistent_interactions)} consistent {interaction_type.replace('_', ' ')} interactions across {max_frames} frames.")
        # Generate 2D network plot for this interaction type
        plot_protein_protein_network(persistent_interactions, output_file=f"ppi_network_{interaction_type}.png")

if __name__ == "__main__":
    main()