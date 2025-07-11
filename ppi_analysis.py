#!/usr/bin/env python3
"""
ppi_analysis.py
Professional script for protein-protein non-bonding interaction analysis and visualization.
Combines analysis and plotting in a single, reproducible, developer-friendly module.
"""

import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import networkx as nx
from scipy.spatial import ConvexHull

# Schrodinger color schemes
AMINO_ACID_COLORS = {
    'ALA': '#87CEEB', 'VAL': '#4682B4', 'LEU': '#1E90FF', 'ILE': '#0000CD',
    'MET': '#4169E1', 'PHE': '#191970', 'TRP': '#000080', 'PRO': '#483D8B',
    'GLY': '#98FB98', 'SER': '#90EE90', 'THR': '#32CD32', 'CYS': '#228B22',
    'TYR': '#006400', 'ASN': '#20B2AA', 'GLN': '#48D1CC',
    'ASP': '#FFB6C1', 'GLU': '#FF69B4',
    'LYS': '#9370DB', 'ARG': '#8A2BE2', 'HIS': '#4B0082',
    'UNK': '#C0C0C0',
}
CHAIN_COLORS = {
    'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 'D': '#96CEB4',
    'E': '#FFEAA7', 'F': '#DDA0DD', 'L': '#FFA07A', 'default': '#87CEEB'
}
INTERACTION_STYLES = {
    'hydrogen_bond': {'color': '#2E86AB', 'style': 'dashed', 'width': 2.0, 'label': 'Hydrogen Bond'},
    'salt_bridge': {'color': '#2E8B57', 'style': 'solid', 'width': 3.0, 'label': 'Salt Bridge'},
    'pi_pi': {'color': '#8A2BE2', 'style': 'dotted', 'width': 2.5, 'label': 'π-π Stacking'},
    'pi_cation': {'color': '#FF4500', 'style': 'dashdot', 'width': 2.5, 'label': 'π-Cation'},
}

# --- Analysis Class ---
class PPIAnalyzer:
    """
    Analyze protein-protein non-bonding interactions from MD trajectories.
    """
    def __init__(self, structure, chain_groups: List[List[str]], logger):
        self.structure = structure
        self.chain_groups = chain_groups
        self.logger = logger

    def analyze_hydrogen_bonds(self, structure, group1: List[str], group2: List[str]) -> List[Dict[str, Any]]:
        from schrodinger.structutils.analyze import hbond
        NON_PROTEIN = {'HOH', 'WAT', 'TIP3', 'SOL', 'SPC', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN'}
        HBOND_CUTOFF = 2.8
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
        CATIONIC_RES = {'ARG', 'LYS'}
        ANIONIC_RES = {'ASP', 'GLU'}
        NON_PROTEIN = {'HOH', 'WAT', 'TIP3', 'SOL', 'SPC', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN'}
        SALT_BRIDGE_CUTOFF = 5.0
        interactions = []
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
        AROMATIC_RES = {'PHE', 'TYR', 'TRP', 'HIS'}
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
        PI_STACK_CUTOFF = 6.0
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
        AROMATIC_RES = {'PHE', 'TYR', 'TRP', 'HIS'}
        CATIONIC_RES = {'ARG', 'LYS'}
        interactions = []
        aroms, cations = [], []
        for atom in structure.atom:
            res = atom.getResidue()
            chain = str(res.chain)
            if res.pdbres.strip() in AROMATIC_RES and chain in group1:
                aroms.append(res)
            elif res.pdbres.strip() in CATIONIC_RES and chain in group2:
                cations.append(res)
        PI_CATION_CUTOFF = 6.0
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

# --- Plotting Class ---
class PPIPlotter:
    """
    Plot protein-protein interaction networks with uniform, professional style.
    """
    def __init__(self, logger):
        self.logger = logger

    def _draw_ppi_network(self, G, node_chain, node_resname, node_resnum, edges_by_type, edge_labels, output_file, title, group1_nodes=None, group2_nodes=None, legend_handles=None):
        import matplotlib.lines as mlines
        plt.figure(figsize=(16, 12))
        ax = plt.gca()
        node_size = 2200
        # Layout: left/right with vertical spread
        if group1_nodes is None or group2_nodes is None:
            # Default: split by unique chains
            unique_chains = sorted(set(node_chain[n] for n in G.nodes()))
            if len(unique_chains) > 1:
                group1_chains = set([unique_chains[0]])
                group2_chains = set(unique_chains[1:])
            else:
                group1_chains = set([unique_chains[0]])
                group2_chains = set()
            group1_nodes = [n for n in G.nodes() if node_chain[n] in group1_chains]
            group2_nodes = [n for n in G.nodes() if node_chain[n] in group2_chains]
        pos = {}
        y_gap = 0.25
        for i, node in enumerate(group1_nodes):
            pos[node] = (-1, i * y_gap - (len(group1_nodes)-1)*y_gap/2)
        for i, node in enumerate(group2_nodes):
            pos[node] = (1, i * y_gap - (len(group2_nodes)-1)*y_gap/2)
        # Draw group outlines (convex hull) for each group
        for group_nodes, color in zip([group1_nodes, group2_nodes], [CHAIN_COLORS.get(node_chain[n], CHAIN_COLORS['default']) for n in [group1_nodes[0], group2_nodes[0]] if len(group1_nodes) > 0 and len(group2_nodes) > 0]):
            if len(group_nodes) >= 3:
                points = np.array([pos[n] for n in group_nodes])
                if len(np.unique(points[:, 0])) > 1 and len(np.unique(points[:, 1])) > 1:
                    hull = ConvexHull(points)
                    hull_pts = points[hull.vertices]
                    poly = mpatches.Polygon(hull_pts, closed=True, facecolor=color, alpha=0.15, edgecolor=color, linewidth=2, zorder=0)
                    ax.add_patch(poly)
        for node in G.nodes():
            x, y = pos[node]
            resname = node_resname[node]
            fill_color = AMINO_ACID_COLORS.get(resname, AMINO_ACID_COLORS['UNK'])
            border_color = AMINO_ACID_COLORS.get(resname, AMINO_ACID_COLORS['UNK'])
            circ = mpatches.Circle((x, y), radius=(node_size/20000), facecolor=fill_color, edgecolor=border_color, linewidth=3, zorder=2)
            ax.add_patch(circ)
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
        # Draw edges by type
        if legend_handles is None:
            legend_handles = []
        for interaction_type, style in INTERACTION_STYLES.items():
            edges = edges_by_type.get(interaction_type, [])
            if edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges, edge_color=style['color'], style=style['style'], width=style['width'], ax=ax
                )
                legend_handles.append(
                    mlines.Line2D([], [], color=style['color'], linestyle=style['style'], linewidth=style['width'], label=style['label'])
                )
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='purple', font_size=13, label_pos=0.5, ax=ax)
        plt.title(title, fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        if legend_handles:
            plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=13, frameon=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"✅ Network plot saved to {output_file}")

    def plot_network(self, interactions: List[Dict[str, Any]], output_file: str, title: str = "Protein-Protein Interaction Network"):
        if not interactions:
            self.logger.info(f"⚠️ No interactions found. Skipping plot: {output_file}")
            return
        G = nx.Graph()
        edge_labels = {}
        node_chain = {}
        node_resname = {}
        node_resnum = {}
        edges_by_type = {k: [] for k in INTERACTION_STYLES}
        for inter in interactions:
            if 'donor' in inter and 'acceptor' in inter:
                n1, n2 = inter['donor'], inter['acceptor']
                interaction_type = inter['type']
            elif 'res1' in inter and 'res2' in inter:
                n1, n2 = inter['res1'], inter['res2']
                interaction_type = inter['type']
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
            G.add_edge(n1, n2, type=interaction_type)
            edges_by_type.setdefault(interaction_type, []).append((n1, n2))
            if 'occupancy' in inter:
                percent = int(round(inter['occupancy'] * 100))
                edge_labels[(n1, n2)] = f"{percent}%"
        # Use the same layout and drawing as consolidated
        self._draw_ppi_network(G, node_chain, node_resname, node_resnum, edges_by_type, edge_labels, output_file, title)

    def plot_consolidated(self, persistent_by_type: Dict[str, List[Dict[str, Any]]], output_file: str):
        import matplotlib.lines as mlines
        G = nx.Graph()
        edge_labels = {}
        edge_styles = {}
        node_chain = {}
        node_resname = {}
        node_resnum = {}
        edges_by_type = {k: [] for k in INTERACTION_STYLES}
        for interaction_type, interactions in persistent_by_type.items():
            style = INTERACTION_STYLES.get(interaction_type, INTERACTION_STYLES['hydrogen_bond'])
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
                G.add_edge(n1, n2, type=interaction_type)
                edges_by_type.setdefault(interaction_type, []).append((n1, n2))
                edge_styles[(n1, n2)] = style
                if 'occupancy' in inter:
                    percent = int(round(inter['occupancy'] * 100))
                    edge_labels[(n1, n2)] = f"{percent}%"
        # Determine groups for left/right layout
        unique_chains = sorted(set(node_chain[n] for n in G.nodes()))
        if len(unique_chains) > 1:
            group1_chains = set([unique_chains[0]])
            group2_chains = set(unique_chains[1:])
        else:
            group1_chains = set([unique_chains[0]])
            group2_chains = set()
        group1_nodes = [n for n in G.nodes() if node_chain[n] in group1_chains]
        group2_nodes = [n for n in G.nodes() if node_chain[n] in group2_chains]
        legend_handles = []
        self._draw_ppi_network(G, node_chain, node_resname, node_resnum, edges_by_type, edge_labels, output_file, "Consolidated Protein-Protein Interaction Diagram", group1_nodes, group2_nodes, legend_handles)

# --- CLI and Main ---
def main():
    import sys
    from schrodinger.structure import StructureReader
    from schrodinger.application.desmond.packages import traj
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
    logger = logging.getLogger("PPIAnalysis")
    parser = argparse.ArgumentParser(description="Protein-protein PPI analysis and visualization (Schrödinger style)")
    parser.add_argument('--cms', required=True, type=str, help='Path to the Desmond .cms structure file')
    parser.add_argument('--traj', required=True, type=str, help='Path to the trajectory file or directory')
    parser.add_argument('--chain-groups', nargs='+', default=None, help='Chain groups to analyze (e.g., A,B C,D). If not specified, available chains will be listed.')
    parser.add_argument('--occupancy-threshold', type=float, default=0.2, help='Minimum occupancy threshold (default: 0.2)')
    parser.add_argument('--output-prefix', type=str, default='ppi_analysis', help='Prefix for output files (default: ppi_analysis)')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug/progress logging')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of trajectory frames to analyze (default: all)')
    args = parser.parse_args()
    debug = args.debug
    cms_file = Path(args.cms)
    trajectory_file = Path(args.traj)
    occupancy_threshold = args.occupancy_threshold
    output_prefix = args.output_prefix
    # Load structure
    with StructureReader(str(cms_file)) as reader:
        structure = next(reader)
    # Parse chain groups
    def parse_chain_groups(chain_groups_list):
        return [group.split(',') for group in chain_groups_list]
    if args.chain_groups is None:
        print("ℹ️  Please specify --chain-groups using the available chain IDs above.")
        sys.exit(1)
    chain_groups = parse_chain_groups(args.chain_groups)
    # Load trajectory
    trj_frames = list(traj.read_traj(str(trajectory_file)))
    if args.max_frames is not None:
        trj_frames = trj_frames[:args.max_frames]
    max_frames = len(trj_frames)
    analyzer = PPIAnalyzer(structure, chain_groups, logger)
    plotter = PPIPlotter(logger)
    persistent_by_type = {}
    INTERACTION_TYPES = [
        ('hydrogen_bond', 'analyze_hydrogen_bonds'),
        ('salt_bridge', 'analyze_salt_bridges'),
        ('pi_pi', 'analyze_pi_pi'),
        ('pi_cation', 'analyze_pi_cation'),
    ]
    for interaction_type, method_name in INTERACTION_TYPES:
        interaction_frames = defaultdict(set)
        interaction_details = {}
        for i in range(max_frames):
            frame = trj_frames[i]
            frame_structure = structure.copy()
            positions = frame.pos()
            for j, atom in enumerate(frame_structure.atom):
                if j < len(positions):
                    atom.xyz = positions[j]
            frame_interactions = []
            for idx, group1 in enumerate(chain_groups):
                for group2 in chain_groups[idx+1:]:
                    analyze_func = getattr(analyzer, method_name)
                    frame_interactions.extend(analyze_func(frame_structure, group1, group2))
            for interaction in frame_interactions:
                key = (interaction['donor'], interaction['acceptor']) if 'donor' in interaction else (interaction.get('res1', ''), interaction.get('res2', ''))
                interaction_frames[key].add(i)
                if key not in interaction_details:
                    interaction_details[key] = interaction
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
        plotter.plot_network(persistent_interactions, output_file=f"{output_prefix}_network_{interaction_type}.png", title=f"{interaction_type.replace('_', ' ').title()} Network")
    plotter.plot_consolidated(persistent_by_type, output_file=f"{output_prefix}_consolidated.png")
    logger.info("\n🎉 Analysis completed successfully!")

if __name__ == "__main__":
    main()