#!/usr/bin/env python3
"""
ppi_analysis.py
Professional script for protein-protein non-bonding interaction analysis and visualization.
Combines analysis and plotting in a single, reproducible, developer-friendly module.
"""

import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import logging

try:
    from schrodinger.utils import log as schrod_log
    from schrodinger.job import jobcontrol
    SCHRODINGER_AVAILABLE = True
except ImportError:
    SCHRODINGER_AVAILABLE = False
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

import numpy as np
import matplotlib
matplotlib.use('Agg')
# Set Helvetica as default font for all plots, with robust fallback
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import networkx as nx
from scipy.spatial import ConvexHull
from matplotlib.patches import FancyArrowPatch

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

# Extend INTERACTION_STYLES to include analyzer aliases
INTERACTION_STYLES.update({
    'salt-bridge': INTERACTION_STYLES.get('salt_bridge', {
        'color': '#2E8B57', 'style': 'solid', 'width': 3.0, 'label': 'Salt Bridge'}),
    'pi-pi': INTERACTION_STYLES.get('pi_pi', {
        'color': '#8A2BE2', 'style': 'dotted', 'width': 2.5, 'label': 'π-π Stacking'}),
    'pi-cat': INTERACTION_STYLES.get('pi_cation', {
        'color': '#FF4500', 'style': 'dashdot', 'width': 2.5, 'label': 'π-Cation'}),
})

# --- Plotting Class ---
class PPIPlotter:
    r"""
    Plot protein-protein interaction networks with uniform, professional style.
    Ensures consistent node size, font size, and layout for all interaction types.

    :param logger: Logger instance
    :type logger: logging.Logger
    """
    def __init__(self, logger):
        self.logger = logger
        # Default values (will be dynamically adjusted)
        self.node_size = 1400
        self.font_size = 10
        self.arc_radius = 0.8
        self.layout_seed = 42  # Fixed seed for reproducible layouts

    def _draw_ppi_network(self, G, node_chain, node_resname, node_resnum, edges_by_type, edge_labels, output_file, title, group1_nodes=None, group2_nodes=None, legend_handles=None):
        import matplotlib.lines as mlines
        n_nodes = len(G.nodes())
        # Dynamic adjustment based on node count
        if n_nodes <= 15:
            node_size, font_size, arc_radius = 2000, 12, 0.7
            fig_size = (12, 10)
        elif n_nodes <= 30:
            node_size, font_size, arc_radius = 1400, 10, 0.8
            fig_size = (14, 12)
        else:
            node_size, font_size, arc_radius = 1000, 8, 0.9
            fig_size = (16, 14)
        fig, ax = plt.subplots(figsize=fig_size)
        # Determine unique chains and group nodes by chain
        unique_chains = sorted(set(node_chain[n] for n in G.nodes()))
        chain_to_nodes = {chain: [n for n in G.nodes() if node_chain[n] == chain] for chain in unique_chains}
        pos = {}
        # Schrodinger-style layout: two chains as left/right "leaves"/arcs
        if len(unique_chains) == 2:
            chain1, chain2 = unique_chains[0], unique_chains[1]
            chain1_nodes = chain_to_nodes[chain1]
            chain2_nodes = chain_to_nodes[chain2]
            # Sort by residue number for each chain
            def resnum_key(n):
                try:
                    return int(node_resnum[n])
                except Exception:
                    return 0
            chain1_nodes.sort(key=resnum_key)
            chain2_nodes.sort(key=resnum_key)
            n1, n2 = len(chain1_nodes), len(chain2_nodes)
            for i, n in enumerate(chain1_nodes):
                angle = (i / max(1, n1-1)) * np.pi - np.pi/2  # from -90 to +90 deg
                x = -1.0 + arc_radius * np.cos(angle)
                y = arc_radius * np.sin(angle)
                pos[n] = (x, y)
            for i, n in enumerate(chain2_nodes):
                angle = (i / max(1, n2-1)) * np.pi - np.pi/2
                x = 1.0 + arc_radius * np.cos(angle)
                y = arc_radius * np.sin(angle)
                pos[n] = (x, y)
        elif len(unique_chains) == 1:
            # Single chain: circular layout
            nodes = list(G.nodes())
            n = len(nodes)
            for i, n_id in enumerate(nodes):
                angle = (i / n) * 2 * np.pi
                r = 1.0
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                pos[n_id] = (x, y)
        else:
            # More than two chains: arrange each as a leaf/arc around a circle
            n_chains = len(unique_chains)
            for idx, chain in enumerate(unique_chains):
                nodes = chain_to_nodes[chain]
                n = len(nodes)
                center_angle = (idx / n_chains) * 2 * np.pi
                center_x = 1.2 * np.cos(center_angle)
                center_y = 1.2 * np.sin(center_angle)
                for i, n_id in enumerate(nodes):
                    angle = (i / max(1, n-1)) * np.pi - np.pi/2
                    r = 0.5
                    x = center_x + r * np.cos(angle)
                    y = center_y + r * np.sin(angle)
                    pos[n_id] = (x, y)
        # Draw nodes as large circles with AA color and black border
        for n in G.nodes():
            x, y = pos[n]
            resname = node_resname[n]
            fill_color = AMINO_ACID_COLORS.get(resname, AMINO_ACID_COLORS['UNK'])
            circ = mpatches.Circle((x, y), radius=(node_size/20000), facecolor=fill_color, edgecolor='black', linewidth=2.5, zorder=2)
            ax.add_patch(circ)
        # Draw node labels centered, with white outline
        for n in G.nodes():
            x, y = pos[n]
            chain = node_chain.get(n, '')
            aa = node_resname.get(n, 'UNK')
            num = node_resnum.get(n, '')
            label_text = f"{chain}\n{aa}\n{num}"
            text = ax.text(x, y, label_text, fontsize=font_size, ha='center', va='center', color='black', zorder=4, linespacing=1.1, fontname='Helvetica')
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal()
            ])
        # Draw edges by interaction type, with legend
        legend_handles = []
        for interaction_type, style in INTERACTION_STYLES.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == interaction_type]
            if edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges, edge_color=style['color'], style=style['style'], width=style['width'], ax=ax
                )
                legend_handles.append(
                    mlines.Line2D([], [], color=style['color'], linestyle=style['style'], linewidth=style['width'], label=style['label'])
                )
        # Draw occupancy labels at edge midpoints, offset to avoid overlap
        for (n1, n2), label in edge_labels.items():
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            # Offset perpendicular to edge
            dx, dy = x2 - x1, y2 - y1
            norm = np.sqrt(dx**2 + dy**2)
            if norm == 0:
                norm = 1
            offset = 0.08
            ox = -offset * dy / norm
            oy = offset * dx / norm
            label_x = mx + ox
            label_y = my + oy
            ax.text(label_x, label_y, label, color='purple', fontsize=font_size-1, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'), fontname='Helvetica')
        # Add professional legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=font_size, frameon=True, prop={'family': 'Helvetica'})
        plt.title(title, fontsize=font_size+8, fontname='Helvetica')
        ax.set_aspect('equal')
        plt.axis('off')
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Add extra bottom/top margin
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"✅ Network plot saved to {output_file}")

    def plot_network(self, interactions: List[Dict[str, Any]], output_file: str, title: str = "Protein-Protein Interaction Network"):
        r"""
        Plot a single-type PPI network.

        :param interactions: List of interaction dicts
        :type interactions: List[Dict[str, Any]]
        :param output_file: Output file path
        :type output_file: str
        :param title: Plot title
        :type title: str
        :return: None
        :rtype: None
        """
        MAX_PER_TYPE_EDGES = 30
        if len(interactions) > MAX_PER_TYPE_EDGES:
            self.logger.info(f"Too many interactions ({len(interactions)}) for {title}; skipping plot for clarity.")
            return
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
        r"""
        Plot a consolidated PPI network for all interaction types.

        :param persistent_by_type: Dict of interaction type to list of interactions
        :type persistent_by_type: Dict[str, List[Dict[str, Any]]]
        :param output_file: Output file path
        :type output_file: str
        :return: None
        :rtype: None
        """
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
        # Layout: left/right with vertical spread
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
                    plt.gca().add_patch(poly)
        # Draw nodes as perfect circles with AA color fill and AA color border
        node_size = self.node_size
        for node in G.nodes():
            x, y = pos[node]
            resname = node_resname[node]
            fill_color = AMINO_ACID_COLORS.get(resname, AMINO_ACID_COLORS['UNK'])
            border_color = AMINO_ACID_COLORS.get(resname, AMINO_ACID_COLORS['UNK'])
            circ = mpatches.Circle((x, y), radius=(node_size/20000), facecolor=fill_color, edgecolor=border_color, linewidth=3, zorder=2)
            plt.gca().add_patch(circ)
        # Custom node labels: all centered and inside the node, with white outline for contrast
        for node in G.nodes():
            x, y = pos[node]
            chain = node_chain[node]
            aa = node_resname[node]
            num = node_resnum[node]
            label_text = f"{chain}\n{aa}\n{num}"
            text = plt.gca().text(x, y, label_text, fontsize=12, ha='center', va='center', color='black', zorder=4, linespacing=1.1, fontname='Helvetica')
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal()
            ])
        # Draw edges by interaction type
        legend_handles = []
        for interaction_type, style in INTERACTION_STYLES.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == interaction_type]
            if edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges, edge_color=style['color'], style=style['style'], width=style['width'], ax=plt.gca()
                )
                legend_handles.append(
                    mlines.Line2D([], [], color=style['color'], linestyle=style['style'], linewidth=style['width'], label=style['label'])
                )
        # Draw occupancy labels
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='purple', font_size=13, label_pos=0.5, ax=plt.gca())
        plt.title("Consolidated Protein-Protein Interaction Diagram", fontsize=18, fontname='Helvetica')
        plt.axis('off')
        plt.tight_layout()
        if legend_handles:
            plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=13, frameon=True, prop={'family': 'Helvetica'})
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"✅ Network plot saved to {output_file}")

# --- CLI and Main ---
def submit_job(args, host, jobname):
    r"""
    Submit this script as a Schrodinger job to the specified host (CLI/batch-safe).

    :param args: Parsed CLI arguments
    :type args: argparse.Namespace
    :param host: Host string for job submission
    :type host: str
    :param jobname: Job name for submission
    :type jobname: str
    :return: Job object
    :rtype: schrodinger.job.jobcontrol.Job
    """
    schrodinger_run = os.path.join(os.environ.get("SCHRODINGER", "/opt/schrodinger"), "run")
    cmd = [
        schrodinger_run, sys.executable, os.path.abspath(__file__),
        "--cms", args.cms,
        "--traj", args.traj,
        "--occupancy-threshold", str(args.occupancy_threshold),
        "--output-prefix", args.output_prefix,
        "--max-frames", str(args.max_frames) if args.max_frames is not None else "0"
    ]
    if args.chain_groups:
        cmd += ["--chain-groups"] + args.chain_groups
    if args.debug:
        cmd.append("--debug")
    # Add host and jobname as command-line arguments
    cmd += ["-HOST", host, "-JOBNAME", jobname]
    job = jobcontrol.launch_job(cmd)
    return job

def test_schrodinger_ppi_analyzer(cms_file, traj_file, logger=None):
    r"""
    Run Schrodinger's ProtProtInter analyzer and print a summary of results.

    :param cms_file: Path to the .cms structure file
    :type cms_file: str
    :param traj_file: Path to the trajectory file or directory
    :type traj_file: str
    :param logger: Logger for output (optional)
    :type logger: logging.Logger or None
    :return: None
    :rtype: None
    """
    try:
        from schrodinger.application.desmond.packages import analysis, topo, traj
    except ImportError:
        print("[ERROR] Schrodinger modules not available. Cannot run analyzer test.")
        return
    import collections
    cms_file = str(cms_file)
    traj_file = str(traj_file)
    print("\n[INFO] Running Schrodinger ProtProtInter analyzer test...")
    msys_model, cms_model = topo.read_cms(cms_file)
    tr = traj.read_traj(traj_file)
    analyzer = analysis.ProtProtInter(msys_model, cms_model, "protein")
    results = analysis.analyze(tr, analyzer)
    print("[DEBUG] Raw analyzer results:", results)
    summary = {}
    if isinstance(results, dict):
        for interaction_type, pairs in results.items():
            summary[interaction_type] = len(pairs)
    else:
        print(f"[ERROR] Unexpected analyzer result type: {type(results)}")
    print("\nInteraction summary (total across all frames):")
    for interaction_type, count in summary.items():
        print(f"  {interaction_type}: {count} interactions")
        # Print a few example pairs
        pairs = results.get(interaction_type, {})
        for i, (pair, frame_count) in enumerate(pairs.items()):
            if i < 3:
                print(f"    {pair}: {frame_count} frames")
        if len(pairs) > 3:
            print("    ...")
    if logger:
        logger.info(f"Schrodinger analyzer summary: {summary}")
    print("[INFO] Schrodinger analyzer test complete.\n")

def run_ppi_analysis_with_schrodinger(cms_file, traj_file, logger=None):
    r"""
    Run Schrodinger's ProtProtInter analyzer and return the summary dictionary.

    :param cms_file: Path to the .cms structure file
    :type cms_file: str
    :param traj_file: Path to the trajectory file or directory
    :type traj_file: str
    :param logger: Logger for output (optional)
    :type logger: logging.Logger or None
    :return: Dictionary mapping interaction types to residue pair/frame count dicts
    :rtype: dict
    """
    try:
        from schrodinger.application.desmond.packages import analysis, topo, traj
    except ImportError:
        msg = "[ERROR] Schrodinger modules not available. Cannot run analyzer."
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None
    cms_file = str(cms_file)
    traj_file = str(traj_file)
    msys_model, cms_model = topo.read_cms(cms_file)
    tr = traj.read_traj(traj_file)
    analyzer = analysis.ProtProtInter(msys_model, cms_model, "protein")
    results = analysis.analyze(tr, analyzer)
    if not isinstance(results, dict):
        msg = f"[ERROR] Unexpected analyzer result type: {type(results)}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None
    return results

def main():
    r"""
    Main function for protein-protein PPI analysis and visualization (Schrödinger style).
    Supports Schrodinger job control and robust logging.
    Outputs all results, logs, and raw data to a dedicated output directory.

    :return: None
    :rtype: None
    """
    from schrodinger.structure import StructureReader
    from schrodinger.application.desmond.packages import traj
    import json
    # Configurable threshold for consolidated plot
    MAX_CONSOLIDATED_EDGES = 30
    parser = argparse.ArgumentParser(description="Protein-protein PPI analysis and visualization (Schrödinger style)")
    parser.add_argument('--cms', required=True, type=str, help='Path to the Desmond .cms structure file')
    parser.add_argument('--traj', required=True, type=str, help='Path to the trajectory file or directory')
    parser.add_argument('--chain-groups', nargs='+', default=None, help='Chain groups to analyze (e.g., A,B C,D). If not specified, available chains will be listed.')
    parser.add_argument('--occupancy-threshold', type=float, default=0.2, help='Minimum occupancy threshold (default: 0.2)')
    parser.add_argument('--output-prefix', type=str, default='ppi_analysis', help='Prefix for output files (default: ppi_analysis)')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug/progress logging')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of trajectory frames to analyze (default: all)')
    parser.add_argument('--submit-job', action='store_true', help='Submit this script as a Schrodinger job and exit')
    parser.add_argument('--host', type=str, default='localhost', help='Host for job submission (default: localhost)')
    parser.add_argument('--jobname', type=str, default='ppi_analysis', help='Job name for submission (default: ppi_analysis)')
    parser.add_argument('--test-schrodinger-analyzer', action='store_true', help='Run a test of Schrodinger ProtProtInter analyzer and print summary')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger: Schrodinger logger + file handler
    if SCHRODINGER_AVAILABLE:
        logger = schrod_log.get_output_logger("PPIAnalysis")
    else:
        logger = logging.getLogger("PPIAnalysis")
    log_file = output_dir / "ppi_analysis.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    if args.submit_job:
        if not SCHRODINGER_AVAILABLE:
            print("❌ Schrodinger job control is not available in this environment.")
            sys.exit(1)
        logger.info(f"Submitting job to host: {args.host} with jobname: {args.jobname}")
        job = submit_job(args, args.host, args.jobname)
        logger.info(f"Job submitted: {job}")
        print(f"Job submitted to host: {args.host} with jobname: {args.jobname}")
        sys.exit(0)

    if getattr(args, 'test_schrodinger_analyzer', False):
        test_schrodinger_ppi_analyzer(args.cms, args.traj)
        sys.exit(0)

    # Parse chain groups
    def parse_chain_groups(chain_groups_list):
        return [set(group.split(',')) for group in chain_groups_list]
    if args.chain_groups is None or len(args.chain_groups) < 2:
        logger.error("You must specify at least two chain groups for inter-chain analysis.")
        sys.exit(1)
    chain_groups = parse_chain_groups(args.chain_groups)
    group1, group2 = chain_groups[0], chain_groups[1]
    def get_chain(res_id):
        # res_id is like 'A:GLU_146'
        return res_id.split(':')[0]
    logger.info("Using Schrodinger ProtProtInter analyzer for PPI analysis.")
    ppi_summary = run_ppi_analysis_with_schrodinger(args.cms, args.traj, logger)
    if ppi_summary is None:
        logger.error("Schrodinger analyzer failed. Exiting.")
        sys.exit(1)
    # Build persistent_by_type as before
    persistent_by_type = {}
    total_frames = None
    for pairs in ppi_summary.values():
        for count in pairs.values():
            if total_frames is None or count > total_frames:
                total_frames = count
    if total_frames is None:
        total_frames = 1
    for interaction_type, pairs in ppi_summary.items():
        interactions = []
        for (res1, res2), count in pairs.items():
            chain1 = get_chain(res1)
            chain2 = get_chain(res2)
            if ((chain1 in group1 and chain2 in group2) or (chain2 in group1 and chain1 in group2)):
                occupancy = count / total_frames
                interactions.append({
                    'res1': res1,
                    'res2': res2,
                    'occupancy': occupancy,
                    'type': interaction_type
                })
        filtered_interactions = [inter for inter in interactions if inter['occupancy'] >= args.occupancy_threshold]
        persistent_by_type[interaction_type] = filtered_interactions
    # Now group H-bond subtypes (excluding hbond_self)
    HBOND_SUBTYPES = {'hbond_bb', 'hbond_ss', 'hbond_sb', 'hbond_bs'}
    grouped_persistent_by_type = {}
    for interaction_type, interactions in persistent_by_type.items():
        if interaction_type in HBOND_SUBTYPES:
            grouped_persistent_by_type.setdefault('hydrogen_bond', []).extend(interactions)
        elif interaction_type == 'hbond_self':
            continue
        else:
            grouped_persistent_by_type[interaction_type] = interactions
    # Plot grouped per-type networks (skip H-bond subtypes)
    plotter = PPIPlotter(logger)
    for interaction_type, interactions in grouped_persistent_by_type.items():
        if interaction_type != 'hydrogen_bond' and interaction_type not in INTERACTION_STYLES:
            logger.warning(f"Skipping unknown interaction type: {interaction_type}")
            continue
        plotter.plot_network(
            interactions,
            output_file=str(output_dir / f"{args.output_prefix}_interactions_{interaction_type}.png"),
            title=f"{interaction_type.replace('_', ' ').replace('-', ' ').title()} Interactions"
        )
    # Plot consolidated network only if number of unique pairs is within threshold
    total_pairs = sum(len(interactions) for interactions in grouped_persistent_by_type.values())
    if total_pairs <= MAX_CONSOLIDATED_EDGES:
        plotter.plot_consolidated(grouped_persistent_by_type, output_file=str(output_dir / f"{args.output_prefix}_consolidated.png"))
    else:
        logger.info(f"Too many interactions ({total_pairs}); skipping consolidated plot for clarity.")
    # Save persistent interaction data as JSON
    with open(output_dir / "persistent_interactions.json", "w") as f:
        json.dump(grouped_persistent_by_type, f, indent=2)
    logger.info("\n🎉 Analysis completed successfully! All results are in: %s", output_dir)
    return

if __name__ == "__main__":
    main()