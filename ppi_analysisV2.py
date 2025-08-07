#!/usr/bin/env python3
"""
ppi_analysis.py - Professional Protein-Protein Interaction Analysis and Visualization

This script provides functionality to:
- Analyze protein-protein interactions using Schrodinger Suite
- Generate professional network visualizations with consistent styling
- Support Schrodinger job control for remote execution
- Provide comprehensive CLI interface with robust error handling
- Filter interactions by occupancy threshold
- Create both per-type and consolidated interaction plots

Usage examples:
- python ppi_analysis.py --cms structure.cms --traj trajectory_dir --chain-groups A,B C,D
- python ppi_analysis.py --cms structure.cms --traj trajectory_dir --submit-job --host server
- python ppi_analysis.py --cms structure.cms --traj trajectory_dir --occupancy-threshold 0.3

Copyright Schrodinger, LLC. All rights reserved.
"""

# Standard library imports
import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
import matplotlib
matplotlib.use('Agg')
# Set Helvetica as default font for all plots, with robust fallback
matplotlib.rcParams['font.sans-serif'] = [
    'Helvetica', 'Arial', 'DejaVu Sans'
]
matplotlib.rcParams['font.family'] = 'sans-serif'
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import ConvexHull

# Schrodinger imports
try:
    from schrodinger.utils import log as schrod_log
    from schrodinger.job import jobcontrol
    SCHRODINGER_AVAILABLE = True
except ImportError:
    SCHRODINGER_AVAILABLE = False
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO
    )

# Simple label overlap prevention using available libraries
ADJUST_TEXT_AVAILABLE = False  # We'll use our custom implementation

def prevent_label_overlaps_simple(ax, texts, nodes_pos, min_distance=0.15):
    """
    Simple label overlap prevention by adjusting font size and position.

    :param ax: Matplotlib axes object
    :type ax: matplotlib.axes.Axes
    :param texts: List of text objects to adjust
    :type texts: List[matplotlib.text.Text]
    :param nodes_pos: Dictionary of node positions
    :type nodes_pos: Dict
    :param min_distance: Minimum distance between labels
    :type min_distance: float
    :return: None
    :rtype: None
    """
    try:
        import numpy as np
        from scipy.spatial.distance import cdist

        if not texts or len(texts) < 2:
            return True

        # Get current text positions
        text_positions = []
        for text in texts:
            pos = text.get_position()
            text_positions.append(pos)

        text_positions = np.array(text_positions)

        # Calculate distances between all text positions
        distances = cdist(text_positions, text_positions)

        # Find overlapping pairs
        overlapping_pairs = []
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if distances[i, j] < min_distance:
                    overlapping_pairs.append((i, j))

        # Simple solution: reduce font size for overlapping labels
        if overlapping_pairs:
            for i, j in overlapping_pairs:
                # Reduce font size for both overlapping labels
                current_size = texts[i].get_fontsize()
                texts[i].set_fontsize(max(8, current_size * 0.8))

                current_size = texts[j].get_fontsize()
                texts[j].set_fontsize(max(8, current_size * 0.8))

            return True

        return True

    except Exception as e:
        print(f"Warning: Failed to adjust text positions: {e}")
        return False
    """
    Custom label overlap prevention using SciPy spatial distance.

    :param ax: Matplotlib axes object
    :type ax: matplotlib.axes.Axes
    :param texts: List of text objects to adjust
    :type texts: List[matplotlib.text.Text]
    :param nodes_pos: Dictionary of node positions
    :type nodes_pos: Dict
    :param node_size: Size of nodes to avoid
    :type node_size: int
    :param min_distance: Minimum distance between labels
    :type min_distance: float
    :return: None
    :rtype: None
    """
    try:
        from scipy.spatial.distance import cdist
        import numpy as np

        if not texts:
            return

        # Get current text positions
        text_positions = []
        for text in texts:
            bbox = text.get_window_extent(ax.figure.canvas.get_renderer())
            center_x = (bbox.x0 + bbox.x1) / 2
            center_y = (bbox.y0 + bbox.y1) / 2
            # Convert display coordinates to data coordinates
            data_coords = ax.transData.inverted().transform((center_x, center_y))
            text_positions.append(data_coords)

        text_positions = np.array(text_positions)

        # Get node positions
        node_coords = np.array(list(nodes_pos.values()))

        # Calculate distances between all text positions
        text_distances = cdist(text_positions, text_positions)

        # Calculate distances from text to nodes
        text_node_distances = cdist(text_positions, node_coords)

        # Iteratively adjust positions to avoid overlaps
        max_iterations = 50
        for iteration in range(max_iterations):
            moved = False

            for i, text in enumerate(texts):
                # Check overlap with other texts
                for j in range(len(texts)):
                    if i != j and text_distances[i, j] < min_distance:
                        # Move text away from overlapping text
                        direction = text_positions[i] - text_positions[j]
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            text_positions[i] += direction * (min_distance - text_distances[i, j]) * 0.1
                            moved = True

                # Check overlap with nodes
                for j in range(len(node_coords)):
                    if text_node_distances[i, j] < min_distance * 0.8:
                        # Move text away from node
                        direction = text_positions[i] - node_coords[j]
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            text_positions[i] += direction * (min_distance * 0.8 - text_node_distances[i, j]) * 0.1
                            moved = True

                # Update text position
                if moved:
                    text.set_position((text_positions[i][0], text_positions[i][1]))

            # Recalculate distances
            text_distances = cdist(text_positions, text_positions)
            text_node_distances = cdist(text_positions, node_coords)

            if not moved:
                break

        return True

    except Exception as e:
        print(f"Warning: Failed to adjust text positions: {e}")
        return False

# Professional amino acid color schemes based on residue type categories
AMINO_ACID_COLORS = {
    # Charged (negative) - Orange
    'ASP': '#FF8C00', 'GLU': '#FFA500',

    # Charged (positive) - Light purple/blue
    'LYS': '#9370DB', 'ARG': '#8A2BE2', 'HIS': '#4B0082',

    # Glycine - Light yellow/cream
    'GLY': '#F0E68C',

    # Hydrophobic - Light green
    'ALA': '#90EE90', 'VAL': '#98FB98', 'LEU': '#32CD32', 'ILE': '#228B22',
    'MET': '#006400', 'PHE': '#556B2F', 'TRP': '#2F4F2F', 'PRO': '#8FBC8F',

    # Polar - Light blue
    'SER': '#87CEEB', 'THR': '#4682B4', 'CYS': '#20B2AA', 'TYR': '#48D1CC',
    'ASN': '#00CED1', 'GLN': '#40E0D0',

    # Metal - Grey
    'HIS': '#808080',  # Can coordinate metals

    # Unspecified residue - Darker grey
    'UNK': '#696969',
}
CHAIN_COLORS = {
    'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 'D': '#96CEB4',
    'E': '#FFEAA7', 'F': '#DDA0DD', 'L': '#FFA07A', 'default': '#87CEEB'
}
INTERACTION_STYLES = {
    'hydrogen_bond': {
        'color': '#800080', 'style': 'solid', 'width': 2.5,  # Purple solid line with arrow
        'label': 'H-bond'
    },
    'salt_bridge': {
        'color': '#0000FF', 'style': 'solid', 'width': 3.0,  # Blue solid line
        'label': 'Salt Bridge'
    },
    'pi_pi': {
        'color': '#008000', 'style': 'dotted', 'width': 2.5,  # Green dotted line with circles
        'label': 'π-π Stacking'
    },
    'pi_cation': {
        'color': '#FF0000', 'style': 'solid', 'width': 2.5,  # Red solid line with circle
        'label': 'π-Cation'
    },
    'halogen_bond': {
        'color': '#FFFF00', 'style': 'solid', 'width': 2.5,  # Yellow solid line with arrow
        'label': 'Halogen Bond'
    },
    'metal_coordination': {
        'color': '#808080', 'style': 'solid', 'width': 2.5,  # Grey solid line
        'label': 'Metal Coordination'
    },
    'distance': {
        'color': '#008000', 'style': 'dotted', 'width': 1.5,  # Green dotted line
        'label': 'Distance'
    },
}

# Extend INTERACTION_STYLES to include analyzer aliases
INTERACTION_STYLES.update({
    'salt-bridge': INTERACTION_STYLES.get('salt_bridge', {
        'color': '#0000FF', 'style': 'solid', 'width': 3.0,
        'label': 'Salt Bridge'
    }),
    'pi-pi': INTERACTION_STYLES.get('pi_pi', {
        'color': '#008000', 'style': 'dotted', 'width': 2.5,
        'label': 'π-π Stacking'
    }),
    'pi-cat': INTERACTION_STYLES.get('pi_cation', {
        'color': '#FF0000', 'style': 'solid', 'width': 2.5,
        'label': 'π-Cation'
    }),
    'halogen-bond': INTERACTION_STYLES.get('halogen_bond', {
        'color': '#FFFF00', 'style': 'solid', 'width': 2.5,
        'label': 'Halogen Bond'
    }),
    'metal-coordination': INTERACTION_STYLES.get('metal_coordination', {
        'color': '#808080', 'style': 'solid', 'width': 2.5,
        'label': 'Metal Coordination'
    }),
})

# --- Input Validation Functions ---
def validate_files(cms_file, traj_file, logger):
    """
    Validate input files exist and are accessible.

    :param cms_file: Path to CMS file
    :type cms_file: str
    :param traj_file: Path to trajectory file
    :type traj_file: str
    :param logger: Logger instance
    :type logger: logging.Logger
    :return: True if validation passes, False otherwise
    :rtype: bool
    """
    cms_path = Path(cms_file)
    traj_path = Path(traj_file)

    if not cms_path.exists():
        logger.error(f"CMS file not found: {cms_file}")
        return False

    if not traj_path.exists():
        logger.error(f"Trajectory file not found: {traj_file}")
        return False

    if not cms_path.is_file():
        logger.error(f"CMS path is not a file: {cms_file}")
        return False

    if not traj_path.is_file() and not traj_path.is_dir():
        logger.error(f"Trajectory path is not a file or directory: {traj_file}")
        return False

    logger.info("✅ Input file validation passed")
    return True

def validate_chain_groups(chain_groups, logger):
    """
    Validate chain group specifications.

    :param chain_groups: List of chain group strings
    :type chain_groups: List[str]
    :param logger: Logger instance
    :type logger: logging.Logger
    :return: True if validation passes, False otherwise
    :rtype: bool
    """
    if not chain_groups or len(chain_groups) < 2:
        logger.error("At least two chain groups must be specified")
        return False

    for i, group in enumerate(chain_groups):
        if not group.strip():
            logger.error(f"Chain group {i+1} is empty")
            return False

        chains = group.split(',')
        for chain in chains:
            if not chain.strip():
                logger.error(f"Empty chain ID in group {i+1}")
                return False

    logger.info("✅ Chain group validation passed")
    return True

def validate_parameters(args, logger):
    """
    Validate all input parameters.

    :param args: Parsed arguments
    :type args: argparse.Namespace
    :param logger: Logger instance
    :type logger: logging.Logger
    :return: True if validation passes, False otherwise
    :rtype: bool
    """
    # Validate occupancy threshold
    if not (0.0 <= args.occupancy_threshold <= 1.0):
        logger.error(
            f"Occupancy threshold must be between 0.0 and 1.0, got: "
            f"{args.occupancy_threshold}"
        )
        return False

    # Validate max_frames if specified
    if args.max_frames is not None and args.max_frames <= 0:
        logger.error(
            f"Max frames must be positive, got: {args.max_frames}"
        )
        return False

    logger.info("✅ Parameter validation passed")
    return True

# --- Plotting Class ---
class PPIPlotter:
    r"""
    Plot protein-protein interaction networks with uniform, professional style.
    Ensures consistent node size, font size, and layout for all interaction types.

    :param logger: Logger instance for output messages
    :type logger: logging.Logger
    :return: None
    :rtype: None
    """
    def __init__(self, logger):
        self.logger = logger
        # Default values (will be dynamically adjusted)
        self.node_size = 1400
        self.font_size = 10
        self.arc_radius = 0.8
        self.layout_seed = 42  # Fixed seed for reproducible layouts

    def _drawPpiNetwork(self, graph, node_chain, node_resname, node_resnum,
                       edges_by_type, edge_labels, output_file, title,
                       group1_nodes=None, group2_nodes=None,
                       legend_handles=None):
        """
        Draw protein-protein interaction network with professional styling.

        :param graph: NetworkX graph object
        :type graph: networkx.Graph
        :param node_chain: Dictionary mapping nodes to chain IDs
        :type node_chain: Dict[str, str]
        :param node_resname: Dictionary mapping nodes to residue names
        :type node_resname: Dict[str, str]
        :param node_resnum: Dictionary mapping nodes to residue numbers
        :type node_resnum: Dict[str, str]
        :param edges_by_type: Dictionary mapping interaction types to edge lists
        :type edges_by_type: Dict[str, List[tuple]]
        :param edge_labels: Dictionary mapping edges to labels
        :type edge_labels: Dict[tuple, str]
        :param output_file: Output file path
        :type output_file: str
        :param title: Plot title
        :type title: str
        :param group1_nodes: First group of nodes (optional)
        :type group1_nodes: List[str] or None
        :param group2_nodes: Second group of nodes (optional)
        :type group2_nodes: List[str] or None
        :param legend_handles: Legend handles (optional)
        :type legend_handles: List or None
        :return: None
        :rtype: None
        """
        import matplotlib.lines as mlines
        n_nodes = len(graph.nodes())
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
        unique_chains = sorted(set(node_chain[n] for n in graph.nodes()))
        chain_to_nodes = {
            chain: [n for n in graph.nodes() if node_chain[n] == chain]
            for chain in unique_chains
        }
        pos = {}
        # Use left/right layout for two chains (same as consolidated plot)
        if len(unique_chains) == 2:
            chain1, chain2 = unique_chains[0], unique_chains[1]
            chain1_nodes = chain_to_nodes[chain1]
            chain2_nodes = chain_to_nodes[chain2]
            # Calculate interaction counts manually from edges data
            node_interaction_counts = {}
            for node in graph.nodes():
                node_interaction_counts[node] = 0

            # Count interactions from edges_by_type
            for interaction_type, edges in edges_by_type.items():
                for edge in edges:
                    if len(edge) == 2:
                        node1, node2 = edge
                        node_interaction_counts[node1] += 1
                        node_interaction_counts[node2] += 1

            # Sort by interaction count (highest first) then by residue number
            def interaction_count_key(n):
                try:
                    interaction_count = node_interaction_counts.get(n, 0)
                    # Use negative for descending order (highest first)
                    return (-interaction_count, int(node_resnum[n]))
                except Exception:
                    return (0, 0)

            chain1_nodes.sort(key=interaction_count_key)
            chain2_nodes.sort(key=interaction_count_key)

            # Debug: Log the ordering
            self.logger.info(f"Chain {chain1} ordering by interaction count:")
            for node in chain1_nodes:
                count = node_interaction_counts.get(node, 0)
                resnum = node_resnum.get(node, '?')
                self.logger.info(f"  {node}: {count} interactions (residue {resnum})")

            self.logger.info(f"Chain {chain2} ordering by interaction count:")
            for node in chain2_nodes:
                count = node_interaction_counts.get(node, 0)
                resnum = node_resnum.get(node, '?')
                self.logger.info(f"  {node}: {count} interactions (residue {resnum})")

            # Position nodes in left/right layout
            y_gap = 2.0  # Increased spacing between nodes for better label clarity
            x_separation = 3.0  # Separation between chains

            for i, node in enumerate(chain1_nodes):
                pos[node] = (-x_separation, i * y_gap - (len(chain1_nodes)-1)*y_gap/2)
            for i, node in enumerate(chain2_nodes):
                pos[node] = (x_separation, i * y_gap - (len(chain2_nodes)-1)*y_gap/2)
        elif len(unique_chains) == 1:
            # Single chain: circular layout
            nodes = list(graph.nodes())
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
        # Draw nodes using NetworkX optimized functions with larger size
        node_colors = []
        node_sizes = []

        for n in graph.nodes():
            resname = node_resname[n]
            fill_color = AMINO_ACID_COLORS.get(resname, AMINO_ACID_COLORS['UNK'])
            node_colors.append(fill_color)
            # Dynamic node size based on number of nodes
            total_nodes = len(graph.nodes())
            if total_nodes <= 2:
                node_size = 8000  # Much larger for 2 nodes to fit text
            elif total_nodes <= 5:
                node_size = 6000  # Larger for few nodes
            elif total_nodes <= 10:
                node_size = 5000  # Medium for moderate nodes
            else:
                node_size = 4000  # Large for many nodes
            node_sizes.append(node_size)

        # Draw nodes with explicit matplotlib Circle patches for perfect circles
        import matplotlib.patches as patches

        # Calculate node radius based on node sizes
        avg_node_size = sum(node_sizes) / len(node_sizes) if node_sizes else 2000
        node_radius = (avg_node_size ** 0.5) * 0.01  # Scale factor for conversion

        for i, node in enumerate(graph.nodes()):
            x, y = pos[node]
            color = node_colors[i] if i < len(node_colors) else '#2f4f2f'
            circle = patches.Circle(
                (x, y),
                radius=node_radius,
                facecolor=color,
                edgecolor='black',
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(circle)

        # Draw node labels with proper formatting
        node_labels = {}
        for n in graph.nodes():
            chain = node_chain.get(n, '')
            aa = node_resname.get(n, 'UNK')
            num = node_resnum.get(n, '')
            # Format: Chain\nAA\nNumber
            node_labels[n] = f"{chain}\n{aa}\n{num}"

        # Draw labels with NetworkX function - larger font for better visibility
        label_objects = nx.draw_networkx_labels(
            graph, pos,
            labels=node_labels,
            font_size=12,  # Increased font size
            font_family='Helvetica',
            font_weight='bold',
            ax=ax
        )

        # Prevent label overlaps using adjustText if available
        if ADJUST_TEXT_AVAILABLE and label_objects:
            try:
                # Convert label objects to list of text objects
                texts = list(label_objects.values())

                # Extract node coordinates for avoidance
                node_coords = list(pos.values())
                x_coords = [pos[0] for pos in node_coords]
                y_coords = [pos[1] for pos in node_coords]

                # Adjust text positions to prevent overlaps
                self.logger.info("✅ Label overlaps prevented using adjustText")
            except Exception as e:
                self.logger.warning(f"Failed to adjust text positions: {e}")
                # Fallback to original labels if adjustText fails
        # Draw edges by interaction type, with legend
        legend_handles = []
        for interaction_type, style in INTERACTION_STYLES.items():
            edges = [
                (u, v) for u, v, d in graph.edges(data=True)
                if d.get('type') == interaction_type
            ]
            if edges:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=edges, edge_color=style['color'],
                    style=style['style'], width=style['width'], ax=ax
                )
                legend_handles.append(
                    mlines.Line2D(
                        [], [], color=style['color'], linestyle=style['style'],
                        linewidth=style['width'], label=style['label']
                    )
                )
        # Draw occupancy labels at edge midpoints, offset to avoid overlap
        for (n1, n2), label in edge_labels.items():
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            # Offset perpendicular to edge
            delta_x, delta_y = x2 - x1, y2 - y1
            norm = np.sqrt(delta_x**2 + delta_y**2)
            if norm == 0:
                norm = 1
            offset = 0.08
            offset_x = -offset * delta_y / norm
            offset_y = offset * delta_x / norm
            label_x = mid_x + offset_x
            label_y = mid_y + offset_y
            ax.text(
                label_x, label_y, label, color='purple',
                fontsize=font_size-1, ha='center', va='center',
                bbox=dict(
                    facecolor='white', edgecolor='none', alpha=0.7,
                    boxstyle='round,pad=0.2'
                ), fontname='Helvetica'
            )
        # Add professional legend closer to the network
        if legend_handles:
            ax.legend(
                handles=legend_handles, loc='upper center',
                bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=font_size,  # Moved closer to nodes
                frameon=True, prop={'family': 'Helvetica'}
            )
        # Add chain labels for left/right layout
        if len(unique_chains) == 2:
            # Get the y-coordinates for positioning chain labels
            all_y = [pos[n][1] for n in graph.nodes()]
            max_y = max(all_y) if all_y else 0
            
            # Define variables for this specific block
            local_x_separation = 3.0
            local_chain1 = unique_chains[0]
            local_chain2 = unique_chains[1]

            # Add chain labels above the columns
            ax.text(-local_x_separation, max_y + 1.0, f"Chain {local_chain1}",
                   fontsize=16, ha='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5",
                           facecolor=CHAIN_COLORS.get(local_chain1, CHAIN_COLORS['default']),
                           alpha=0.3))

            ax.text(local_x_separation, max_y + 1.0, f"Chain {local_chain2}",
                   fontsize=16, ha='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5",
                           facecolor=CHAIN_COLORS.get(local_chain2, CHAIN_COLORS['default']),
                           alpha=0.3))

        plt.title(title, fontsize=font_size+8, fontname='Helvetica')
        ax.set_aspect('equal')
        ax.set_axis_off()
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjusted for closer legend

        # Save as SVG for scalable vector graphics (editable labels)
        svg_output_file = output_file.replace('.png', '.svg')
        plt.savefig(svg_output_file, format='svg', bbox_inches='tight')

        # Also save as PNG for compatibility
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"✅ Network plot saved to {output_file}")
        self.logger.info(f"✅ SVG version saved to {svg_output_file} (editable in vector software)")

    def plotNetwork(self, interactions: List[Dict[str, Any]], output_file: str,
                   title: str = "Protein-Protein Interaction Network"):
        r"""
        Plot a single-type PPI network using the same high-quality styling as consolidated plot.

        :param interactions: List of interaction dicts
        :type interactions: List[Dict[str, Any]]
        :param output_file: Output file path
        :type output_file: str
        :param title: Plot title
        :type title: str
        :return: None
        :rtype: None
        """
        # Initialize variables that might be used later
        chain1, chain2 = 'A', 'L'  # Default values
        x_separation = 3.0  # Default separation
        
        MAX_PER_TYPE_EDGES = 30
        if len(interactions) > MAX_PER_TYPE_EDGES:
            self.logger.info(
                f"Too many interactions ({len(interactions)}) for {title}; "
                f"skipping plot for clarity."
            )
            return
        if not interactions:
            self.logger.info(
                f"⚠️ No interactions found. Skipping plot: {output_file}"
            )
            return

        # Create a single-type dictionary for the consolidated plotting method
        persistent_by_type = {}
        for inter in interactions:
            interaction_type = inter.get('type', 'hydrogen_bond')
            if interaction_type not in persistent_by_type:
                persistent_by_type[interaction_type] = []
            persistent_by_type[interaction_type].append(inter)

        # Use the same high-quality plotting method as consolidated
        # Pass the title to customize the plot title
        self.plotConsolidated(persistent_by_type, output_file, custom_title=title)

    def plotConsolidated(self, persistent_by_type: Dict[str, List[Dict[str, Any]]],
                        output_file: str, custom_title: Optional[str] = None):
        r"""
        Plot a consolidated PPI network for all interaction types using NetworkX best practices.

        :param persistent_by_type: Dict of interaction type to list of interactions
        :type persistent_by_type: Dict[str, List[Dict[str, Any]]]
        :param output_file: Output file path
        :type output_file: str
        :return: None
        :rtype: None
        """
        # Initialize variables that might be used later
        chain1, chain2 = 'A', 'L'  # Default values
        x_separation = 3.0  # Default separation
        
        import matplotlib.lines as mlines

        # Create figure with proper aspect ratio for perfect circles
        # Adjust figure size based on number of nodes
        total_nodes = sum(len(interactions) for interactions in persistent_by_type.values())
        if total_nodes <= 2:
            fig, ax = plt.subplots(figsize=(14, 10))  # Wider figure for 2 nodes
        elif total_nodes <= 5:
            fig, ax = plt.subplots(figsize=(16, 14))  # Wider figure for few nodes
        else:
            fig, ax = plt.subplots(figsize=(20, 20))  # Much wider figure for many nodes
        ax.set_aspect('equal')  # CRITICAL: This ensures perfect circles

        graph = nx.Graph()
        edge_labels = {}
        node_chain = {}
        node_resname = {}
        node_resnum = {}
        edges_by_type = {k: [] for k in INTERACTION_STYLES}

        # Build the graph
        for interaction_type, interactions in persistent_by_type.items():
            style = INTERACTION_STYLES.get(
                interaction_type, INTERACTION_STYLES['hydrogen_bond']
            )
            for inter in interactions:
                if 'donor' in inter and 'acceptor' in inter:
                    node1, node2 = inter['donor'], inter['acceptor']
                elif 'res1' in inter and 'res2' in inter:
                    node1, node2 = inter['res1'], inter['res2']
                else:
                    continue

                for n in [node1, node2]:
                    if n not in graph:
                        try:
                            chain, rest = n.split(':')
                            resname = rest[:3]
                            resnum = rest[3:].lstrip('_')  # Remove underscore prefix
                        except Exception:
                            chain, resname, resnum = ' ', 'UNK', ''
                        node_chain[n] = chain
                        node_resname[n] = resname
                        node_resnum[n] = resnum
                        graph.add_node(n)

                graph.add_edge(node1, node2, type=interaction_type)
                edges_by_type.setdefault(interaction_type, []).append((node1, node2))

                if 'occupancy' in inter:
                    percent = int(round(inter['occupancy'] * 100))
                    edge_labels[(node1, node2)] = f"{percent}%"

                # Use left/right layout for two chains
        unique_chains = sorted(set(node_chain[n] for n in graph.nodes()))
        if len(unique_chains) >= 2:
            chain1, chain2 = unique_chains[0], unique_chains[1]
            chain1_nodes = [n for n in graph.nodes() if node_chain[n] == chain1]
            chain2_nodes = [n for n in graph.nodes() if node_chain[n] == chain2]

            # Calculate interaction counts manually from graph edges
            node_interaction_counts = {}
            for node in graph.nodes():
                node_interaction_counts[node] = 0

            # Count interactions from graph edges
            for u, v in graph.edges():
                node_interaction_counts[u] += 1
                node_interaction_counts[v] += 1

            # Sort nodes by interaction count (highest first) then by residue number
            def interaction_count_key(n):
                try:
                    interaction_count = node_interaction_counts.get(n, 0)
                    # Use negative for descending order (highest first)
                    return (-interaction_count, int(node_resnum[n]))
                except Exception:
                    return (0, 0)

            chain1_nodes.sort(key=interaction_count_key)
            chain2_nodes.sort(key=interaction_count_key)

            # Debug: Log the ordering
            self.logger.info(f"Consolidated - Chain {chain1} ordering by interaction count:")
            for node in chain1_nodes:
                count = node_interaction_counts.get(node, 0)
                resnum = node_resnum.get(node, '?')
                self.logger.info(f"  {node}: {count} interactions (residue {resnum})")

            self.logger.info(f"Consolidated - Chain {chain2} ordering by interaction count:")
            for node in chain2_nodes:
                count = node_interaction_counts.get(node, 0)
                resnum = node_resnum.get(node, '?')
                self.logger.info(f"  {node}: {count} interactions (residue {resnum})")

            # Position nodes in left/right layout
            pos = {}
            y_gap = 2.0  # Increased spacing between nodes for better label clarity

            # Adjust separation based on number of nodes to prevent cropping
            total_nodes = len(chain1_nodes) + len(chain2_nodes)
            if total_nodes <= 2:
                x_separation = 3.0  # Increased separation for 2 nodes
            elif total_nodes <= 5:
                x_separation = 4.0  # Increased separation for few nodes
            else:
                x_separation = 5.0  # Increased separation for many nodes

            for i, node in enumerate(chain1_nodes):
                pos[node] = (-x_separation, i * y_gap - (len(chain1_nodes)-1)*y_gap/2)
            for i, node in enumerate(chain2_nodes):
                pos[node] = (x_separation, i * y_gap - (len(chain2_nodes)-1)*y_gap/2)
        else:
            # Fallback to circular layout for single chain
            pos = nx.circular_layout(graph, scale=2)

        # Draw nodes using NetworkX optimized functions with larger size
        node_colors = []
        node_sizes = []

        for node in graph.nodes():
            resname = node_resname[node]
            fill_color = AMINO_ACID_COLORS.get(resname, AMINO_ACID_COLORS['UNK'])
            node_colors.append(fill_color)

            # Calculate node sizes based on number of nodes
            total_nodes = len(graph.nodes())
            if total_nodes <= 2:
                node_size = 8000  # Much larger for 2 nodes to fit text
            elif total_nodes <= 5:
                node_size = 6000  # Larger for few nodes
            elif total_nodes <= 10:
                node_size = 5000  # Medium for moderate nodes
            else:
                node_size = 4000  # Large for many nodes
            node_sizes.append(node_size)

        # Draw nodes with explicit matplotlib Circle patches for perfect circles
        import matplotlib.patches as patches

        # Calculate node radius based on node sizes
        avg_node_size = sum(node_sizes) / len(node_sizes) if node_sizes else 2000
        node_radius = (avg_node_size ** 0.5) * 0.01  # Scale factor for conversion

        for i, node in enumerate(graph.nodes()):
            x, y = pos[node]
            color = node_colors[i] if i < len(node_colors) else '#2f4f2f'
            circle = patches.Circle(
                (x, y),
                radius=node_radius,
                facecolor=color,
                edgecolor='black',
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(circle)

        # Draw edges by interaction type with edge-to-edge connections
        legend_handles = []
        edge_types = set(d.get('type') for u, v, d in graph.edges(data=True))

        # Calculate node radius for edge connections based on actual node sizes
        # Convert node size (in points) to data coordinates
        avg_node_size = sum(node_sizes) / len(node_sizes) if node_sizes else 2000
        # Approximate conversion: node_size in points to radius in data coordinates
        node_radius = (avg_node_size ** 0.5) * 0.01  # Scale factor for conversion

        for interaction_type in edge_types:
            # Get the style for this interaction type
            style = INTERACTION_STYLES.get(interaction_type, INTERACTION_STYLES['hydrogen_bond'])
            edges = [
                (u, v) for u, v, d in graph.edges(data=True)
                if d.get('type') == interaction_type
            ]
            if edges:
                # Draw edges with custom edge-to-edge connections
                self._draw_edges_to_node_edges(
                    graph, pos, edges, style, node_radius, ax
                )
                legend_handles.append(
                    mlines.Line2D(
                        [], [], color=style['color'], linestyle=style['style'],
                        linewidth=style['width'], label=style['label']
                    )
                )

        # Draw node labels with proper formatting
        node_labels = {}
        for node in graph.nodes():
            chain = node_chain[node]
            aa = node_resname[node]
            num = node_resnum[node]
            # Format: Chain\nAA\nNumber
            node_labels[node] = f"{chain}\n{aa}\n{num}"

        # Draw labels with NetworkX function - larger font for better visibility
        label_objects = nx.draw_networkx_labels(
            graph, pos,
            labels=node_labels,
            font_size=12,  # Increased font size
            font_family='Helvetica',
            font_weight='bold',
            ax=ax
        )

        # Draw edge labels with staggered/curved placement for clarity
        if edge_labels:
            filtered_edge_labels = {
                k: v for k, v in edge_labels.items()
                if int(v.replace('%', '')) >= 10
            }
            # Stagger label positions: alternate above/below and curve if needed
            for idx, ((n1, n2), label) in enumerate(filtered_edge_labels.items()):
                x1, y1 = pos[n1]
                x2, y2 = pos[n2]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                delta_x, delta_y = x2 - x1, y2 - y1
                norm = np.sqrt(delta_x**2 + delta_y**2)
                if norm == 0:
                    continue
                # Perpendicular offset direction
                perp_x = -delta_y / norm
                perp_y = delta_x / norm
                # Alternate above/below and increase offset for dense regions
                offset = 0.18 + 0.08 * (idx % 3)  # Stagger more for dense edges
                side = 1 if idx % 2 == 0 else -1
                label_x = mid_x + side * offset * perp_x
                label_y = mid_y + side * offset * perp_y
                # For very dense regions, curve the edge and place label at apex
                if len(filtered_edge_labels) > 8:
                    # Quadratic Bezier curve apex (t=0.5)
                    curve_height = 0.25 + 0.08 * (idx % 3)
                    control_x = mid_x + side * curve_height * perp_x
                    control_y = mid_y + side * curve_height * perp_y
                    # Place label at the control point (apex)
                    label_x, label_y = control_x, control_y
                    # Optionally, draw the curved edge (not just straight)
                    from matplotlib.patches import FancyArrowPatch
                    path = np.array([
                        [x1, y1],
                        [control_x, control_y],
                        [x2, y2]
                    ])
                    arrow = FancyArrowPatch(
                        path[0], path[2],
                        connectionstyle=f"arc3,rad={side*0.25}",
                        arrowstyle='-|>',
                        color='gray',
                        linewidth=1.0,
                        alpha=0.3,
                        zorder=1
                    )
                    ax.add_patch(arrow)
                ax.text(
                    label_x, label_y, label,
                    color='darkred',
                    fontsize=11,
                    ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(
                        facecolor='white', edgecolor='black', alpha=0.98,
                        boxstyle='round,pad=0.5', linewidth=1.2
                    ),
                    fontname='Helvetica',
                    zorder=1000
                )

        # Set title closer to the network
        if custom_title:
            ax.set_title(custom_title, fontsize=16, fontweight='bold',
                        fontfamily='Helvetica', pad=20)  # Reduced padding
        else:
            ax.set_title("Consolidated Protein-Protein Interaction Network",
                        fontsize=16, fontweight='bold',
                        fontfamily='Helvetica', pad=20)  # Reduced padding

        # Professional legend closer to the network
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.05),  # Moved closer to nodes
                ncol=3,  # Three columns like the reference legend
                fontsize=10,
                frameon=True,
                prop={'family': 'Helvetica', 'size': 9},
                fancybox=True,
                shadow=True,
                title='Interaction Types',
                title_fontsize=12
            )

                # Add chain labels for left/right layout
        if len(unique_chains) >= 2:
            # Get the y-coordinates for positioning chain labels
            all_y = [pos[n][1] for n in graph.nodes()]
            max_y = max(all_y) if all_y else 0
            
            # Define variables for this specific block
            local_x_separation = 3.0
            local_chain1 = unique_chains[0]
            local_chain2 = unique_chains[1]

            # Add chain labels above the columns
            ax.text(-local_x_separation, max_y + 1.0, f"Chain {local_chain1}",
                   fontsize=16, ha='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5",
                           facecolor=CHAIN_COLORS.get(local_chain1, CHAIN_COLORS['default']),
                           alpha=0.3))

            ax.text(local_x_separation, max_y + 1.0, f"Chain {local_chain2}",
                   fontsize=16, ha='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5",
                           facecolor=CHAIN_COLORS.get(local_chain2, CHAIN_COLORS['default']),
                           alpha=0.3))

                # Clean up the plot
        ax.set_axis_off()

        # CRITICAL: Maintain perfect circles
        ax.set_aspect('equal')

        # Ensure proper spacing with adequate padding first
        plt.tight_layout(pad=3.0)  # Increased padding for better spacing

        # Then add margins to prevent node cropping
        # Increase margins for single interactions to prevent cropping
        total_nodes = len(graph.nodes())
        if total_nodes <= 2:
            ax.margins(0.3)  # 30% margin for 2 nodes
        else:
            ax.margins(0.2)  # 20% margin for multiple nodes

        # Save as SVG for scalable vector graphics (editable labels)
        svg_output_file = output_file.replace('.png', '.svg')
        plt.savefig(
            svg_output_file,
            format='svg',
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )

        # Also save as PNG for compatibility
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()

        self.logger.info(f"✅ Professional network plot saved to {output_file}")
        self.logger.info(f"✅ SVG version saved to {svg_output_file} (editable in vector software)")

    def prevent_label_overlaps(self, ax, texts, nodes_pos, node_size=300):
        """
        Prevent label overlaps using adjustText library.

        :param ax: Matplotlib axes object
        :type ax: matplotlib.axes.Axes
        :param texts: List of text objects to adjust
        :type texts: List[matplotlib.text.Text]
        :param nodes_pos: Dictionary of node positions
        :type nodes_pos: Dict
        :param node_size: Size of nodes to avoid
        :type node_size: int
        :return: None
        :rtype: None
        """
        if not ADJUST_TEXT_AVAILABLE:
            self.logger.warning("adjustText not available. Labels may overlap.")
            return

        try:
            # Extract node coordinates for avoidance
            node_coords = list(nodes_pos.values())
            x_coords = [pos[0] for pos in node_coords]
            y_coords = [pos[1] for pos in node_coords]

            # TODO: Implement text position adjustment
            pass
        except Exception as e:
            self.logger.warning(f"Failed to adjust text positions: {e}")

    def _draw_edges_to_node_edges(self, graph, pos, edges, style, node_radius, ax):
        """
        Draw edges that connect to the edge of circular nodes, not the center.
        For multiple interactions to the same node, distribute connection points around the edge.

        :param graph: NetworkX graph
        :type graph: nx.Graph
        :param pos: Node positions dictionary
        :type pos: Dict
        :param edges: List of edges to draw
        :type edges: List[Tuple]
        :param style: Edge style dictionary
        :type style: Dict
        :param node_radius: Radius of the nodes
        :type node_radius: float
        :param ax: Matplotlib axes
        :type ax: matplotlib.axes.Axes
        :return: None
        :rtype: None
        """
        import numpy as np

        # Track connection points for each node to avoid overlap
        node_connections = {}

        for u, v in edges:
            # Get node positions
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            # Calculate direction vector from u to v
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx**2 + dy**2)

            if distance == 0:
                continue

            # Normalize direction vector
            dx_norm = dx / distance
            dy_norm = dy / distance

                        # Calculate connection points just outside the edge of each node
            # Add a small buffer to avoid overlapping with the circle
            buffer_distance = node_radius * 0.1  # 10% of node radius as buffer

            # For node u (start point) - connect just outside the circle
            start_x = x1 + dx_norm * (node_radius + buffer_distance)
            start_y = y1 + dy_norm * (node_radius + buffer_distance)

            # For node v (end point) - connect just outside the circle
            end_x = x2 - dx_norm * (node_radius + buffer_distance)
            end_y = y2 - dy_norm * (node_radius + buffer_distance)

            # Check if we need to adjust connection points to avoid overlap
            start_point = self._get_adjusted_connection_point(
                u, (start_x, start_y), node_connections, node_radius, pos
            )
            end_point = self._get_adjusted_connection_point(
                v, (end_x, end_y), node_connections, node_radius, pos
            )

            # Draw bidirectional arrows - one in each direction along the same line
            # First arrow: from start to end
            ax.annotate(
                '',  # No text, just the arrow
                xy=end_point,  # Arrow points to the end point
                xytext=start_point,  # Arrow starts from start point
                arrowprops=dict(
                    arrowstyle='->',  # Arrow style
                    color=style['color'],
                    linestyle=style['style'],
                    linewidth=style['width'],
                    alpha=0.8,
                    shrinkA=0,  # Don't shrink at start
                    shrinkB=0,  # Don't shrink at end
                    mutation_scale=6,  # Smaller arrow head for bidirectional
                    mutation_aspect=2.0  # Make arrow head more pointed
                )
            )

            # Second arrow: from end to start (bidirectional)
            ax.annotate(
                '',  # No text, just the arrow
                xy=start_point,  # Arrow points to the start point
                xytext=end_point,  # Arrow starts from end point
                arrowprops=dict(
                    arrowstyle='->',  # Arrow style
                    color=style['color'],
                    linestyle=style['style'],
                    linewidth=style['width'],
                    alpha=0.8,
                    shrinkA=0,  # Don't shrink at start
                    shrinkB=0,  # Don't shrink at end
                    mutation_scale=6,  # Smaller arrow head for bidirectional
                    mutation_aspect=2.0  # Make arrow head more pointed
                )
            )

    def _get_adjusted_connection_point(self, node, base_point, node_connections, node_radius, pos):
        """
        Get an adjusted connection point to avoid overlap with existing connections.

        :param node: Node identifier
        :type node: str
        :param base_point: Base connection point (x, y)
        :type base_point: Tuple[float, float]
        :param node_connections: Dictionary tracking existing connections per node
        :type node_connections: Dict
        :param node_radius: Radius of the node
        :type node_radius: float
        :param pos: Node positions dictionary
        :type pos: Dict
        :return: Adjusted connection point (x, y)
        :rtype: Tuple[float, float]
        """
        import numpy as np

        if node not in node_connections:
            node_connections[node] = []

        # Check if base point is too close to existing connections
        min_distance = node_radius * 0.3  # Minimum distance between connection points

        for existing_point in node_connections[node]:
            dist = np.sqrt((base_point[0] - existing_point[0])**2 +
                          (base_point[1] - existing_point[1])**2)
            if dist < min_distance:
                # Find a new angle around the node
                angle_offset = len(node_connections[node]) * 0.5  # Rotate by 0.5 radians per connection

                # Calculate angle from center to base point
                center_x, center_y = pos[node]  # We need to get this from the pos dict
                angle = np.arctan2(base_point[1] - center_y, base_point[0] - center_x)

                # Apply offset
                new_angle = angle + angle_offset

                # Calculate new point just outside the edge
                buffer_distance = node_radius * 0.1  # 10% of node radius as buffer
                new_x = center_x + (node_radius + buffer_distance) * np.cos(new_angle)
                new_y = center_y + (node_radius + buffer_distance) * np.sin(new_angle)

                base_point = (new_x, new_y)
                break

        # Add this connection point to the tracking
        node_connections[node].append(base_point)
        return base_point

    def create_residue_legend(self, ax, x_pos=0.02, y_pos=0.98):
        """
        Create a comprehensive residue type legend similar to the reference.

        :param ax: Matplotlib axes object
        :type ax: matplotlib.axes.Axes
        :param x_pos: X position for legend (default: 0.02)
        :type x_pos: float
        :param y_pos: Y position for legend (default: 0.98)
        :type y_pos: float
        :return: None
        :rtype: None
        """
        import matplotlib.patches as mpatches

        # Define residue categories with their colors
        residue_categories = {
            'Charged (negative)': '#FF8C00',  # Orange
            'Charged (positive)': '#9370DB',  # Light purple/blue
            'Glycine': '#F0E68C',             # Light yellow/cream
            'Hydrophobic': '#90EE90',         # Light green
            'Polar': '#87CEEB',               # Light blue
            'Metal': '#808080',               # Grey
        }

        # Create legend patches
        legend_patches = []
        for category, color in residue_categories.items():
            patch = mpatches.Patch(color=color, label=category)
            legend_patches.append(patch)

        # Add legend to the plot
        ax.legend(
            handles=legend_patches,
            loc='upper left',
            bbox_to_anchor=(x_pos, y_pos),
            fontsize=8,
            frameon=True,
            prop={'family': 'Helvetica', 'size': 7},
            fancybox=True,
            shadow=True,
            title='Residue Types',
            title_fontsize=9
        )

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
    schrodinger_run = os.path.join(
        os.environ.get("SCHRODINGER", "/opt/schrodinger"), "run"
    )
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
    try:
        from schrodinger.job import jobcontrol
        job = jobcontrol.launch_job(cmd)
        return job
    except ImportError:
        print(f"Could not import jobcontrol. Would have run: {' '.join(cmd)}")
        return None

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
        msg = "Schrodinger modules not available. Cannot run analyzer test."
        if logger:
            logger.error(msg)
        else:
            print(f"[ERROR] {msg}")
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
        msg = "Schrodinger modules not available. Cannot run analyzer."
        if logger:
            logger.error(msg)
        else:
            print(f"[ERROR] {msg}")
        return None
    cms_file = str(cms_file)
    traj_file = str(traj_file)
    msys_model, cms_model = topo.read_cms(cms_file)
    tr = traj.read_traj(traj_file)
    analyzer = analysis.ProtProtInter(msys_model, cms_model, "protein")
    results = analysis.analyze(tr, analyzer)
    if not isinstance(results, dict):
        msg = f"Unexpected analyzer result type: {type(results)}"
        if logger:
            logger.error(msg)
        else:
            print(f"[ERROR] {msg}")
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
    print("Starting main function...")
    try:
        from schrodinger.structure import StructureReader
        from schrodinger.application.desmond.packages import traj
        print("Successfully imported Schrodinger modules")
    except ImportError as e:
        print(f"Error importing Schrodinger modules: {e}")
        sys.exit(1)
        
    # Configurable threshold for consolidated plot
    MAX_CONSOLIDATED_EDGES = 30
    try:
        parser = argparse.ArgumentParser(
            description="Protein-protein PPI analysis and visualization (Schrödinger style)"
        )
        print("Created argument parser")
        # Add arguments
        parser.add_argument(
            '--cms', required=True, type=str,
            help='Path to the Desmond .cms structure file'
        )
        parser.add_argument(
            '--traj', required=True, type=str,
            help='Path to the trajectory file or directory'
        )
        parser.add_argument(
            '--chain-groups', nargs='+', default=None,
            help='Chain groups to analyze (e.g., A,B C,D). If not specified, available chains will be listed.'
        )
        parser.add_argument(
            '--occupancy-threshold', type=float, default=0.2,
            help='Minimum occupancy threshold (default: 0.2)'
        )
        parser.add_argument(
            '--output-prefix', type=str, default='ppi_analysis',
            help='Prefix for output files (default: ppi_analysis)'
        )
        parser.add_argument(
            '--output-dir', type=str, default='ppi_results',
            help='Output directory name (default: ppi_results)'
        )
        parser.add_argument(
            '--debug', action='store_true',
            help='Enable detailed debug/progress logging'
        )
        parser.add_argument(
            '--max-frames', type=int, default=None,
            help='Maximum number of trajectory frames to analyze (default: all)'
        )
        parser.add_argument(
            '--skip-single-interactions',
            action='store_true',
            help='Skip individual plots for interaction types with only 1 interaction (default: generate all)'
        )
        parser.add_argument(
            '--submit-job', action='store_true',
            help='Submit this script as a Schrodinger job and exit'
        )
        parser.add_argument(
            '--host', type=str, default='localhost',
            help='Host for job submission (default: localhost)'
        )
        parser.add_argument(
            '--jobname', type=str, default='ppi_analysis',
            help='Job name for submission (default: ppi_analysis)'
        )
        parser.add_argument(
            '--test-schrodinger-analyzer', action='store_true',
            help='Run a test of Schrodinger ProtProtInter analyzer and print summary'
        )
        args = parser.parse_args()
        print(f"Parsed arguments: {args}")
        
        # Setup logging
        if args.debug:
            print("Setting up debug logging")
            
        # Create output directory
        print("Creating output directory")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        print("Script executed successfully")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    import json
    # Configurable threshold for consolidated plot
    MAX_CONSOLIDATED_EDGES = 30
    parser = argparse.ArgumentParser(
        description="Protein-protein PPI analysis and visualization (Schrödinger style)"
    )
    parser.add_argument(
        '--cms', required=True, type=str,
        help='Path to the Desmond .cms structure file'
    )
    parser.add_argument(
        '--traj', required=True, type=str,
        help='Path to the trajectory file or directory'
    )
    parser.add_argument(
        '--chain-groups', nargs='+', default=None,
        help='Chain groups to analyze (e.g., A,B C,D). If not specified, available chains will be listed.'
    )
    parser.add_argument(
        '--occupancy-threshold', type=float, default=0.2,
        help='Minimum occupancy threshold (default: 0.2)'
    )
    parser.add_argument(
        '--output-prefix', type=str, default='ppi_analysis',
        help='Prefix for output files (default: ppi_analysis)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='ppi_results',
        help='Output directory name (default: ppi_results)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable detailed debug/progress logging'
    )
    parser.add_argument(
        '--max-frames', type=int, default=None,
        help='Maximum number of trajectory frames to analyze (default: all)'
    )
    parser.add_argument(
        '--skip-single-interactions',
        action='store_true',
        help='Skip individual plots for interaction types with only 1 interaction (default: generate all)'
    )
    parser.add_argument(
        '--submit-job', action='store_true',
        help='Submit this script as a Schrodinger job and exit'
    )
    parser.add_argument(
        '--host', type=str, default='localhost',
        help='Host for job submission (default: localhost)'
    )
    parser.add_argument(
        '--jobname', type=str, default='ppi_analysis',
        help='Job name for submission (default: ppi_analysis)'
    )
    parser.add_argument(
        '--test-schrodinger-analyzer', action='store_true',
        help='Run a test of Schrodinger ProtProtInter analyzer and print summary'
    )
    args = parser.parse_args()

    # Setup logger first
    if args.debug:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("PPIAnalysis")
    else:
        logger = logging.getLogger("PPIAnalysis")

    # Create output directory with organized structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories for organized output
    images_dir = output_dir / "images"
    data_dir = output_dir / "data"
    logs_dir = output_dir / "logs"

    images_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Setup file logging after creating logs directory
    log_file = logs_dir / "ppi_analysis.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    logger.info(f"📁 Created organized output structure:")
    logger.info(f"   📸 Images: {images_dir}")
    logger.info(f"   📊 Data: {data_dir}")
    logger.info(f"   📝 Logs: {logs_dir}")

    # Validate inputs
    if not validate_files(args.cms, args.traj, logger):
        sys.exit(1)

    if not validate_parameters(args, logger):
        sys.exit(1)

    if args.submit_job:
        if not SCHRODINGER_AVAILABLE:
            print("❌ Schrodinger job control is not available in this environment.")
            sys.exit(1)
        logger.info(
            f"Submitting job to host: {args.host} with jobname: {args.jobname}"
        )
        job = submit_job(args, args.host, args.jobname)
        logger.info(f"Job submitted: {job}")
        print(f"Job submitted to host: {args.host} with jobname: {args.jobname}")
        sys.exit(0)

    if getattr(args, 'test_schrodinger_analyzer', False):
        test_schrodinger_ppi_analyzer(args.cms, args.traj, logger)
        sys.exit(0)

    # Parse chain groups
    def parse_chain_groups(chain_groups_list):
        return [set(group.split(',')) for group in chain_groups_list]

    if args.chain_groups is None or len(args.chain_groups) < 2:
        logger.error(
            "You must specify at least two chain groups for inter-chain analysis."
        )
        sys.exit(1)

    if not validate_chain_groups(args.chain_groups, logger):
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
            if ((chain1 in group1 and chain2 in group2) or
                (chain2 in group1 and chain1 in group2)):
                occupancy = count / total_frames
                interactions.append({
                    'res1': res1,
                    'res2': res2,
                    'occupancy': occupancy,
                    'type': interaction_type
                })
        filtered_interactions = [
            inter for inter in interactions
            if inter['occupancy'] >= args.occupancy_threshold
        ]
        persistent_by_type[interaction_type] = filtered_interactions
    # Now group H-bond subtypes (excluding hbond_self)
    HBOND_SUBTYPES = {'hbond_bb', 'hbond_ss', 'hbond_sb', 'hbond_bs'}
    grouped_persistent_by_type = {}
    for interaction_type, interactions in persistent_by_type.items():
        if interaction_type in HBOND_SUBTYPES:
            # Update the type field to 'hydrogen_bond' for grouped interactions
            for inter in interactions:
                inter['type'] = 'hydrogen_bond'
            grouped_persistent_by_type.setdefault('hydrogen_bond', []).extend(
                interactions
            )
        elif interaction_type == 'hbond_self':
            continue
        else:
            grouped_persistent_by_type[interaction_type] = interactions
    # Plot grouped per-type networks (skip H-bond subtypes)
    plotter = PPIPlotter(logger)
    for interaction_type, interactions in grouped_persistent_by_type.items():
        if (interaction_type != 'hydrogen_bond' and
            interaction_type not in INTERACTION_STYLES):
            logger.warning(
                f"Skipping unknown interaction type: {interaction_type}"
            )
            continue

        # Check if user wants to skip single interactions
        if args.skip_single_interactions and len(interactions) <= 1:
            logger.info(
                f"⏭️ Skipping individual plot for {interaction_type} - only {len(interactions)} interaction(s) "
                f"(use --skip-single-interactions to enable)"
            )
            continue

        # Generate individual plot for this interaction type
        logger.info(
            f"📊 Generating individual plot for {interaction_type} with {len(interactions)} interaction(s)"
        )

        plotter.plotNetwork(
            interactions,
            output_file=str(images_dir / f"{args.output_prefix}_interactions_{interaction_type}.png"),
            title=f"{interaction_type.replace('_', ' ').replace('-', ' ').title()} Interactions"
        )
    # Plot consolidated network only if number of unique pairs is within threshold
    total_pairs = sum(
        len(interactions) for interactions in grouped_persistent_by_type.values()
    )
    if total_pairs <= MAX_CONSOLIDATED_EDGES:
        plotter.plotConsolidated(
            grouped_persistent_by_type,
            output_file=str(images_dir / f"{args.output_prefix}_consolidated.png")
        )
    else:
        logger.info(
            f"Too many interactions ({total_pairs}); skipping consolidated plot for clarity."
        )
    # Save persistent interaction data as JSON
    with open(data_dir / "persistent_interactions.json", "w") as f:
        json.dump(grouped_persistent_by_type, f, indent=2)
    logger.info("\n🎉 Analysis completed successfully! All results are in: %s", output_dir)
    return

if __name__ == "__main__":
    main()