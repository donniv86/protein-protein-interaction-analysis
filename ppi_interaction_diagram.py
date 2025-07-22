import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any
import traceback
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.patheffects as path_effects

# Schrodinger-style color and style conventions
INTERACTION_STYLES = {
    'hydrogen_bond': {'color': '#2E86AB', 'style': 'dashed', 'width': 2.0, 'label': 'Hydrogen Bond'},
    'salt_bridge': {'color': '#2E8B57', 'style': 'solid', 'width': 3.0, 'label': 'Salt Bridge'},
    'pi_pi': {'color': '#8A2BE2', 'style': 'dotted', 'width': 2.5, 'label': 'π-π Stacking'},
    'pi_cation': {'color': '#FF4500', 'style': 'dashdot', 'width': 2.5, 'label': 'π-Cation'},
}
# Schrödinger professional color schemes
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


def consolidated_ppi_diagram(persistent_by_type: Dict[str, List[Dict[str, Any]]], output_file: str = "ppi_consolidated.png"):
    r"""
    Create a consolidated 2D protein-protein interaction diagram with all interaction types.
    Each node is a residue (drawn as a colored circle), edges are colored/styled by interaction type.
    Occupancy is shown as edge labels.
    :param persistent_by_type: Dictionary mapping interaction types to lists of persistent interactions.
    :type persistent_by_type: Dict[str, List[Dict[str, Any]]]
    :param output_file: Output filename for the plot.
    :type output_file: str
    :raise RuntimeError: If plotting or saving fails.
    """
    try:
        G = nx.Graph()
        edge_labels = {}
        edge_styles = {}
        node_chain = {}
        node_resname = {}
        node_resnum = {}
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
                        # Parse node info
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
                edge_styles[(n1, n2)] = style
                if 'occupancy' in inter:
                    percent = int(round(inter['occupancy'] * 100))
                    edge_labels[(n1, n2)] = f"{percent}%"

        # Determine groups for left/right layout
        # Group nodes by chain (or chain group)
        group1_chains = set()
        group2_chains = set()
        # Try to infer groups from the first and second half of unique chains
        unique_chains = sorted(set(node_chain[n] for n in G.nodes()))
        if len(unique_chains) > 1:
            group1_chains = set([unique_chains[0]])
            group2_chains = set(unique_chains[1:])
        else:
            group1_chains = set([unique_chains[0]])
            group2_chains = set()
        # Assign nodes to groups
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
                    ax.add_patch(poly)
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
        # Draw edges by interaction type
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
        # Draw occupancy labels
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='purple', font_size=13, label_pos=0.5, ax=ax)
        plt.title("Consolidated Protein-Protein Interaction Diagram", fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        if legend_handles:
            plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=13, frameon=True)
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ Consolidated PPI diagram saved to {output_file}")
        except Exception as save_exc:
            print(f"❌ Error saving consolidated PPI diagram to {output_file}: {save_exc}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to save plot: {save_exc}") from save_exc
        finally:
            plt.close()
    except Exception as exc:
        print(f"❌ Error generating consolidated PPI diagram: {exc}")
        traceback.print_exc()
        raise RuntimeError(f"Failed to generate consolidated PPI diagram: {exc}") from exc

# Set up figure and axis before drawing
plt.figure(figsize=(16, 12))
ax = plt.gca()
node_size = 2200