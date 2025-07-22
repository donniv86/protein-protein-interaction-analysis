#!/usr/bin/env python3
"""
Clean Protein-Protein Interaction Visualization Module
Shows ONLY residues involved in the specified interactions, not all residues from the chains.

Features:
- Clean, focused visualization
- Only shows interacting residues
- Professional Schrodinger-style styling
- Publication-quality output
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings
warnings.filterwarnings('ignore')

# Schrodinger imports
try:
    from schrodinger.structure import StructureReader
    from schrodinger.structutils.analyze import evaluate_asl, hbond
    from schrodinger.application.desmond.packages import topo, traj
    from schrodinger.utils import sea
    from schrodinger.ui import sketcher
    from schrodinger.application.desmond.image_generator import SchrodImageGenerator
    from schrodinger.application.desmond.event_analysis import pl_image_tools
    from schrodinger.utils import fileutils
    from PyQt6 import QtWidgets, QtCore, QtGui
    from PyQt6.QtCore import QPointF
    SCHRODINGER_AVAILABLE = True
    print("✅ Schrodinger modules imported successfully")
except ImportError as e:
    SCHRODINGER_AVAILABLE = False
    print(f"❌ Schrodinger modules not available: {e}")
    sys.exit(1)

# Data structures for protein-protein interactions
class PPResData:
    """Class to store information about a residue in a protein-protein interaction."""

    def __init__(self, chain: str, res_name: str, res_num: int, interaction_atom: str = None):
        self.chain = chain
        self.res_name = res_name
        self.res_num = res_num
        self.interaction_atom = interaction_atom

    @classmethod
    def fromFullName(cls, full_name: str) -> 'PPResData':
        """Create a PPResData object from a protein label e.g. 'A:THR_45:O' or 'A:ASN117'."""
        parts = full_name.split(':')
        chain = parts[0]

        # Handle different formats: 'THR_45:O' or 'ASN117'
        if '_' in parts[1]:
            res_info = parts[1].split('_')
            res_name = res_info[0]
            res_num = int(res_info[1])
            interaction_atom = parts[2] if len(parts) > 2 else None
        else:
            # Handle format like 'ASN117'
            import re
            match = re.match(r'([A-Z]+)(\d+)', parts[1])
            if match:
                res_name = match.group(1)
                res_num = int(match.group(2))
                interaction_atom = parts[2] if len(parts) > 2 else None
            else:
                # Fallback
                res_name = parts[1]
                res_num = 0
                interaction_atom = parts[2] if len(parts) > 2 else None

        return cls(chain, res_name, res_num, interaction_atom)

    def fullName(self) -> str:
        """Get the full name of the residue."""
        if self.interaction_atom:
            return f"{self.chain}:{self.res_name}_{self.res_num}:{self.interaction_atom}"
        return f"{self.chain}:{self.res_name}_{self.res_num}"

    def __str__(self) -> str:
        return self.fullName()

    def __repr__(self) -> str:
        return f"PPResData({self})"

class PPInteraction:
    """Class storing information about a protein-protein interaction."""

    def __init__(self, frame_num: int, res1: PPResData, res2: PPResData,
                 interaction_type: str, distance: float = None, occupancy: float = 1.0):
        self.frame_num = frame_num
        self.res1 = res1
        self.res2 = res2
        self.interaction_type = interaction_type
        self.distance = distance
        self.occupancy = occupancy

    def __str__(self) -> str:
        return f"{self.interaction_type}: {self.res1} <-> {self.res2} (occ: {self.occupancy:.2f})"

    def __repr__(self) -> str:
        return f"PPInteraction({self})"

class CleanPPIVisualizer:
    """
    Clean protein-protein interaction visualizer showing ONLY interacting residues.
    """

    def __init__(self):
        """Initialize the clean PPI visualizer."""
        self.qapp = None
        self.sketch = None
        self.image_generator = None
        self.residues = {}
        self.interactions = []

        # Professional Schrodinger-style interaction styling
        self.interaction_styles = {
            'hydrogen_bond': {
                'color': '#2E86AB',  # Schrodinger blue
                'style': 'dashed',
                'linewidth': 2.0,
                'label': 'Hydrogen Bond',
                'symbol': '---'
            },
            'salt_bridge': {
                'color': '#2E8B57',  # Schrodinger green
                'style': 'solid',
                'linewidth': 3.0,
                'label': 'Salt Bridge',
                'symbol': '==='
            },
            'pi_pi': {
                'color': '#8A2BE2',  # Schrodinger purple
                'style': 'dotted',
                'linewidth': 2.5,
                'label': 'π-π Stacking',
                'symbol': '...'
            },
            'pi_cation': {
                'color': '#FF4500',  # Schrodinger orange
                'style': 'dashdot',
                'linewidth': 2.5,
                'label': 'π-Cation',
                'symbol': '-.-'
            },
            'hydrophobic': {
                'color': '#FFD700',  # Schrodinger gold
                'style': 'solid',
                'linewidth': 1.5,
                'label': 'Hydrophobic',
                'symbol': '---'
            },
            'disulfide': {
                'color': '#FF6347',  # Schrodinger red
                'style': 'solid',
                'linewidth': 2.5,
                'label': 'Disulfide',
                'symbol': '==='
            }
        }

        # Professional chain color scheme (Schrodinger standard)
        self.chain_colors = {
            'A': '#FF6B6B',      # Red
            'B': '#4ECDC4',      # Teal
            'C': '#45B7D1',      # Blue
            'D': '#96CEB4',      # Green
            'E': '#FFEAA7',      # Yellow
            'F': '#DDA0DD',      # Plum
            'L': '#FFA07A',      # Light Salmon
            'default': '#87CEEB' # Sky Blue
        }

        # Schrodinger Amino Acid Color Scheme (professional standard)
        self.amino_acid_colors = {
            # Hydrophobic (Non-polar) - Cool blues and grays
            'ALA': '#87CEEB',  # Sky blue
            'VAL': '#4682B4',  # Steel blue
            'LEU': '#1E90FF',  # Dodger blue
            'ILE': '#0000CD',  # Medium blue
            'MET': '#4169E1',  # Royal blue
            'PHE': '#191970',  # Midnight blue
            'TRP': '#000080',  # Navy blue
            'PRO': '#483D8B',  # Dark slate blue

            # Hydrophilic (Polar) - Soft greens and teals
            'GLY': '#98FB98',  # Pale green
            'SER': '#90EE90',  # Light green
            'THR': '#32CD32',  # Lime green
            'CYS': '#228B22',  # Forest green
            'TYR': '#006400',  # Dark green
            'ASN': '#20B2AA',  # Light sea green
            'GLN': '#48D1CC',  # Medium turquoise

            # Charged (Acidic) - Warm reds and pinks
            'ASP': '#FFB6C1',  # Light pink
            'GLU': '#FF69B4',  # Hot pink

            # Charged (Basic) - Cool purples and blues
            'LYS': '#9370DB',  # Medium purple
            'ARG': '#8A2BE2',  # Blue violet
            'HIS': '#4B0082',  # Indigo

            # Special cases
            'UNK': '#C0C0C0',  # Silver
        }

        print("🎨 Clean PPI Visualizer Initialized")

    def create_clean_ppi_diagram(self, interactions: List[PPInteraction],
                               output_prefix: str = "clean_ppi",
                               threshold: float = 0.3, dpi: int = 300):
        """
        Create clean protein-protein interaction diagram showing ONLY interacting residues.
        """
        print("🎨 Creating clean PPI diagram (interacting residues only)...")

        try:
            # Filter interactions by threshold
            filtered_interactions = [i for i in interactions if i.occupancy >= threshold]
            print(f"   Showing {len(filtered_interactions)} interactions (threshold: {threshold})")

            # Extract unique residues from interactions
            unique_residues = self.extract_unique_residues(filtered_interactions)
            print(f"   Found {len(unique_residues)} unique interacting residues")

            # Create clean diagram
            self.create_clean_diagram(unique_residues, filtered_interactions, output_prefix)

            print(f"✅ Clean PPI diagram created successfully!")
            print(f"   Image: {output_prefix}.png")

        except Exception as e:
            print(f"❌ Error creating clean diagram: {e}")
            raise

    def extract_unique_residues(self, interactions: List[PPInteraction]) -> Dict[str, PPResData]:
        """Extract unique residues from interactions."""
        unique_residues = {}

        for interaction in interactions:
            # Add residue 1
            res1_key = f"{interaction.res1.chain}:{interaction.res1.res_name}_{interaction.res1.res_num}"
            unique_residues[res1_key] = interaction.res1

            # Add residue 2
            res2_key = f"{interaction.res2.chain}:{interaction.res2.res_name}_{interaction.res2.res_num}"
            unique_residues[res2_key] = interaction.res2

        return unique_residues

    def create_clean_diagram(self, unique_residues: Dict[str, PPResData],
                           interactions: List[PPInteraction], output_prefix: str):
        """Create clean diagram with only interacting residues."""

        # Set up the plot with professional dimensions
        fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(16, 10),
                                                gridspec_kw={'width_ratios': [4, 1]})

        # Professional Schrodinger color scheme
        fig.patch.set_facecolor('#FAFAFA')
        ax_main.set_facecolor('#FFFFFF')
        ax_legend.set_facecolor('#FAFAFA')

        # Create clean layout for only interacting residues
        pos = self.create_clean_layout(unique_residues)

        # Draw only interacting residues
        self.draw_clean_residues(ax_main, unique_residues, pos)

        # Draw interactions
        self.draw_clean_interactions(ax_main, interactions, pos)

        # Add professional legend
        self.create_clean_legend(ax_legend, interactions, unique_residues)

        # Professional title and styling
        self.apply_clean_styling(ax_main, len(interactions))

        # Save high-quality image
        output_file = f"{output_prefix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Clean diagram saved: {output_file}")

    def create_clean_layout(self, unique_residues: Dict[str, PPResData]) -> Dict[str, Tuple[float, float]]:
        """Create clean layout for only interacting residues."""
        pos = {}

        # Separate residues by chain
        chain_residues = defaultdict(list)
        for res_key, res_data in unique_residues.items():
            chain_residues[res_data.chain].append((res_key, res_data))

        chains_list = list(chain_residues.keys())

        if len(chains_list) == 2:
            # Two-chain interaction: Clean side-by-side layout
            chain1, chain2 = chains_list[0], chains_list[1]

            # Chain 1: Left side
            chain1_residues = chain_residues[chain1]
            chain1_center = np.array([-0.8, 0])
            chain1_radius = 0.3

            for i, (res_key, res_data) in enumerate(chain1_residues):
                angle = (i / len(chain1_residues)) * 2 * np.pi
                radius_var = chain1_radius + 0.05 * np.sin(3 * angle)
                x = chain1_center[0] + radius_var * np.cos(angle)
                y = chain1_center[1] + radius_var * np.sin(angle)
                pos[res_key] = (x, y)

            # Chain 2: Right side
            chain2_residues = chain_residues[chain2]
            chain2_center = np.array([0.8, 0])
            chain2_radius = 0.3

            for i, (res_key, res_data) in enumerate(chain2_residues):
                angle = (i / len(chain2_residues)) * 2 * np.pi
                radius_var = chain2_radius + 0.05 * np.sin(3 * angle)
                x = chain2_center[0] + radius_var * np.cos(angle)
                y = chain2_center[1] + radius_var * np.sin(angle)
                pos[res_key] = (x, y)

        else:
            # Single chain: Circular layout
            all_residues = list(unique_residues.items())
            center = np.array([0, 0])
            radius = 0.5

            for i, (res_key, res_data) in enumerate(all_residues):
                angle = (i / len(all_residues)) * 2 * np.pi
                radius_var = radius + 0.05 * np.sin(3 * angle)
                x = center[0] + radius_var * np.cos(angle)
                y = center[1] + radius_var * np.sin(angle)
                pos[res_key] = (x, y)

        return pos

    def draw_clean_residues(self, ax, unique_residues: Dict[str, PPResData], pos: Dict[str, Tuple[float, float]]):
        """Draw only interacting residues with professional styling."""
        for res_key, res_data in unique_residues.items():
            if res_key in pos:
                x, y = pos[res_key]

                # Professional amino acid coloring
                aa_color = self.amino_acid_colors.get(res_data.res_name, self.amino_acid_colors['UNK'])

                # Professional node sizing based on importance
                if res_data.res_name in ['ARG', 'LYS', 'ASP', 'GLU']:  # Charged
                    size = 500
                elif res_data.res_name in ['PHE', 'TYR', 'TRP', 'HIS']:  # Aromatic
                    size = 450
                elif res_data.res_name in ['SER', 'THR', 'ASN', 'GLN']:  # Polar
                    size = 400
                else:  # Hydrophobic
                    size = 350

                # Draw professional node
                circle = plt.Circle((x, y), size/10000, facecolor=aa_color,
                                  edgecolor='black', linewidth=2, alpha=0.9)
                ax.add_patch(circle)

                # Professional labeling
                ax.text(x, y, f"{res_data.chain}:{res_data.res_name}\n{res_data.res_num}",
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white', bbox=dict(boxstyle="round,pad=0.3",
                                               facecolor='black', alpha=0.7))

    def draw_clean_interactions(self, ax, interactions: List[PPInteraction],
                              pos: Dict[str, Tuple[float, float]]):
        """Draw interactions with professional styling."""
        for interaction in interactions:
            res1_key = f"{interaction.res1.chain}:{interaction.res1.res_name}_{interaction.res1.res_num}"
            res2_key = f"{interaction.res2.chain}:{interaction.res2.res_name}_{interaction.res2.res_num}"

            if res1_key in pos and res2_key in pos:
                x1, y1 = pos[res1_key]
                x2, y2 = pos[res2_key]

                # Get professional interaction styling
                style_config = self.interaction_styles.get(interaction.interaction_type,
                                                         self.interaction_styles['hydrogen_bond'])

                # Professional interaction line
                ax.plot([x1, x2], [y1, y2], color=style_config['color'],
                       linestyle=style_config['style'], linewidth=style_config['linewidth'],
                       alpha=min(0.9, max(0.3, interaction.occupancy)))

                # Professional interaction label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, f'{interaction.occupancy*100:.0f}%',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color=style_config['color'],
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    def create_clean_legend(self, ax, interactions: List[PPInteraction],
                          unique_residues: Dict[str, PPResData]):
        """Create professional legend."""
        legend_elements = []

        # Interaction types legend
        for interaction_type, style_config in self.interaction_styles.items():
            if any(i.interaction_type == interaction_type for i in interactions):
                legend_elements.append(
                    mpatches.Patch(color=style_config['color'], label=style_config['label'])
                )

        # Amino acid categories legend
        aa_categories = {
            'Charged (Basic)': ['LYS', 'ARG', 'HIS'],
            'Charged (Acidic)': ['ASP', 'GLU'],
            'Polar': ['SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN'],
            'Hydrophobic': ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']
        }

        for category, residues in aa_categories.items():
            if any(res_data.res_name in residues for res_data in unique_residues.values()):
                sample_res = residues[0]
                color = self.amino_acid_colors.get(sample_res, '#CCCCCC')
                legend_elements.append(
                    mpatches.Patch(color=color, label=f'Amino Acid: {category}')
                )

        # Professional legend layout
        ax.legend(handles=legend_elements, loc='center', fontsize=10,
                 frameon=True, fancybox=True, shadow=True)
        ax.axis('off')

    def apply_clean_styling(self, ax, num_interactions: int):
        """Apply professional styling."""
        # Professional title
        ax.set_title(
            f"Protein-Protein Interaction Network\n"
            f"Clean View - Interacting Residues Only\n"
            f"({num_interactions} interactions)",
            fontsize=16, fontweight='bold', pad=20, color='#2C3E50'
        )

        # Professional axis styling
        ax.axis('off')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])

        # Professional grid
        ax.grid(True, alpha=0.05, linestyle='-', linewidth=0.3)

def parse_ppi_interactions_from_analyzer(analyzer_results: List[Dict]) -> List[PPInteraction]:
    """Parse interaction results from PPI analyzer into PPInteraction objects."""
    interactions = []

    for result in analyzer_results:
        try:
            # Extract residue information - handle different formats
            res1_str = result.get('donor', result.get('residue1', ''))
            res2_str = result.get('acceptor', result.get('residue2', ''))

            if not res1_str or not res2_str:
                continue

            # Parse residue data
            res1 = PPResData.fromFullName(res1_str)
            res2 = PPResData.fromFullName(res2_str)

            # Create interaction object
            interaction = PPInteraction(
                frame_num=result.get('frame', 0),
                res1=res1,
                res2=res2,
                interaction_type=result.get('type', 'unknown'),
                distance=result.get('distance', None),
                occupancy=result.get('occupancy', 1.0)
            )

            interactions.append(interaction)

        except Exception as e:
            print(f"   Warning: Could not parse interaction {result}: {e}")
            continue

    return interactions

def main():
    """Main function for testing."""
    if not SCHRODINGER_AVAILABLE:
        print("❌ Schrodinger modules not available")
        return

    # Test the clean visualizer
    visualizer = CleanPPIVisualizer()

    # Load test data
    try:
        with open('my_analysis21_interactions.json', 'r') as f:
            analyzer_results = json.load(f)

        # Limit to first 10 interactions for clean visualization
        limited_results = analyzer_results[:10]
        interactions = parse_ppi_interactions_from_analyzer(limited_results)

        # Create clean diagram
        visualizer.create_clean_ppi_diagram(
            interactions=interactions,
            output_prefix='clean_ppi_10_interactions'
        )

        print("✅ Clean PPI visualization completed!")

    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()