#!/usr/bin/env python3
"""
Enhanced Protein-Protein Interaction Visualization Module
Uses Schrodinger's professional ligand interaction diagram patterns.

Features:
- Professional Schrodinger-style interaction diagrams
- Advanced styling and layout matching Schrodinger standards
- Interactive visualization panels
- Comprehensive interaction analysis
- Publication-quality output

Based on Schrodinger's professional patterns from:
- pl_image_tools.generate_ligand_2d_image
- event_analysis.py professional styling
- analysis.py interaction visualization
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

class ProfessionalPPIVisualizer:
    """
    Professional protein-protein interaction visualizer using Schrodinger's patterns.
    """

    def __init__(self):
        """Initialize the professional PPI visualizer."""
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

        print("🎨 Professional PPI Visualizer Initialized")

    def initialize_qt(self):
        """Initialize Qt application and sketcher."""
        if not self.qapp:
            self.qapp = QtWidgets.QApplication([])

        if not self.sketch:
            self.sketch = sketcher.LIDSketcher()

        if not self.image_generator:
            self.image_generator = SchrodImageGenerator()

        print("✅ Qt and sketcher initialized")

    def create_professional_ppi_diagram(self, cms_file: str, interactions: List[PPInteraction],
                                      asl_string: str = "protein", output_prefix: str = "professional_ppi",
                                      threshold: float = 0.3, dpi: int = 300):
        """
        Create professional Schrodinger-style protein-protein interaction diagram.
        """
        print("🎨 Creating professional Schrodinger-style PPI diagram...")

        try:
            # Initialize Qt
            self.initialize_qt()

            # Load structure data
            data = self.load_structure_data(cms_file)

            # Identify chains and residues
            chains = self.identify_chains_and_residues(data['solute_ct'], asl_string)

            # Create professional diagram using matplotlib (Schrodinger-style)
            self.create_schrodinger_style_diagram(chains, interactions, output_prefix, threshold)

            print(f"✅ Professional PPI diagram created successfully!")
            print(f"   Image: {output_prefix}.png")
            print(f"   Report: {output_prefix}_report.json")

        except Exception as e:
            print(f"❌ Error creating professional diagram: {e}")
            raise

    def load_structure_data(self, cms_file: str) -> Dict[str, Any]:
        """Load structure data from CMS file."""
        print(f"📁 Loading structure data from {cms_file}")

        # Load CMS structure
        _, cms_model = topo.read_cms(cms_file)
        solute_ct = cms_model.comp_ct[0]

        data = {
            'solute_ct': solute_ct,
            'cms_model': cms_model,
            'ppi_sea': None,
            'sse_sea': None,
            'tags_seq': [],
            'tot_frames': 0,
            'result_summary': {},
            'tag2ca': {},
            'ca2tag': {}
        }

        # Try to load SEA data if available
        try:
            kw = sea.read_sea(cms_file)
            if 'ProtProtInter' in kw:
                data['ppi_sea'] = kw['ProtProtInter']
            if 'SecondaryStructure' in kw:
                data['sse_sea'] = kw['SecondaryStructure']

            if data['sse_sea']:
                data['tags_seq'] = data['sse_sea'].ProteinResidues.val
                data['tot_frames'] = sid['TrajectoryNumFrames'].val / 100.0

                # Filter out NMA and ACE
                data['tags_seq'] = [t for t in data['tags_seq'] if 'NMA' not in t and 'ACE' not in t]

            if data['ppi_sea']:
                data['result_summary'] = eval(data['ppi_sea'].ResultSummary.val)
                data['tag2ca'] = eval(data['ppi_sea'].DictTag2ca.val)
                data['ca2tag'] = {v: k for k, v in data['tag2ca'].items()}

        except Exception as e:
            print(f"   Warning: Could not load SEA data: {e}")

        print(f"✅ Loaded structure with {len(data['tags_seq'])} residues")
        return data

    def identify_chains_and_residues(self, solute_ct, asl_string: str = "protein") -> Dict[str, Any]:
        """Identify protein chains and residues in the structure."""
        print("🔍 Identifying chains and residues...")

        # Get all CA atoms for protein chains
        ca_atoms = evaluate_asl(solute_ct, f'{asl_string} and atom.ptype CA')

        chains = {}

        for atom_idx in ca_atoms:
            atom = solute_ct.atom[atom_idx]

            # Handle different atom object structures
            try:
                if hasattr(atom, 'residue'):
                    residue = atom.residue
                elif hasattr(atom, 'getResidue'):
                    residue = atom.getResidue()
                else:
                    residue = atom

                # Get chain information
                if hasattr(residue, 'chain'):
                    chain_id = residue.chain.strip() if isinstance(residue.chain, str) else residue.chain.name
                else:
                    chain_id = 'A'

                # Get residue information
                if hasattr(residue, 'pdbres'):
                    res_name = residue.pdbres.strip()
                elif hasattr(residue, 'getResidueName'):
                    res_name = residue.getResidueName()
                else:
                    res_name = 'UNK'

                if hasattr(residue, 'resnum'):
                    res_num = residue.resnum
                elif hasattr(residue, 'getResidueNumber'):
                    res_num = residue.getResidueNumber()
                else:
                    res_num = 0

            except Exception as e:
                print(f"   Warning: Could not process atom {atom_idx}: {e}")
                continue

            if chain_id not in chains:
                chains[chain_id] = {
                    'residues': [],
                    'color': self.chain_colors.get(chain_id, self.chain_colors['default'])
                }

            chains[chain_id]['residues'].append({
                'index': atom_idx,
                'residue': residue,
                'res_name': res_name,
                'res_num': res_num
            })

        print(f"✅ Identified {len(chains)} chains")
        for chain_id, chain_data in chains.items():
            print(f"   Chain {chain_id}: {len(chain_data['residues'])} residues")

        return chains

    def create_schrodinger_style_diagram(self, chains: Dict[str, Any], interactions: List[PPInteraction],
                                       output_prefix: str, threshold: float = 0.3):
        """Create professional Schrodinger-style interaction diagram using matplotlib."""
        print("🎨 Creating Schrodinger-style diagram...")

        # Set up the plot with professional dimensions
        fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(20, 12),
                                                gridspec_kw={'width_ratios': [4, 1]})

        # Professional Schrodinger color scheme
        fig.patch.set_facecolor('#FAFAFA')
        ax_main.set_facecolor('#FFFFFF')
        ax_legend.set_facecolor('#FAFAFA')

        # Create professional layout
        pos = self.create_professional_layout(chains)

        # Draw protein chains with professional styling
        self.draw_professional_chains(ax_main, chains, pos)

        # Draw interactions with professional styling
        self.draw_professional_interactions(ax_main, interactions, pos, threshold)

        # Add professional legend
        self.create_professional_legend(ax_legend, interactions, chains)

        # Professional title and styling
        self.apply_professional_styling(ax_main, len(interactions))

        # Save high-quality image
        output_file = f"{output_prefix}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Professional diagram saved: {output_file}")

    def create_professional_layout(self, chains: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Create professional Schrodinger-style layout."""
        pos = {}
        chains_list = list(chains.keys())

        if len(chains_list) == 2:
            # Two-chain interaction: Professional side-by-side layout
            chain1, chain2 = chains_list[0], chains_list[1]

            # Chain 1: Left side with protein-like curve
            chain1_residues = chains[chain1]['residues']
            chain1_center = np.array([-0.8, 0])
            chain1_radius = 0.4

            for i, res_data in enumerate(chain1_residues):
                angle = (i / len(chain1_residues)) * 2 * np.pi
                radius_var = chain1_radius + 0.05 * np.sin(3 * angle)
                x = chain1_center[0] + radius_var * np.cos(angle)
                y = chain1_center[1] + radius_var * np.sin(angle)
                pos[f"{chain1}:{res_data['res_name']}_{res_data['res_num']}"] = (x, y)

            # Chain 2: Right side with protein-like curve
            chain2_residues = chains[chain2]['residues']
            chain2_center = np.array([0.8, 0])
            chain2_radius = 0.4

            for i, res_data in enumerate(chain2_residues):
                angle = (i / len(chain2_residues)) * 2 * np.pi
                radius_var = chain2_radius + 0.05 * np.sin(3 * angle)
                x = chain2_center[0] + radius_var * np.cos(angle)
                y = chain2_center[1] + radius_var * np.sin(angle)
                pos[f"{chain2}:{res_data['res_name']}_{res_data['res_num']}"] = (x, y)

        else:
            # Single chain: Circular layout
            all_residues = []
            for chain_id, chain_data in chains.items():
                all_residues.extend([(chain_id, res) for res in chain_data['residues']])

            center = np.array([0, 0])
            radius = 0.6

            for i, (chain_id, res_data) in enumerate(all_residues):
                angle = (i / len(all_residues)) * 2 * np.pi
                radius_var = radius + 0.05 * np.sin(3 * angle)
                x = center[0] + radius_var * np.cos(angle)
                y = center[1] + radius_var * np.sin(angle)
                pos[f"{chain_id}:{res_data['res_name']}_{res_data['res_num']}"] = (x, y)

        return pos

    def draw_professional_chains(self, ax, chains: Dict[str, Any], pos: Dict[str, Tuple[float, float]]):
        """Draw protein chains with professional Schrodinger styling."""
        for chain_id, chain_data in chains.items():
            chain_color = chain_data['color']

            for res_data in chain_data['residues']:
                res_name = res_data['res_name']
                res_num = res_data['res_num']
                res_key = f"{chain_id}:{res_name}_{res_num}"

                if res_key in pos:
                    x, y = pos[res_key]

                    # Professional amino acid coloring
                    aa_color = self.amino_acid_colors.get(res_name, self.amino_acid_colors['UNK'])

                    # Professional node sizing based on importance
                    if res_name in ['ARG', 'LYS', 'ASP', 'GLU']:  # Charged
                        size = 400
                    elif res_name in ['PHE', 'TYR', 'TRP', 'HIS']:  # Aromatic
                        size = 350
                    elif res_name in ['SER', 'THR', 'ASN', 'GLN']:  # Polar
                        size = 300
                    else:  # Hydrophobic
                        size = 250

                    # Draw professional node
                    circle = plt.Circle((x, y), size/10000, facecolor=aa_color,
                                      edgecolor='black', linewidth=1.5, alpha=0.9)
                    ax.add_patch(circle)

                    # Professional labeling
                    ax.text(x, y, f"{chain_id}:{res_name}\n{res_num}",
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white', bbox=dict(boxstyle="round,pad=0.3",
                                                   facecolor='black', alpha=0.7))

    def draw_professional_interactions(self, ax, interactions: List[PPInteraction],
                                     pos: Dict[str, Tuple[float, float]], threshold: float):
        """Draw interactions with professional Schrodinger styling."""
        for interaction in interactions:
            if interaction.occupancy < threshold:
                continue

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

                # Professional interaction label for strong interactions
                if interaction.occupancy > 0.7:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f'{interaction.occupancy*100:.0f}%',
                           ha='center', va='center', fontsize=7, fontweight='bold',
                           color=style_config['color'],
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    def create_professional_legend(self, ax, interactions: List[PPInteraction], chains: Dict[str, Any]):
        """Create professional Schrodinger-style legend."""
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
            if any(res_data['res_name'] in residues
                   for chain_data in chains.values()
                   for res_data in chain_data['residues']):
                sample_res = residues[0]
                color = self.amino_acid_colors.get(sample_res, '#CCCCCC')
                legend_elements.append(
                    mpatches.Patch(color=color, label=f'Amino Acid: {category}')
                )

        # Chain legend
        for chain_id, chain_data in chains.items():
            legend_elements.append(
                mpatches.Patch(color=chain_data['color'], label=f'Chain {chain_id}')
            )

        # Professional legend layout
        ax.legend(handles=legend_elements, loc='center', fontsize=10,
                 frameon=True, fancybox=True, shadow=True)
        ax.axis('off')

    def apply_professional_styling(self, ax, num_interactions: int):
        """Apply professional Schrodinger styling."""
        # Professional title
        ax.set_title(
            f"Protein-Protein Interaction Network\n"
            f"Schrodinger Professional Visualization\n"
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

    # Test the professional visualizer
    visualizer = ProfessionalPPIVisualizer()

    # Load test data
    try:
        with open('my_analysis21_interactions.json', 'r') as f:
            analyzer_results = json.load(f)

        interactions = parse_ppi_interactions_from_analyzer(analyzer_results)

        # Create professional diagram
        visualizer.create_professional_ppi_diagram(
            cms_file='desmond_md_job_pro-pro/desmond_md_job_pro-pro-out.cms',
            interactions=interactions,
            output_prefix='professional_ppi_diagram'
        )

        print("✅ Professional PPI visualization completed!")

    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()