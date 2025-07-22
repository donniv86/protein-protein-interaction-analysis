#!/usr/bin/env python3
"""
True Schrodinger-style Protein-Protein Interaction Visualization
Uses LIDSketcher with water droplet residues and professional styling.

Features:
- Water droplet-style residues (like Schrodinger's ligand interaction diagrams)
- Professional amino acid labels with chain information
- Proper non-bonding interaction visualization
- Schrodinger's native styling and layout
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

class SchrodingerStylePPIVisualizer:
    """
    True Schrodinger-style protein-protein interaction visualizer.
    Uses LIDSketcher with water droplet residues and professional styling.
    """

    def __init__(self):
        """Initialize the Schrodinger-style PPI visualizer."""
        self.qapp = None
        self.sketch = None
        self.image_generator = None
        self.residues = {}
        self.interactions = []

        # Schrodinger's professional interaction styling
        self.interaction_styles = {
            'hydrogen_bond': {
                'style_key': 'hbond_ss',  # Schrodinger hydrogen bond style
                'color': (0.0, 0.7, 1.0),  # Light blue
                'line_style': 'solid',
                'line_width': 2.0,
                'label': 'Hydrogen Bond'
            },
            'salt_bridge': {
                'style_key': 'salt-bridge',  # Schrodinger salt bridge style
                'color': (1.0, 0.5, 0.0),  # Orange
                'line_style': 'dashed',
                'line_width': 3.0,
                'label': 'Salt Bridge'
            },
            'pi_pi': {
                'style_key': 'pi-pi',  # Schrodinger pi-pi style
                'color': (0.8, 0.0, 0.8),  # Purple
                'line_style': 'dotted',
                'line_width': 2.5,
                'label': 'π-π Stacking'
            },
            'pi_cation': {
                'style_key': 'pi-cat',  # Schrodinger pi-cation style
                'color': (0.0, 0.8, 0.0),  # Green
                'line_style': 'solid',
                'line_width': 2.0,
                'label': 'π-Cation'
            },
            'hydrophobic': {
                'style_key': 'hbond_self',  # Schrodinger hydrophobic style
                'color': (0.5, 0.5, 0.5),  # Gray
                'line_style': 'solid',
                'line_width': 1.5,
                'label': 'Hydrophobic'
            },
            'disulfide': {
                'style_key': 'hbond_bb',  # Schrodinger disulfide style
                'color': (1.0, 0.0, 0.0),  # Red
                'line_style': 'solid',
                'line_width': 2.5,
                'label': 'Disulfide'
            }
        }

        # Schrodinger's professional chain color scheme
        self.chain_colors = {
            'A': (1.0, 0.0, 0.0),    # Red
            'B': (0.0, 1.0, 0.0),    # Green
            'C': (0.0, 0.0, 1.0),    # Blue
            'D': (1.0, 1.0, 0.0),    # Yellow
            'E': (1.0, 0.0, 1.0),    # Magenta
            'F': (0.0, 1.0, 1.0),    # Cyan
            'L': (1.0, 0.5, 0.0),    # Orange
            'default': (0.7, 0.7, 0.7)  # Light gray
        }

        # Schrodinger's professional amino acid color scheme
        self.amino_acid_colors = {
            # Hydrophobic (Non-polar) - Cool blues and grays
            'ALA': (0.53, 0.81, 0.92),  # Sky blue
            'VAL': (0.27, 0.51, 0.71),  # Steel blue
            'LEU': (0.12, 0.56, 1.0),   # Dodger blue
            'ILE': (0.0, 0.0, 0.8),     # Medium blue
            'MET': (0.25, 0.41, 0.88),  # Royal blue
            'PHE': (0.10, 0.10, 0.44),  # Midnight blue
            'TRP': (0.0, 0.0, 0.5),     # Navy blue
            'PRO': (0.28, 0.24, 0.55),  # Dark slate blue

            # Hydrophilic (Polar) - Soft greens and teals
            'GLY': (0.60, 0.98, 0.60),  # Pale green
            'SER': (0.56, 0.93, 0.56),  # Light green
            'THR': (0.20, 0.80, 0.20),  # Lime green
            'CYS': (0.13, 0.55, 0.13),  # Forest green
            'TYR': (0.0, 0.39, 0.0),    # Dark green
            'ASN': (0.13, 0.70, 0.67),  # Light sea green
            'GLN': (0.28, 0.82, 0.80),  # Medium turquoise

            # Charged (Acidic) - Warm reds and pinks
            'ASP': (1.0, 0.71, 0.76),   # Light pink
            'GLU': (1.0, 0.41, 0.71),   # Hot pink

            # Charged (Basic) - Cool purples and blues
            'LYS': (0.58, 0.44, 0.86),  # Medium purple
            'ARG': (0.54, 0.17, 0.89),  # Blue violet
            'HIS': (0.29, 0.0, 0.51),   # Indigo

            # Special cases
            'UNK': (0.75, 0.75, 0.75),  # Light gray
        }

        print("🎨 Schrodinger-style PPI Visualizer Initialized")

    def initialize_qt(self):
        """Initialize Qt application and sketcher."""
        if not self.qapp:
            self.qapp = QtWidgets.QApplication([])

        if not self.sketch:
            self.sketch = sketcher.LIDSketcher()

        if not self.image_generator:
            self.image_generator = SchrodImageGenerator()

        print("✅ Qt and sketcher initialized")

    def create_schrodinger_style_diagram(self, interactions: List[PPInteraction],
                                       output_prefix: str = "schrodinger_ppi",
                                       threshold: float = 0.3, dpi: int = 300):
        """
        Create true Schrodinger-style protein-protein interaction diagram.
        Uses LIDSketcher with water droplet residues and professional styling.
        """
        print("🎨 Creating true Schrodinger-style PPI diagram...")

        try:
            # Initialize Qt
            self.initialize_qt()

            # Filter interactions by threshold
            filtered_interactions = [i for i in interactions if i.occupancy >= threshold]
            print(f"   Showing {len(filtered_interactions)} interactions (threshold: {threshold})")

            # Extract unique residues from interactions
            unique_residues = self.extract_unique_residues(filtered_interactions)
            print(f"   Found {len(unique_residues)} unique interacting residues")

            # Add water droplet residues to sketcher
            self.add_water_droplet_residues(unique_residues)

            # Add professional interactions
            self.add_schrodinger_interactions(filtered_interactions)

            # Apply professional layout
            self.apply_schrodinger_layout(unique_residues)

            # Generate high-quality image
            output_file = f"{output_prefix}.png"
            self.generate_schrodinger_image(output_file, dpi=dpi)

            print(f"✅ Schrodinger-style diagram created successfully!")
            print(f"   Image: {output_file}")

        except Exception as e:
            print(f"❌ Error creating Schrodinger-style diagram: {e}")
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

    def add_water_droplet_residues(self, unique_residues: Dict[str, PPResData]):
        """Add water droplet-style residues to sketcher with professional styling."""
        print("💧 Adding water droplet residues to sketcher...")

        for res_key, res_data in unique_residues.items():
            # Create clean, professional residue label (like Schrodinger's style)
            label = f"{res_data.res_name}{res_data.res_num}"  # Clean label without chain prefix

            # Get Schrodinger's professional amino acid color
            aa_color = self.amino_acid_colors.get(res_data.res_name, self.amino_acid_colors['UNK'])
            red, green, blue = aa_color

            # Add water droplet residue to sketcher (Schrodinger's native method)
            residue_obj = self.sketch.addResidue(
                label,                # clean residue label
                res_data.res_num,     # residue number
                0, 0,                 # x, y (let layout handle positioning)
                int(red * 255),       # red component
                int(green * 255),     # green component
                int(blue * 255),      # blue component
                res_key               # unique identifier
            )

            # Store reference for later positioning
            self.residues[res_key] = residue_obj

        print(f"✅ Added {len(self.residues)} water droplet residues")

    def add_schrodinger_interactions(self, interactions: List[PPInteraction]):
        """Add professional Schrodinger-style interactions."""
        print("🔗 Adding Schrodinger-style interactions...")

        interaction_count = 0

        for interaction in interactions:
            res1_key = f"{interaction.res1.chain}:{interaction.res1.res_name}_{interaction.res1.res_num}"
            res2_key = f"{interaction.res2.chain}:{interaction.res2.res_name}_{interaction.res2.res_num}"

            # Check if both residues exist in sketcher
            if res1_key not in self.residues or res2_key not in self.residues:
                continue

            # Skip self-interactions
            if res1_key == res2_key:
                continue

            try:
                # Add Schrodinger-style interaction
                interaction_obj = self.sketch.addResidueInteraction(
                    self.residues[res1_key],
                    self.residues[res2_key]
                )

                # Apply Schrodinger's professional styling
                style_config = self.interaction_styles.get(interaction.interaction_type,
                                                         self.interaction_styles['hydrogen_bond'])

                # Set Schrodinger's native style key
                if hasattr(interaction_obj, 'setStyleKey'):
                    interaction_obj.setStyleKey(style_config['style_key'])

                # Set professional opacity based on occupancy
                opacity = min(0.9, max(0.3, interaction.occupancy))
                if hasattr(interaction_obj, 'setFloatProperty'):
                    interaction_obj.setFloatProperty("opacity_value", opacity)

                # Add professional interaction label
                if interaction.occupancy > 0.7:
                    label = f'{interaction.occupancy*100:.0f}%'
                    if hasattr(interaction_obj, 'setCustomLabel'):
                        interaction_obj.setCustomLabel(label)

                interaction_count += 1

            except Exception as e:
                print(f"⚠️ Error adding interaction between {res1_key} and {res2_key}: {e}")

        print(f"✅ Added {interaction_count} Schrodinger-style interactions")

    def apply_schrodinger_layout(self, unique_residues: Dict[str, PPResData]):
        """Apply professional Schrodinger-style layout with proper spacing."""
        print("🎨 Applying professional Schrodinger-style layout...")

        # Separate residues by chain
        chain_residues = defaultdict(list)
        for res_key, res_data in unique_residues.items():
            chain_residues[res_data.chain].append((res_key, res_data))

        chains_list = list(chain_residues.keys())

        if len(chains_list) == 2:
            # Two-chain interaction: Clean side-by-side layout like Schrodinger
            chain1, chain2 = chains_list[0], chains_list[1]

            # Chain 1: Left side - Clean vertical arrangement
            chain1_residues = chain_residues[chain1]
            chain1_x = -1.2  # Further apart for cleaner look

            for i, (res_key, res_data) in enumerate(chain1_residues):
                # Clean vertical spacing with proper gaps
                y = 0.8 - (i * 0.3)  # Start from top, go down with proper spacing

                # Position residue in sketcher
                if res_key in self.residues:
                    residue_obj = self.residues[res_key]
                    if hasattr(residue_obj, 'setPosition'):
                        residue_obj.setPosition(QPointF(float(chain1_x), float(y)))

            # Chain 2: Right side - Clean vertical arrangement
            chain2_residues = chain_residues[chain2]
            chain2_x = 1.2  # Further apart for cleaner look

            for i, (res_key, res_data) in enumerate(chain2_residues):
                # Clean vertical spacing with proper gaps
                y = 0.8 - (i * 0.3)  # Start from top, go down with proper spacing

                # Position residue in sketcher
                if res_key in self.residues:
                    residue_obj = self.residues[res_key]
                    if hasattr(residue_obj, 'setPosition'):
                        residue_obj.setPosition(QPointF(float(chain2_x), float(y)))

        else:
            # Single chain: Clean horizontal arrangement
            all_residues = list(unique_residues.items())

            for i, (res_key, res_data) in enumerate(all_residues):
                # Clean horizontal spacing
                x = -1.0 + (i * 0.4)  # Start from left, go right with proper spacing
                y = 0.0  # Center vertically

                # Position residue in sketcher
                if res_key in self.residues:
                    residue_obj = self.residues[res_key]
                    if hasattr(residue_obj, 'setPosition'):
                        residue_obj.setPosition(QPointF(float(x), float(y)))

        print("✅ Professional Schrodinger-style layout applied")

    def generate_schrodinger_image(self, output_file: str, format: str = 'png', dpi: int = 300):
        """Generate high-quality Schrodinger-style image."""
        print(f"🎨 Generating Schrodinger-style image: {output_file}")

        try:
            # Export image using Schrodinger's native method
            qimage = self.sketch.getQImage()
            qimage.save(output_file, format.upper())

            print(f"✅ Schrodinger-style image saved: {output_file}")

        except Exception as e:
            print(f"❌ Error generating Schrodinger-style image: {e}")
            raise

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

    # Test the Schrodinger-style visualizer
    visualizer = SchrodingerStylePPIVisualizer()

    # Load test data
    try:
        with open('my_analysis21_interactions.json', 'r') as f:
            analyzer_results = json.load(f)

        # Limit to first 10 interactions for testing
        limited_results = analyzer_results[:10]
        interactions = parse_ppi_interactions_from_analyzer(limited_results)

        # Create Schrodinger-style diagram
        visualizer.create_schrodinger_style_diagram(
            interactions=interactions,
            output_prefix='schrodinger_ppi_10_interactions'
        )

        print("✅ Schrodinger-style PPI visualization completed!")

    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()