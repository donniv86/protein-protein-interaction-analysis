#!/usr/bin/env python3
"""
Advanced Protein-Protein Interaction Visualization Module
Adapts protein-ligand interaction patterns for protein-protein interactions.

Features:
- Professional interaction diagrams with Schrodinger's native capabilities
- Advanced styling and layout options
- Interactive visualization panels
- Comprehensive interaction analysis
- Publication-quality output

Based on patterns from:
- pl_interactions.py (protein-ligand interaction analysis)
- analysis.py (Schrodinger's core analysis framework)
- event_analysis.py (professional visualization)
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
    # evaluate_asl is already imported from schrodinger.structutils.analyze above
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
        """Return the full name in format 'A:THR_45:O'."""
        atom_part = f":{self.interaction_atom}" if self.interaction_atom else ""
        return f"{self.chain}:{self.res_name}_{self.res_num}{atom_part}"

    def __str__(self) -> str:
        return self.fullName()

    def __repr__(self) -> str:
        return f"PPResData({self.fullName()})"

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

class AdvancedPPIVisualizer:
    """
    Advanced protein-protein interaction visualizer using Schrodinger's capabilities.
    """

    def __init__(self):
        """Initialize the advanced PPI visualizer."""
        self.qapp = None
        self.sketch = None
        self.image_generator = None
        self.residues = {}
        self.interactions = []

        # Advanced styling options for protein-protein interactions
        self.style_config = {
            'hydrogen_bond': {
                'style_key': 'hbond_ss',
                'color': (0.0, 0.7, 1.0),  # Light blue
                'line_style': 'solid',
                'line_width': 2.0,
                'label': 'Hydrogen Bond'
            },
            'salt_bridge': {
                'style_key': 'salt-bridge',
                'color': (1.0, 0.5, 0.0),  # Orange
                'line_style': 'dashed',
                'line_width': 3.0,
                'label': 'Salt Bridge'
            },
            'pi_pi': {
                'style_key': 'pi-pi',
                'color': (0.8, 0.0, 0.8),  # Purple
                'line_style': 'dotted',
                'line_width': 2.5,
                'label': 'π-π Stacking'
            },
            'pi_cation': {
                'style_key': 'pi-cat',
                'color': (0.0, 0.8, 0.0),  # Green
                'line_style': 'solid',
                'line_width': 2.0,
                'label': 'π-Cation'
            },
            'hydrophobic': {
                'style_key': 'hbond_self',
                'color': (0.5, 0.5, 0.5),  # Gray
                'line_style': 'solid',
                'line_width': 1.5,
                'label': 'Hydrophobic'
            },
            'disulfide': {
                'style_key': 'hbond_bb',
                'color': (1.0, 0.0, 0.0),  # Red
                'line_style': 'solid',
                'line_width': 2.5,
                'label': 'Disulfide'
            }
        }

        # Chain color scheme
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

        # Amino acid color scheme (based on Schrodinger standards)
        self.amino_acid_colors = {
            # Hydrophobic (Non-polar)
            'ALA': (0.53, 0.81, 0.92),  # Sky blue
            'VAL': (0.27, 0.51, 0.71),  # Steel blue
            'LEU': (0.12, 0.56, 1.0),   # Dodger blue
            'ILE': (0.0, 0.0, 0.8),     # Medium blue
            'MET': (0.25, 0.41, 0.88),  # Royal blue
            'PHE': (0.10, 0.10, 0.44),  # Midnight blue
            'TRP': (0.0, 0.0, 0.5),     # Navy blue
            'PRO': (0.28, 0.24, 0.55),  # Dark slate blue

            # Hydrophilic (Polar)
            'GLY': (0.60, 0.98, 0.60),  # Pale green
            'SER': (0.56, 0.93, 0.56),  # Light green
            'THR': (0.20, 0.80, 0.20),  # Lime green
            'CYS': (0.13, 0.55, 0.13),  # Forest green
            'TYR': (0.0, 0.39, 0.0),    # Dark green
            'ASN': (0.13, 0.70, 0.67),  # Light sea green
            'GLN': (0.28, 0.82, 0.80),  # Medium turquoise

            # Charged (Acidic)
            'ASP': (1.0, 0.71, 0.76),   # Light pink
            'GLU': (1.0, 0.41, 0.71),   # Hot pink

            # Charged (Basic)
            'LYS': (0.58, 0.44, 0.86),  # Medium purple
            'ARG': (0.54, 0.17, 0.89),  # Blue violet
            'HIS': (0.29, 0.0, 0.51),   # Indigo

            # Special cases
            'UNK': (0.75, 0.75, 0.75),  # Silver
        }

    def initialize_qt(self):
        """Initialize Qt application and sketcher."""
        if not self.qapp:
            self.qapp = QtWidgets.QApplication([])

        if not self.sketch:
            self.sketch = sketcher.LIDSketcher()

        if not self.image_generator:
            self.image_generator = SchrodImageGenerator()

        print("✅ Qt and sketcher initialized")

    def load_structure_data(self, cms_file: str, eaf_file: str = None) -> Dict[str, Any]:
        """
        Load structure data from CMS and optionally EAF files.

        Args:
            cms_file: Path to Desmond CMS file
            eaf_file: Path to EAF file from event analysis (optional)

        Returns:
            Dictionary containing loaded structure data
        """
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

        # Load EAF data if provided
        if eaf_file and os.path.exists(eaf_file):
            print(f"📁 Loading EAF data from {eaf_file}")
            sid = sea.Map(open(eaf_file).read())

            # Extract keywords from EAF
            for kw in sid.Keywords:
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

        print(f"✅ Loaded structure with {len(data['tags_seq'])} residues")
        return data

    def identify_chains_and_residues(self, solute_ct, asl_string: str = "protein") -> Dict[str, Any]:
        """
        Identify protein chains and residues in the structure.

        Args:
            solute_ct: Solute structure
            asl_string: ASL string for selection

        Returns:
            Dictionary containing chain and residue information
        """
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
                    # Try to get residue info from atom properties
                    residue = atom

                # Get chain information
                if hasattr(residue, 'chain'):
                    chain_id = residue.chain.strip() if isinstance(residue.chain, str) else residue.chain.name
                else:
                    chain_id = 'A'  # Default chain

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

    def add_protein_residues_to_sketch(self, chains: Dict[str, Any], solute_ct):
        """Add protein residues to sketch with professional styling."""
        print("🎨 Adding protein residues to sketch...")

        for chain_id, chain_data in chains.items():
            chain_color = chain_data['color']

            for res_data in chain_data['residues']:
                residue = res_data['residue']
                res_name = res_data['res_name']
                res_num = res_data['res_num']

                # Create residue label
                label = f"{res_name}{res_num}"

                # Get color for amino acid
                aa_color = self.amino_acid_colors.get(res_name, self.amino_acid_colors['UNK'])
                red, green, blue = aa_color

                # Add residue to sketch using positional arguments
                residue_obj = self.sketch.addResidue(
                    label,                # residue label
                    res_num,               # residue number
                    0, 0,                  # x, y (let layout handle)
                    int(red * 255),        # red
                    int(green * 255),      # green
                    int(blue * 255),       # blue
                    f"{chain_id}:{res_name}_{res_num}"  # uid
                    # x3d, y3d, z3d use defaults
                )

                # Store reference
                tag = f"{chain_id}:{res_name}_{res_num}"
                self.residues[tag] = residue_obj

        print(f"✅ Added {len(self.residues)} residues to sketch")

    def add_interactions_to_sketch(self, interactions: List[PPInteraction],
                                  threshold: float = 0.3):
        """Add interactions to sketch with professional styling."""
        print("🔗 Adding interactions to sketch...")

        interaction_count = 0

        for interaction in interactions:
            # Apply occupancy threshold
            if interaction.occupancy < threshold:
                continue

            res1_tag = f"{interaction.res1.chain}:{interaction.res1.res_name}_{interaction.res1.res_num}"
            res2_tag = f"{interaction.res2.chain}:{interaction.res2.res_name}_{interaction.res2.res_num}"

            # Check if both residues exist in sketch
            if res1_tag not in self.residues or res2_tag not in self.residues:
                continue

            # Skip self-interactions
            if res1_tag == res2_tag:
                continue

            try:
                # Add interaction
                interaction_obj = self.sketch.addResidueInteraction(
                    self.residues[res1_tag],
                    self.residues[res2_tag]
                )

                # Set style based on interaction type
                style_config = self.style_config.get(interaction.interaction_type,
                                                   self.style_config['hydrogen_bond'])

                if hasattr(interaction_obj, 'setStyleKey'):
                    interaction_obj.setStyleKey(style_config['style_key'])

                # Set opacity based on occupancy
                opacity = min(0.9, max(0.3, interaction.occupancy))
                if hasattr(interaction_obj, 'setFloatProperty'):
                    interaction_obj.setFloatProperty("opacity_value", opacity)

                # Add label for strong interactions
                if interaction.occupancy > 0.7:
                    label = f'{interaction.occupancy*100:.0f}%'
                    if hasattr(interaction_obj, 'setCustomLabel'):
                        interaction_obj.setCustomLabel(label)

                interaction_count += 1

            except Exception as e:
                print(f"⚠️ Error adding interaction between {res1_tag} and {res2_tag}: {e}")

        print(f"✅ Added {interaction_count} interactions to sketch")

    def apply_advanced_layout(self):
        """Apply advanced layout for protein-protein interaction visualization."""
        print("🎨 Applying advanced layout...")

        # Get unique chains
        chains = set()
        for tag in self.residues.keys():
            chain = tag.split(':')[0]
            chains.add(chain)

        chains_list = list(chains)

        if len(chains_list) == 2:
            # Two-chain interaction: Create proper outlines for each chain
            chain1, chain2 = chains_list[0], chains_list[1]

            # Separate nodes by chain
            chain1_nodes = [tag for tag in self.residues.keys() if tag.startswith(f"{chain1}:")]
            chain2_nodes = [tag for tag in self.residues.keys() if tag.startswith(f"{chain2}:")]

            # Sort nodes by residue number
            chain1_nodes.sort(key=lambda x: int(x.split('_')[1]))
            chain2_nodes.sort(key=lambda x: int(x.split('_')[1]))

            # Apply protein-like arrangement
            self._arrange_chain_residues(chain1_nodes, (-0.6, 0), 0.3)
            self._arrange_chain_residues(chain2_nodes, (0.6, 0), 0.3)

        elif len(chains_list) == 1:
            # Single chain: Use circular layout
            nodes = list(self.residues.keys())
            self._arrange_chain_residues(nodes, (0, 0), 0.5)

        print("✅ Advanced layout applied")

    def _arrange_chain_residues(self, residue_tags: List[str], center: Tuple[float, float], radius: float):
        """Arrange residues in a protein-like curve."""
        center_x, center_y = center

        for i, tag in enumerate(residue_tags):
            # Arrange in a protein-like curve (helix-like arrangement)
            angle = (i / len(residue_tags)) * 2 * np.pi
            # Add some variation to make it look more like a protein structure
            radius_var = radius + 0.05 * np.sin(3 * angle)
            x = center_x + radius_var * np.cos(angle)
            y = center_y + radius_var * np.sin(angle)

            # Set position for residue (if supported by sketcher)
            if tag in self.residues:
                residue_obj = self.residues[tag]
                if hasattr(residue_obj, 'setPosition'):
                    residue_obj.setPosition(QPointF(float(x), float(y)))

    def generate_high_quality_image(self, output_file: str, format: str = 'png',
                                   dpi: int = 300, size: Tuple[int, int] = (1200, 800)):
        """Generate high-quality interaction diagram."""
        print(f"🎨 Generating high-quality image: {output_file}")

        try:
            # Apply advanced layout
            self.apply_advanced_layout()

            # Export image using Qt QImage
            qimage = self.sketch.getQImage()
            qimage.save(output_file, format.upper())

            print(f"✅ High-quality image saved: {output_file}")

        except Exception as e:
            print(f"❌ Error generating image: {e}")
            raise

    def generate_interaction_report(self, interactions: List[PPInteraction], output_file: str):
        """Generate comprehensive interaction report."""
        print(f"📊 Generating interaction report: {output_file}")

        # Group interactions by type
        interaction_stats = defaultdict(list)
        for interaction in interactions:
            interaction_stats[interaction.interaction_type].append(interaction)

        # Generate report
        report_data = {
            'summary': {
                'total_interactions': len(interactions),
                'interaction_types': {}
            },
            'detailed_interactions': []
        }

        for interaction_type, type_interactions in interaction_stats.items():
            avg_occupancy = np.mean([i.occupancy for i in type_interactions])
            report_data['summary']['interaction_types'][interaction_type] = {
                'count': len(type_interactions),
                'average_occupancy': avg_occupancy
            }

        for interaction in interactions:
            report_data['detailed_interactions'].append({
                'residue1': interaction.res1.fullName(),
                'residue2': interaction.res2.fullName(),
                'type': interaction.interaction_type,
                'occupancy': interaction.occupancy,
                'distance': interaction.distance
            })

        # Save report
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"✅ Interaction report saved: {output_file}")

    def create_protein_protein_diagram(self, cms_file: str, interactions: List[PPInteraction],
                                     asl_string: str = "protein", output_prefix: str = "ppi_diagram",
                                     threshold: float = 0.3, dpi: int = 300):
        """
        Create comprehensive protein-protein interaction diagram.

        Args:
            cms_file: Path to CMS file
            interactions: List of PPInteraction objects
            asl_string: ASL string for protein selection
            output_prefix: Output file prefix
            threshold: Minimum occupancy threshold
            dpi: Image resolution
        """
        print("🎨 Creating protein-protein interaction diagram...")

        try:
            # Initialize Qt
            self.initialize_qt()

            # Load structure data
            data = self.load_structure_data(cms_file)

            # Identify chains and residues
            chains = self.identify_chains_and_residues(data['solute_ct'], asl_string)

            # Add residues to sketch
            self.add_protein_residues_to_sketch(chains, data['solute_ct'])

            # Add interactions to sketch
            self.add_interactions_to_sketch(interactions, threshold)

            # Generate outputs
            image_file = f"{output_prefix}.png"
            report_file = f"{output_prefix}_report.json"

            self.generate_high_quality_image(image_file, dpi=dpi)
            self.generate_interaction_report(interactions, report_file)

            print(f"✅ Protein-protein interaction diagram created successfully!")
            print(f"   Image: {image_file}")
            print(f"   Report: {report_file}")

        except Exception as e:
            print(f"❌ Error creating diagram: {e}")
            raise

def parse_ppi_interactions_from_analyzer(analyzer_results: List[Dict]) -> List[PPInteraction]:
    """
    Parse interaction results from PPI analyzer into PPInteraction objects.

    Args:
        analyzer_results: List of interaction dictionaries from ppi_analyzer.py

    Returns:
        List of PPInteraction objects
    """
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
    """Main function for testing the PPI visualizer."""
    import argparse

    parser = argparse.ArgumentParser(description='Advanced Protein-Protein Interaction Visualizer')
    parser.add_argument('cms', help='Desmond CMS file')
    parser.add_argument('interactions', help='JSON file with interaction data')
    parser.add_argument('--asl', default='protein', help='ASL string for protein selection')
    parser.add_argument('--output', default='ppi_diagram', help='Output file prefix')
    parser.add_argument('--threshold', type=float, default=0.3, help='Minimum occupancy threshold')
    parser.add_argument('--dpi', type=int, default=300, help='Image resolution')

    args = parser.parse_args()

    # Load interaction data
    with open(args.interactions, 'r') as f:
        interaction_data = json.load(f)

    # Parse interactions
    interactions = parse_ppi_interactions_from_analyzer(interaction_data)

    # Create visualizer
    visualizer = AdvancedPPIVisualizer()

    # Create diagram
    visualizer.create_protein_protein_diagram(
        cms_file=args.cms,
        interactions=interactions,
        asl_string=args.asl,
        output_prefix=args.output,
        threshold=args.threshold,
        dpi=args.dpi
    )

if __name__ == '__main__':
    main()