#!/usr/bin/env python3
"""
Single Comprehensive Protein-Protein Interaction Analyzer
Consolidates all working functionality into one clean script.

Features:
- Structure analysis with protein filtering (no waters, ligands, metals)
- Multiple interaction types (H-bonds, salt bridges, π-π, π-cation, hydrophobic)
- Multi-chain analysis with customizable chain groups
- Network visualization with professional styling
- Statistical analysis and reporting
- Optional trajectory analysis (when available)
- Occupancy-based filtering
- Comprehensive output generation

Usage:
    python ppi_analyzer.py --cms structure.cms --traj trajectory_dir --chain-groups A,B C,D
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
warnings.filterwarnings('ignore')

# Schrodinger imports
try:
    from schrodinger.structure import StructureReader
    from schrodinger.structutils import analyze
    from schrodinger.structutils.analyze import evaluate_asl, hbond
    from schrodinger.application.desmond.packages import topo, traj
    from schrodinger.utils import sea
    SCHRODINGER_AVAILABLE = True
    print("✅ Schrodinger modules imported successfully")
except ImportError as e:
    SCHRODINGER_AVAILABLE = False
    print(f"❌ Schrodinger modules not available: {e}")
    sys.exit(1)

class ComprehensivePPIAnalyzer:
    """Comprehensive Protein-Protein Interaction Analyzer with all features."""

    def __init__(self, cms_file, trajectory_file=None, chain_groups=None, occupancy_threshold=0.2):
        """
        Initialize the analyzer.

        Args:
            cms_file (str): Path to the CMS file
            trajectory_file (str): Path to the trajectory file/directory (optional)
            chain_groups (list): List of chain groups to analyze, e.g., [['A', 'B'], ['C', 'D']]
            occupancy_threshold (float): Minimum occupancy threshold for trajectory analysis
        """
        self.cms_file = Path(cms_file)
        self.trajectory_file = Path(trajectory_file) if trajectory_file else None
        self.chain_groups = chain_groups or [['A'], ['L']]  # Default to A vs L analysis
        self.occupancy_threshold = occupancy_threshold

        # Initialize data structures
        self.structure = None
        self.chain_info = {}
        self.interactions = []
        self.residue_info = {}
        self.interaction_occupancy = {}
        self.interaction_distance_history = {}
        self.total_frames_analyzed = 0

        # Temporal analysis data structures
        self.temporal_interactions = {}  # Track interactions over time
        self.interaction_timeline = {}   # Timeline of when interactions appear/disappear
        self.chain_interaction_stats = {}  # Statistics per chain pair
        self.frame_interactions = []     # List of interactions per frame

        # Interaction type definitions with professional styling
        self.interaction_types = {
            'hydrogen_bond': {
                'color': '#2E86AB', 'style': 'dashed', 'label': 'Hydrogen Bond',
                'network_color': '#3498db', 'linewidth': 2.0
            },
            'salt_bridge': {
                'color': '#2E8B57', 'style': 'solid', 'label': 'Salt Bridge',
                'network_color': '#27ae60', 'linewidth': 3.0
            },
            'pi_pi': {
                'color': '#8A2BE2', 'style': 'dotted', 'label': 'π-π Stacking',
                'network_color': '#9b59b6', 'linewidth': 2.5
            },
            'pi_cation': {
                'color': '#FF4500', 'style': 'dashdot', 'label': 'π-Cation',
                'network_color': '#e67e22', 'linewidth': 2.5
            },
            'hydrophobic': {
                'color': '#FFD700', 'style': 'solid', 'label': 'Hydrophobic',
                'network_color': '#f1c40f', 'linewidth': 1.5
            },
            'disulfide': {
                'color': '#FF6347', 'style': 'solid', 'label': 'Disulfide',
                'network_color': '#e74c3c', 'linewidth': 2.0
            }
        }

        # Chain color scheme
        self.chain_colors = {
            'A': '#FF6B6B',      # Red
            'B': '#4ECDC4',      # Teal
            'C': '#45B7D1',      # Blue
            'D': '#96CEB4',      # Green
            'E': '#FFEAA7',      # Yellow
            'F': '#DDA0DD',      # Plum
            'L': '#FFA07A',      # Light Salmon
            'MAIN': '#87CEEB'    # Sky Blue
        }

        # Schrodinger Amino Acid Color Scheme (based on 2D interaction diagram standards)
        # Enhanced with visible, contrasting colors for clear distinction
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

        print("🔬 Comprehensive PPI Analyzer Initialized")
        print(f"📁 CMS File: {self.cms_file}")
        if self.trajectory_file:
            print(f"📁 Trajectory: {self.trajectory_file}")
        print(f"🔗 Chain Groups: {self.chain_groups}")
        print(f"📊 Occupancy Threshold: {self.occupancy_threshold*100:.0f}%")

    def validate_files(self):
        """Validate input files exist."""
        print("🔍 Validating input files...")

        if not self.cms_file.exists():
            raise FileNotFoundError(f"CMS file not found: {self.cms_file}")

        if self.trajectory_file and not self.trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_file}")

        print("✅ Files validated successfully")

    def load_structure(self):
        """Load the CMS structure file with protein filtering."""
        print("📂 Loading CMS structure...")

        try:
            with StructureReader(str(self.cms_file)) as reader:
                self.structure = next(reader)

            print(f"✅ Loaded structure with {len(self.structure.atom)} atoms")
            print(f"✅ Structure has {len(self.structure.chain)} chains")

            # Extract chain information (filter out waters, solvents, metals, counter ions)
            for chain in self.structure.chain:
                chain_id = chain.name
                # Filter out non-protein residues
                protein_residues = []
                for res in chain.residue:
                    res_name = res.pdbres.strip()
                    if res_name and res_name not in ['HOH', 'WAT', 'TIP3', 'SOL', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN', 'NA+', 'CL-', 'K+', 'MG+2', 'CA+2']:
                        protein_residues.append(res)

                if protein_residues:  # Only include chains with protein residues
                    self.chain_info[chain_id] = {
                        'residues': protein_residues,
                        'count': len(protein_residues),
                        'atoms': sum(len(res.atom) for res in protein_residues)
                    }
                    print(f"   Chain {chain_id}: {len(protein_residues)} protein residues, {sum(len(res.atom) for res in protein_residues)} atoms")

        except Exception as e:
            print(f"❌ Error loading structure: {e}")
            raise

    def extract_residue_info(self):
        """Extract detailed residue information for all chains."""
        print("📊 Extracting residue information...")

        for chain in self.structure.chain:
            chain_id = chain.name
            for residue in chain.residue:
                if not residue.pdbres.strip():
                    continue

                # Skip non-protein residues
                if residue.pdbres.strip() in ['HOH', 'WAT', 'TIP3', 'SOL', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN', 'NA+', 'CL-', 'K+', 'MG+2', 'CA+2']:
                    continue

                # Get CA atom for residue center
                ca_atom = None
                for atom in residue.atom:
                    if atom.pdbname.strip() == 'CA':
                        ca_atom = atom
                        break

                if not ca_atom:
                    continue

                # Create residue identifier (handle empty chain names)
                chain_display = chain_id if chain_id else "MAIN"
                res_id = f"{chain_display}:{residue.pdbres.strip()}{residue.resnum}"

                self.residue_info[res_id] = {
                    'chain': chain_display,
                    'name': residue.pdbres.strip(),
                    'number': residue.resnum,
                    'xyz': ca_atom.xyz,
                    'atoms': [atom.pdbname.strip() for atom in residue.atom],
                    'residue': residue
                }

        print(f"✅ Extracted info for {len(self.residue_info)} residues")

        # Debug: Print chain information
        print("🔍 DEBUG: Chain information:")
        for chain in self.structure.chain:
            chain_name = chain.name if chain.name else "EMPTY"
            print(f"   Chain '{chain_name}': {len(chain.residue)} residues")
            if len(chain.residue) > 0:
                sample_res = list(chain.residue)[0]
                print(f"     Sample residue: {sample_res.pdbres.strip()}{sample_res.resnum}")

    def analyze_hydrogen_bonds(self, frame_structure=None):
        """Analyze hydrogen bonds between protein chains."""
        print("🔗 Analyzing hydrogen bonds...")

        structure = frame_structure or self.structure
        hbonds = hbond.get_hydrogen_bonds(structure)

        protein_interactions = []

        # Handle chain groups properly - analyze interactions between each pair of groups
        for i, group1 in enumerate(self.chain_groups):
            for group2 in self.chain_groups[i+1:]:
                for hb in hbonds:
                    try:
                        # Handle different hydrogen bond formats
                        if len(hb) >= 2:
                            donor_atom = hb[0]
                            acceptor_atom = hb[1]
                        else:
                            # Skip if not enough atoms in hydrogen bond
                            continue

                        donor_res = donor_atom.getResidue()
                        acceptor_res = acceptor_atom.getResidue()

                        # Skip water and non-protein residues
                        if (donor_res.pdbres.strip() in ['HOH', 'WAT', 'TIP3', 'SOL', 'SPC', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN'] or
                            acceptor_res.pdbres.strip() in ['HOH', 'WAT', 'TIP3', 'SOL', 'SPC', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN']):
                            continue

                        # Get chain names - handle empty chain names properly
                        donor_chain = donor_res.chain if isinstance(donor_res.chain, str) else donor_res.chain.name if donor_res.chain.name else ""
                        acceptor_chain = acceptor_res.chain if isinstance(acceptor_res.chain, str) else acceptor_res.chain.name if acceptor_res.chain.name else ""

                        # Check if donor is in group1 and acceptor is in group2, or vice versa
                        donor_in_group1 = donor_chain in group1
                        acceptor_in_group2 = acceptor_chain in group2
                        donor_in_group2 = donor_chain in group2
                        acceptor_in_group1 = acceptor_chain in group1

                        # Only include inter-chain protein interactions
                        if ((donor_in_group1 and acceptor_in_group2) or
                            (donor_in_group2 and acceptor_in_group1)):

                            # Calculate distance
                            distance = np.linalg.norm(np.array(donor_atom.xyz) - np.array(acceptor_atom.xyz))

                            # Create interaction record
                            donor_id = f"{donor_chain}:{donor_res.pdbres.strip()}{donor_res.resnum}"
                            acceptor_id = f"{acceptor_chain}:{acceptor_res.pdbres.strip()}{acceptor_res.resnum}"

                            interaction = {
                                'type': 'hydrogen_bond',
                                'donor': donor_id,
                                'acceptor': acceptor_id,
                                'distance': distance,
                                'group1': group1,
                                'group2': group2
                            }

                            protein_interactions.append(interaction)

                    except Exception as e:
                        # Skip problematic hydrogen bonds instead of printing warnings
                        continue

        print(f"✅ Found {len(protein_interactions)} protein-protein hydrogen bonds")
        return protein_interactions

    def analyze_salt_bridges(self, frame_structure=None):
        """Analyze salt bridges between protein chains."""
        print("⚡ Analyzing salt bridges...")

        structure = frame_structure or self.structure
        salt_bridges = []

        # Handle chain groups properly - analyze interactions between each pair of groups
        for i, group1 in enumerate(self.chain_groups):
            for group2 in self.chain_groups[i+1:]:
                try:
                    # Get charged residues from each group
                    charged1 = []
                    charged2 = []

                    for chain in structure.chain:
                        chain_name = chain.name if chain.name else ""

                        if chain_name in group1:
                            for residue in chain.residue:
                                # Skip water and non-protein residues
                                if residue.pdbres.strip() in ['HOH', 'WAT', 'TIP3', 'SOL', 'SPC', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN']:
                                    continue
                                if residue.pdbres.strip() in ['ARG', 'LYS', 'ASP', 'GLU']:
                                    charged1.append(residue)
                        elif chain_name in group2:
                            for residue in chain.residue:
                                # Skip water and non-protein residues
                                if residue.pdbres.strip() in ['HOH', 'WAT', 'TIP3', 'SOL', 'SPC', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN']:
                                    continue
                                if residue.pdbres.strip() in ['ARG', 'LYS', 'ASP', 'GLU']:
                                    charged2.append(residue)

                    # Check for salt bridges between groups
                    for res1 in charged1:
                        for res2 in charged2:
                            # Check if they form a salt bridge (opposite charges)
                            if ((res1.pdbres.strip() in ['ARG', 'LYS'] and res2.pdbres.strip() in ['ASP', 'GLU']) or
                                (res1.pdbres.strip() in ['ASP', 'GLU'] and res2.pdbres.strip() in ['ARG', 'LYS'])):

                                # Get charged atoms
                                atoms1 = []
                                atoms2 = []

                                for atom in res1.atom:
                                    if res1.pdbres.strip() in ['ARG', 'LYS']:
                                        if atom.pdbname.strip() in ['NH1', 'NH2', 'NZ']:
                                            atoms1.append(atom)
                                    else:  # ASP, GLU
                                        if atom.pdbname.strip() in ['OD1', 'OD2', 'OE1', 'OE2']:
                                            atoms1.append(atom)

                                for atom in res2.atom:
                                    if res2.pdbres.strip() in ['ARG', 'LYS']:
                                        if atom.pdbname.strip() in ['NH1', 'NH2', 'NZ']:
                                            atoms2.append(atom)
                                    else:  # ASP, GLU
                                        if atom.pdbname.strip() in ['OD1', 'OD2', 'OE1', 'OE2']:
                                            atoms2.append(atom)

                                # Check distances
                                for atom1 in atoms1:
                                    for atom2 in atoms2:
                                        # Check if they form a salt bridge
                                        distance = np.linalg.norm(np.array(atom1.xyz) - np.array(atom2.xyz))

                                        if distance <= 4.0:  # Salt bridge threshold
                                            # Handle empty chain names
                                            chain1 = res1.chain if isinstance(res1.chain, str) else res1.chain.name if res1.chain.name else "MAIN"
                                            chain2 = res2.chain if isinstance(res2.chain, str) else res2.chain.name if res2.chain.name else "MAIN"

                                            res1_id = f"{chain1}:{res1.pdbres.strip()}{res1.resnum}"
                                            res2_id = f"{chain2}:{res2.pdbres.strip()}{res2.resnum}"

                                            salt_bridges.append({
                                                'type': 'salt_bridge',
                                                'residue1': res1_id,
                                                'residue2': res2_id,
                                                'distance': distance,
                                                'group1': group1,
                                                'group2': group2
                                            })

                except Exception as e:
                    print(f"⚠️ Warning: Error analyzing salt bridges between groups {group1} and {group2}: {e}")

        print(f"✅ Found {len(salt_bridges)} protein-protein salt bridges")
        return salt_bridges

    def analyze_pi_interactions(self, frame_structure=None):
        """Analyze π-π and π-cation interactions between protein chains."""
        print("🔄 Analyzing π interactions...")

        structure = frame_structure or self.structure
        pi_interactions = []

        # Handle chain groups properly - analyze interactions between each pair of groups
        for i, group1 in enumerate(self.chain_groups):
            for group2 in self.chain_groups[i+1:]:
                try:
                    # Get aromatic and charged residues from each group
                    aromatic1 = []
                    aromatic2 = []
                    charged1 = []
                    charged2 = []

                    for chain in structure.chain:
                        chain_name = chain.name if chain.name else ""

                        if chain_name in group1:
                            for residue in chain.residue:
                                # Skip water and non-protein residues
                                if residue.pdbres.strip() in ['HOH', 'WAT', 'TIP3', 'SOL', 'SPC', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN']:
                                    continue
                                if residue.pdbres.strip() in ['PHE', 'TYR', 'TRP', 'HIS']:
                                    aromatic1.append(residue)
                                if residue.pdbres.strip() in ['ARG', 'LYS', 'ASP', 'GLU']:
                                    charged1.append(residue)
                        elif chain_name in group2:
                            for residue in chain.residue:
                                # Skip water and non-protein residues
                                if residue.pdbres.strip() in ['HOH', 'WAT', 'TIP3', 'SOL', 'SPC', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN']:
                                    continue
                                if residue.pdbres.strip() in ['PHE', 'TYR', 'TRP', 'HIS']:
                                    aromatic2.append(residue)
                                if residue.pdbres.strip() in ['ARG', 'LYS', 'ASP', 'GLU']:
                                    charged2.append(residue)

                    # Analyze π-π interactions
                    for res1 in aromatic1:
                        for res2 in aromatic2:
                            # Calculate centers
                            center1 = self._get_aromatic_center(res1)
                            center2 = self._get_aromatic_center(res2)

                            if center1 is not None and center2 is not None:
                                distance = np.linalg.norm(center1 - center2)

                                if distance <= 6.0:  # π-π interaction threshold
                                    # Handle empty chain names
                                    chain1 = res1.chain if isinstance(res1.chain, str) else res1.chain.name if res1.chain.name else "MAIN"
                                    chain2 = res2.chain if isinstance(res2.chain, str) else res2.chain.name if res2.chain.name else "MAIN"

                                    res1_id = f"{chain1}:{res1.pdbres.strip()}{res1.resnum}"
                                    res2_id = f"{chain2}:{res2.pdbres.strip()}{res2.resnum}"

                                    pi_interactions.append({
                                        'type': 'pi_pi',
                                        'residue1': res1_id,
                                        'residue2': res2_id,
                                        'distance': distance,
                                        'group1': group1,
                                        'group2': group2
                                    })

                    # Analyze π-cation interactions
                    for res1 in aromatic1:
                        for res2 in charged2:
                            # Calculate centers
                            center1 = self._get_aromatic_center(res1)
                            center2 = self._get_charged_center(res2)

                            if center1 is not None and center2 is not None:
                                distance = np.linalg.norm(center1 - center2)

                                if distance <= 6.0:  # π-cation interaction threshold
                                    # Handle empty chain names
                                    chain1 = res1.chain if isinstance(res1.chain, str) else res1.chain.name if res1.chain.name else "MAIN"
                                    chain2 = res2.chain if isinstance(res2.chain, str) else res2.chain.name if res2.chain.name else "MAIN"

                                    res1_id = f"{chain1}:{res1.pdbres.strip()}{res1.resnum}"
                                    res2_id = f"{chain2}:{res2.pdbres.strip()}{res2.resnum}"

                                    pi_interactions.append({
                                        'type': 'pi_cation',
                                        'residue1': res1_id,
                                        'residue2': res2_id,
                                        'distance': distance,
                                        'group1': group1,
                                        'group2': group2
                                    })

                    # Analyze π-cation interactions (reverse direction)
                    for res1 in charged1:
                        for res2 in aromatic2:
                            # Calculate centers
                            center1 = self._get_charged_center(res1)
                            center2 = self._get_aromatic_center(res2)

                            if center1 is not None and center2 is not None:
                                distance = np.linalg.norm(center1 - center2)

                                if distance <= 6.0:  # π-cation interaction threshold
                                    # Handle empty chain names
                                    chain1 = res1.chain if isinstance(res1.chain, str) else res1.chain.name if res1.chain.name else "MAIN"
                                    chain2 = res2.chain if isinstance(res2.chain, str) else res2.chain.name if res2.chain.name else "MAIN"

                                    res1_id = f"{chain1}:{res1.pdbres.strip()}{res1.resnum}"
                                    res2_id = f"{chain2}:{res2.pdbres.strip()}{res2.resnum}"

                                    pi_interactions.append({
                                        'type': 'pi_cation',
                                        'residue1': res1_id,
                                        'residue2': res2_id,
                                        'distance': distance,
                                        'group1': group1,
                                        'group2': group2
                                    })

                except Exception as e:
                    print(f"⚠️ Warning: Error analyzing π interactions between groups {group1} and {group2}: {e}")

        print(f"✅ Found {len(pi_interactions)} protein-protein π interactions")
        return pi_interactions

    def _get_aromatic_center(self, residue):
        """Calculate the center of an aromatic ring."""
        aromatic_atoms = []

        if residue.pdbres.strip() == 'PHE':
            # Phenylalanine: CG, CD1, CD2, CE1, CE2, CZ
            atom_names = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
        elif residue.pdbres.strip() == 'TYR':
            # Tyrosine: CG, CD1, CD2, CE1, CE2, CZ, OH
            atom_names = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
        elif residue.pdbres.strip() == 'TRP':
            # Tryptophan: CG, CD1, CD2, CE2, CE3, CZ2, CZ3, CH2
            atom_names = ['CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']
        elif residue.pdbres.strip() == 'HIS':
            # Histidine: CG, CD2, ND1, CE1, NE2
            atom_names = ['CG', 'CD2', 'ND1', 'CE1', 'NE2']
        else:
            return None

        for atom in residue.atom:
            if atom.pdbname.strip() in atom_names:
                aromatic_atoms.append(atom.xyz)

        if aromatic_atoms:
            return np.mean(aromatic_atoms, axis=0)
        return None

    def _get_charged_center(self, residue):
        """Calculate the center of charged groups."""
        charged_atoms = []

        if residue.pdbres.strip() in ['ARG', 'LYS']:
            # Basic residues: use NZ for LYS, NH1/NH2 for ARG
            if residue.pdbres.strip() == 'LYS':
                atom_names = ['NZ']
            else:  # ARG
                atom_names = ['NH1', 'NH2']
        elif residue.pdbres.strip() in ['ASP', 'GLU']:
            # Acidic residues: use OD1/OD2 for ASP, OE1/OE2 for GLU
            if residue.pdbres.strip() == 'ASP':
                atom_names = ['OD1', 'OD2']
            else:  # GLU
                atom_names = ['OE1', 'OE2']
        else:
            return None

        for atom in residue.atom:
            if atom.pdbname.strip() in atom_names:
                charged_atoms.append(atom.xyz)

        if charged_atoms:
            return np.mean(charged_atoms, axis=0)
        return None

    def analyze_frame(self, frame_structure):
        """Analyze all interaction types for a single frame."""
        print("🔍 Analyzing interactions...")

        all_interactions = []

        # Analyze different interaction types
        hbonds = self.analyze_hydrogen_bonds(frame_structure)
        salt_bridges = self.analyze_salt_bridges(frame_structure)
        pi_interactions = self.analyze_pi_interactions(frame_structure)

        all_interactions.extend(hbonds)
        all_interactions.extend(salt_bridges)
        all_interactions.extend(pi_interactions)

        print(f"✅ Found {len(all_interactions)} interactions:")
        print(f"   Hydrogen bonds: {len(hbonds)}")
        print(f"   Salt bridges: {len(salt_bridges)}")
        print(f"   π interactions: {len(pi_interactions)}")

        return all_interactions

    def analyze_trajectory_with_occupancy(self, max_frames=1000, max_workers=4):
        """Analyze multiple frames from trajectory to calculate occupancy percentages using multi-threading."""
        # Ensure structure is loaded
        if self.structure is None:
            self.load_structure()
        print(f"🎬 Analyzing trajectory for occupancy calculation (max {max_frames} frames, {max_workers} workers)...")

        if not self.trajectory_file or not self.trajectory_file.exists():
            print("⚠️ No trajectory file found, analyzing single frame only")
            return self.analyze_frame(self.structure)

        # Reset occupancy tracking
        self.interaction_occupancy = {}
        self.interaction_distance_history = {}
        self.total_frames_analyzed = 0

        try:
            # Use proper Desmond trajectory reader
            print(f"📂 Loading trajectory using Desmond reader: {self.trajectory_file}")

            # Read trajectory frames
            trajectory_frames = list(traj.read_traj(str(self.trajectory_file)))
            total_available_frames = len(trajectory_frames)
            print(f"📊 Found {total_available_frames} trajectory frames")

            if total_available_frames == 0:
                print("⚠️ No trajectory frames found, using single frame analysis")
                return self.analyze_frame(self.structure)

            # Limit to max_frames if specified
            frames_to_analyze = min(total_available_frames, max_frames)
            print(f"🎯 Will analyze {frames_to_analyze} frames using {max_workers} threads")

            # Sample frames evenly across the trajectory
            frame_indices = np.linspace(0, total_available_frames-1, frames_to_analyze, dtype=int)

            # Thread-safe counters
            frame_counter = 0
            frame_lock = threading.Lock()

            def analyze_single_frame(frame_idx):
                """Analyze a single frame - thread-safe function."""
                nonlocal frame_counter

                try:
                    # Get frame and apply coordinates to structure
                    frame = trajectory_frames[frame_idx]

                    # Create a copy of the structure and apply frame coordinates
                    frame_structure = self.structure.copy()
                    positions = frame.pos()

                    # Apply coordinates to the structure
                    for i, atom in enumerate(frame_structure.atom):
                        if i < len(positions):
                            atom.xyz = positions[i]

                    # Analyze current frame
                    frame_interactions = self.analyze_frame(frame_structure)

                    # Update progress counter
                    with frame_lock:
                        nonlocal frame_counter
                        frame_counter += 1
                        if frame_counter % 10 == 0:
                            print(f"   📊 Analyzed {frame_counter}/{frames_to_analyze} frames...")

                    return frame_interactions

                except Exception as frame_error:
                    print(f"   ⚠️ Error analyzing frame {frame_idx}: {frame_error}")
                    return []

            # Use ThreadPoolExecutor for parallel processing
            all_interactions = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all frame analysis tasks
                future_to_frame = {executor.submit(analyze_single_frame, frame_idx): frame_idx
                                 for frame_idx in frame_indices}

                # Collect results as they complete
                for future in as_completed(future_to_frame):
                    frame_idx = future_to_frame[future]
                    try:
                        frame_interactions = future.result()

                        # Track interactions for occupancy calculation (thread-safe)
                        with frame_lock:
                            self.total_frames_analyzed += 1
                            for interaction in frame_interactions:
                                self._update_occupancy(interaction)
                                self._update_distance_history(interaction)
                            all_interactions.extend(frame_interactions)

                    except Exception as e:
                        print(f"   ⚠️ Error processing frame {frame_idx}: {e}")

            print(f"✅ Successfully analyzed {self.total_frames_analyzed} trajectory frames using multi-threading")

            # Filter by occupancy threshold
            filtered_interactions = self._filter_by_occupancy(all_interactions, self.occupancy_threshold)

            return filtered_interactions

        except Exception as e:
            print(f"❌ Error during trajectory analysis: {e}")
            print("⚠️ Falling back to single frame analysis")
            return self.analyze_frame(self.structure)

    def _get_interaction_key(self, interaction):
        """Generate a unique key for an interaction."""
        res1 = interaction.get('donor', interaction.get('residue1', ''))
        res2 = interaction.get('acceptor', interaction.get('residue2', ''))
        itype = interaction['type']
        return f"{res1}_{res2}_{itype}"

    def _update_occupancy(self, interaction):
        """Update occupancy tracking for an interaction."""
        key = self._get_interaction_key(interaction)
        if key not in self.interaction_occupancy:
            self.interaction_occupancy[key] = {
                'count': 0,
                'example': interaction
            }
        self.interaction_occupancy[key]['count'] += 1

    def _update_distance_history(self, interaction):
        """Update distance history for an interaction."""
        key = self._get_interaction_key(interaction)
        if key not in self.interaction_distance_history:
            self.interaction_distance_history[key] = []
        self.interaction_distance_history[key].append(interaction.get('distance', 0))

    def _filter_by_occupancy(self, interactions, occupancy_threshold):
        """Filter interactions by occupancy threshold."""
        filtered_interactions = []

        for interaction in interactions:
            occupancy = interaction.get('occupancy', 1.0)
            if occupancy >= occupancy_threshold:
                filtered_interactions.append(interaction)

        return filtered_interactions

    def create_network_graph(self, interactions):
        """Create a network graph from interactions."""
        G = nx.Graph()

        # Add nodes
        for interaction in interactions:
            res1 = interaction.get('donor', interaction.get('residue1', ''))
            res2 = interaction.get('acceptor', interaction.get('residue2', ''))

            if res1 not in G:
                G.add_node(res1,
                          chain=self.residue_info.get(res1, {}).get('chain', 'UNK'),
                          name=self.residue_info.get(res1, {}).get('name', 'UNK'),
                          number=self.residue_info.get(res1, {}).get('number', 0))

            if res2 not in G:
                G.add_node(res2,
                          chain=self.residue_info.get(res2, {}).get('chain', 'UNK'),
                          name=self.residue_info.get(res2, {}).get('name', 'UNK'),
                          number=self.residue_info.get(res2, {}).get('number', 0))

            # Add edge
            G.add_edge(res1, res2,
                      type=interaction['type'],
                      distance=interaction.get('distance', 0),
                      occupancy=interaction.get('occupancy', 1.0))

        return G

    def create_network_diagram(self, G, interactions, output_file='ppi_network.png'):
        """Create a journal-quality network diagram with professional layout."""
        if len(G.nodes()) == 0:
            print(f"⚠️ No nodes in the network graph. Skipping network diagram plot: {output_file}")
            return
        print(f"🎨 Creating journal-quality network diagram: {output_file}")

        # Set up the plot with journal-quality dimensions (A4 ratio)
        fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(24, 12),
                                                gridspec_kw={'width_ratios': [4, 1]})

        # Professional color scheme for background
        fig.patch.set_facecolor('#FAFAFA')
        ax_main.set_facecolor('#FFFFFF')
        ax_legend.set_facecolor('#FAFAFA')

        # Protein-protein interaction specific layout
        # Separate nodes by chain and position them to show interaction interface
        pos = {}

        # Get unique chains in the network
        chains_in_network = set()
        for node in G.nodes():
            chain = G.nodes[node]['chain']
            chains_in_network.add(chain)

        chains_list = list(chains_in_network)

        if len(chains_list) == 2:
            # Two-chain interaction: Create proper outlines for each chain
            chain1, chain2 = chains_list[0], chains_list[1]

            # Separate nodes by chain
            chain1_nodes = [node for node in G.nodes() if G.nodes[node]['chain'] == chain1]
            chain2_nodes = [node for node in G.nodes() if G.nodes[node]['chain'] == chain2]

            # Sort nodes by residue number for proper sequential arrangement
            chain1_nodes.sort(key=lambda x: G.nodes[x]['number'])
            chain2_nodes.sort(key=lambda x: G.nodes[x]['number'])

            # Create protein chain outlines - chain1 on left, chain2 on right
            # Chain1: Arrange in a protein-like shape on the left
            chain1_center = np.array([-0.6, 0])  # Moved closer to center
            chain1_radius = 0.3  # Smaller radius to stay within bounds
            for i, node in enumerate(chain1_nodes):
                # Arrange in a protein-like curve (helix-like arrangement)
                angle = (i / len(chain1_nodes)) * 2 * np.pi
                # Add some variation to make it look more like a protein structure
                radius_var = chain1_radius + 0.05 * np.sin(3 * angle)  # Reduced variation
                pos[node] = chain1_center + np.array([radius_var * np.cos(angle), radius_var * np.sin(angle)])

            # Chain2: Arrange in a protein-like shape on the right
            chain2_center = np.array([0.6, 0])  # Moved closer to center
            chain2_radius = 0.3  # Smaller radius to stay within bounds
            for i, node in enumerate(chain2_nodes):
                # Arrange in a protein-like curve (helix-like arrangement)
                angle = (i / len(chain2_nodes)) * 2 * np.pi
                # Add some variation to make it look more like a protein structure
                radius_var = chain2_radius + 0.05 * np.sin(3 * angle)  # Reduced variation
                pos[node] = chain2_center + np.array([radius_var * np.cos(angle), radius_var * np.sin(angle)])

        elif len(chains_list) == 1:
            # Single chain: Use circular layout
            nodes = list(G.nodes())
            for i, node in enumerate(nodes):
                angle = (i / len(nodes)) * 2 * np.pi
                radius = 0.5  # Smaller radius to stay within bounds
                pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        else:
            # Multiple chains: Use bipartite-style layout
            for i, chain in enumerate(chains_list):
                chain_nodes = [node for node in G.nodes() if G.nodes[node]['chain'] == chain]
                y_pos = 0.8 - (i * 1.6 / (len(chains_list) - 1)) if len(chains_list) > 1 else 0

                for j, node in enumerate(chain_nodes):
                    x_pos = (j / len(chain_nodes)) * 1.2 - 0.6  # Reduced spread
                    pos[node] = np.array([x_pos, y_pos])

        # Draw edges by interaction type with enhanced PPI styling
        for interaction_type, props in self.interaction_types.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == interaction_type]

            if edges:
                # Thinner edge styling for cleaner PPI visualization
                edge_width = props['linewidth'] * 0.8  # Make edges thinner
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edges,
                    edge_color=props['network_color'],
                    style=props['style'],
                    width=edge_width,
                    alpha=1.0,  # Full opacity for clear visibility
                    ax=ax_main
                )

                # Add subtle black outline to edges for better visibility
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edges,
                    edge_color='black',
                    style=props['style'],
                    width=edge_width + 0.5,  # Slightly thicker black outline
                    alpha=0.2,  # Very subtle outline
                    ax=ax_main
                )

        # Draw nodes with clean, professional appearance
        node_colors = []
        node_sizes = []
        node_labels = {}
        node_edge_colors = []
        node_alpha = []

        # Protein-protein interaction specific sizing - much bigger circles
        # Base size is significantly larger for PPI visualization
        base_size = min(3500, max(2000, 5000 // len(G.nodes())))

        for node in G.nodes():
            chain = G.nodes[node]['chain']
            res_name = G.nodes[node]['name']
            res_num = G.nodes[node]['number']

            # Color by amino acid type (Schrodinger standard)
            aa_color = self.amino_acid_colors.get(res_name, self.amino_acid_colors['UNK'])
            node_colors.append(aa_color)

            # Enhanced sizing for PPI visualization - much bigger circles
            if res_name in ['ARG', 'LYS', 'ASP', 'GLU']:  # Charged residues (most important for PPI)
                node_sizes.append(int(base_size * 1.5))
                node_alpha.append(0.95)
            elif res_name in ['PHE', 'TYR', 'TRP', 'HIS']:  # Aromatic residues (important for PPI)
                node_sizes.append(int(base_size * 1.4))
                node_alpha.append(0.9)
            elif res_name in ['SER', 'THR', 'ASN', 'GLN']:  # Polar residues
                node_sizes.append(int(base_size * 1.2))
                node_alpha.append(0.85)
            else:  # Hydrophobic residues
                node_sizes.append(base_size)
                node_alpha.append(0.8)

            # Edge color based on chain with professional styling
            node_edge_colors.append(self.chain_colors.get(chain, '#CCCCCC'))

            # Enhanced labeling with chain name always included
            if len(G.nodes()) <= 25:
                node_labels[node] = f"{chain}:{res_name}\n{res_num}"
            else:
                node_labels[node] = f"{chain}:{res_name}{res_num}"

        # Draw nodes with enhanced PPI styling
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=1.5,
            ax=ax_main
        )

        # Add dim light color imaginary island outline around both chains
        if len(chains_list) == 2:
            # Create smooth island outline for chain 1 (left side)
            chain1_center = np.array([-0.6, 0])
            chain1_radius = 0.45  # Bigger to cover all amino acids
            island1_theta = np.linspace(0, 2*np.pi, 100)  # More points for smoothness
            # Add gentle smooth irregularity for chain 1
            irregular_radius1 = chain1_radius + 0.03 * np.sin(2 * island1_theta) + 0.02 * np.cos(4 * island1_theta)
            island1_x = chain1_center[0] + irregular_radius1 * np.cos(island1_theta)
            island1_y = chain1_center[1] + irregular_radius1 * np.sin(island1_theta)
            ax_main.plot(island1_x, island1_y, color='steelblue', alpha=0.6, linewidth=2.5, linestyle='-')

            # Create smooth island outline for chain 2 (right side)
            chain2_center = np.array([0.6, 0])
            chain2_radius = 0.45  # Bigger to cover all amino acids
            island2_theta = np.linspace(0, 2*np.pi, 100)  # More points for smoothness
            # Add gentle smooth irregularity for chain 2
            irregular_radius2 = chain2_radius + 0.025 * np.sin(3 * island2_theta) + 0.015 * np.cos(5 * island2_theta)
            island2_x = chain2_center[0] + irregular_radius2 * np.cos(island2_theta)
            island2_y = chain2_center[1] + irregular_radius2 * np.sin(island2_theta)
            ax_main.plot(island2_x, island2_y, color='darkred', alpha=0.6, linewidth=2.5, linestyle='-')
        elif len(chains_list) == 1:
            # Single chain smooth island outline
            center = np.array([0, 0])
            radius = 0.65  # Bigger to cover all amino acids
            island_theta = np.linspace(0, 2*np.pi, 100)  # More points for smoothness
            # Add gentle smooth irregularity
            irregular_radius = radius + 0.04 * np.sin(2.5 * island_theta) + 0.025 * np.cos(6 * island_theta)
            island_x = center[0] + irregular_radius * np.cos(island_theta)
            island_y = center[1] + irregular_radius * np.sin(island_theta)
            ax_main.plot(island_x, island_y, color='darkgreen', alpha=0.6, linewidth=2.5, linestyle='-')

        # Draw labels with stroke/outline effect for visibility on any background
        font_size = min(14, max(8, 24 // len(G.nodes())))  # Slightly larger fonts

        # Create clean labels without background
        enhanced_labels = {}
        for node, label in node_labels.items():
            if '\n' in label:  # Multi-line label
                chain_part, res_part = label.split('\n')
                enhanced_labels[node] = f"{chain_part}\n{res_part}"
            else:  # Single line label
                enhanced_labels[node] = label

        # Draw text stroke/outline effect using multiple thin white rings
        # Create multiple offset positions for stroke effect
        stroke_offsets = [
            (0.003, 0), (0, 0.003), (-0.003, 0), (0, -0.003),  # Cardinal directions
            (0.002, 0.002), (0.002, -0.002), (-0.002, 0.002), (-0.002, -0.002)  # Diagonals
        ]

        # Draw white stroke rings
        for dx, dy in stroke_offsets:
            stroke_pos = {node: (pos[node][0] + dx, pos[node][1] + dy) for node in pos}
            nx.draw_networkx_labels(
                G, stroke_pos,
                labels=enhanced_labels,
                font_size=font_size,
                font_weight='bold',
                font_color='white',
                ax=ax_main
            )

        # Draw the main text on top
        nx.draw_networkx_labels(
            G, pos,
            labels=enhanced_labels,
            font_size=font_size,
            font_weight='bold',
            font_color='#2C3E50',
            ax=ax_main
        )

        # Edge labels with distance (only for small networks to avoid clutter)
        if len(G.nodes()) <= 30:
            edge_labels = {}
            for interaction in interactions:
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))
                distance = interaction.get('distance', 0)
                occupancy = interaction.get('occupancy', 1.0)

                if occupancy < 1.0:
                    edge_labels[(res1, res2)] = f"{distance:.1f}Å\n{occupancy:.0%}"
                else:
                    edge_labels[(res1, res2)] = f"{distance:.1f}Å"

            # Calculate adaptive font size for edge labels based on node sizes
            avg_node_size = np.mean(node_sizes) if node_sizes else 2000
            edge_font_size = min(16, max(10, int(avg_node_size / 200)))  # Scale with node size

            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_color='black',
                font_size=edge_font_size,
                font_weight='bold',
                ax=ax_main
            )

        # Enhanced Legend with Amino Acid Color Scheme
        legend_elements = []

        # Interaction types legend
        for interaction_type, props in self.interaction_types.items():
            if interaction_type in [d['type'] for d in interactions]:
                legend_elements.append(
                    mpatches.Patch(color=props['network_color'], label=props['label'])
                )

        # Add amino acid color legend
        aa_categories = {
            'Charged (Basic)': ['LYS', 'ARG', 'HIS'],
            'Charged (Acidic)': ['ASP', 'GLU'],
            'Polar': ['SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN'],
            'Hydrophobic': ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']
        }

        for category, residues in aa_categories.items():
            if any(G.nodes[node]['name'] in residues for node in G.nodes()):
                sample_res = residues[0]
                color = self.amino_acid_colors.get(sample_res, '#CCCCCC')
                legend_elements.append(
                    mpatches.Patch(color=color, label=f'Amino Acid: {category}')
                )

        # Add chain legend
        for chain, color in self.chain_colors.items():
            if any(G.nodes[node]['chain'] == chain for node in G.nodes()):
                legend_elements.append(
                    mpatches.Patch(color=color, label=f'Chain {chain} (Edge)')
                )

        # Professional legend layout in separate subplot
        ax_legend.legend(handles=legend_elements, loc='center', fontsize=9,
                        frameon=True, fancybox=True, shadow=True)
        ax_legend.axis('off')

        # Enhanced Title with professional styling
        chain_groups_str = " vs ".join([f"Chain{'s' if len(g) > 1 else ''} {','.join(g)}" for g in self.chain_groups])
        ax_main.set_title(
            f"Protein-Protein Interaction Network\n"
            f"{chain_groups_str} | Schrodinger Amino Acid Coloring\n"
            f"({len(interactions)} interactions)",
            fontsize=16, fontweight='bold', pad=20, color='#2C3E50'
        )

        # Professional axis styling with PPI-specific enhancements
        ax_main.axis('off')
        ax_main.set_xlim([-1.2, 1.2])
        ax_main.set_ylim([-1.2, 1.2])

        # Add subtle grid for better spatial perception
        ax_main.grid(True, alpha=0.05, linestyle='-', linewidth=0.3)

        plt.tight_layout(pad=2.0)

        print(f"   Saving diagram to {output_file}...")
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"✅ Network diagram saved as {output_file}")
        plt.close()

    def print_summary(self, interactions):
        """Print comprehensive analysis summary."""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE PPI ANALYSIS SUMMARY")
        print("="*60)

        print(f"📁 Input Files:")
        print(f"   CMS: {self.cms_file}")
        if self.trajectory_file:
            print(f"   Trajectory: {self.trajectory_file}")
        print(f"   Chain Groups: {self.chain_groups}")

        print(f"\n📈 Structure Statistics:")
        print(f"   Total chains: {len(self.chain_info)}")
        for chain, info in self.chain_info.items():
            print(f"   Chain {chain}: {info['count']} residues, {info['atoms']} atoms")
        print(f"   Total interactions: {len(interactions)}")

        # Interaction type breakdown
        interaction_types = Counter([d['type'] for d in interactions])
        print(f"\n🔗 Interaction Types:")
        for itype, count in interaction_types.most_common():
            label = self.interaction_types[itype]['label']
            print(f"   {label}: {count} interactions")

        if interactions:
            print(f"\n🏆 Top 10 Strongest Interactions:")
            for i, interaction in enumerate(interactions[:10]):
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))
                itype_label = self.interaction_types[interaction['type']]['label']
                distance = interaction.get('distance', 0)
                occupancy = interaction.get('occupancy', 1.0)

                if occupancy < 1.0:
                    print(f"   {i+1:2d}. {res1} ↔ {res2} "
                          f"({itype_label}: {distance:.2f}Å, {occupancy:.1%} occupancy)")
                else:
                    print(f"   {i+1:2d}. {res1} ↔ {res2} "
                          f"({itype_label}: {distance:.2f}Å)")

        print("="*60)

    def print_occupancy_summary(self, interactions):
        """Print occupancy analysis summary."""
        if not hasattr(self, 'total_frames_analyzed') or self.total_frames_analyzed <= 1:
            return

        print("\n" + "="*60)
        print("📊 OCCUPANCY ANALYSIS SUMMARY")
        print("="*60)

        print(f"🎬 Trajectory Analysis:")
        print(f"   Total frames analyzed: {self.total_frames_analyzed}")
        print(f"   Total unique interactions found: {len(self.interaction_occupancy)}")
        print(f"   Interactions with occupancy data: {len(interactions)}")

        if interactions:
            # Occupancy statistics
            occupancies = [interaction.get('occupancy', 0) for interaction in interactions]
            print(f"\n📈 Occupancy Statistics:")
            print(f"   Mean occupancy: {np.mean(occupancies):.2%}")
            print(f"   Median occupancy: {np.median(occupancies):.2%}")
            print(f"   Min occupancy: {min(occupancies):.2%}")
            print(f"   Max occupancy: {max(occupancies):.2%}")

            # Distance statistics
            distances = [interaction.get('distance', 0) for interaction in interactions]
            print(f"\n📏 Distance Statistics:")
            print(f"   Mean distance: {np.mean(distances):.2f} Å")
            print(f"   Median distance: {np.median(distances):.2f} Å")
            print(f"   Min distance: {min(distances):.2f} Å")
            print(f"   Max distance: {max(distances):.2f} Å")

            print(f"\n🏆 Top 10 Most Persistent Interactions:")
            sorted_interactions = sorted(interactions, key=lambda x: x.get('occupancy', 0), reverse=True)
            for i, interaction in enumerate(sorted_interactions[:10]):
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))
                itype_label = self.interaction_types[interaction['type']]['label']
                occupancy = interaction.get('occupancy', 0)
                distance = interaction.get('distance', 0)
                print(f"   {i+1:2d}. {res1} ↔ {res2} "
                      f"({itype_label}: {distance:.2f}Å, {occupancy:.1%} occupancy)")

        print("="*60)

    def save_results(self, interactions, output_prefix='ppi_analysis'):
        """Save analysis results to files."""
        print(f"💾 Saving results with prefix: {output_prefix}")

        # Save interactions as JSON
        json_file = f"{output_prefix}_interactions.json"
        with open(json_file, 'w') as f:
            json.dump(interactions, f, indent=2, default=str)
        print(f"✅ Interactions saved to {json_file}")

        # Save summary statistics
        summary_file = f"{output_prefix}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PROTEIN-PROTEIN INTERACTION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Input Files:\n")
            f.write(f"  CMS: {self.cms_file}\n")
            if self.trajectory_file:
                f.write(f"  Trajectory: {self.trajectory_file}\n")
            f.write(f"  Chain Groups: {self.chain_groups}\n\n")

            f.write(f"Structure Statistics:\n")
            f.write(f"  Total chains: {len(self.chain_info)}\n")
            for chain, info in self.chain_info.items():
                f.write(f"  Chain {chain}: {info['count']} residues, {info['atoms']} atoms\n")
            f.write(f"  Total interactions: {len(interactions)}\n\n")

            # Interaction type breakdown
            interaction_types = Counter([d['type'] for d in interactions])
            f.write(f"Interaction Types:\n")
            for itype, count in interaction_types.most_common():
                label = self.interaction_types[itype]['label']
                f.write(f"  {label}: {count} interactions\n")

            if hasattr(self, 'total_frames_analyzed') and self.total_frames_analyzed > 1:
                f.write(f"\nTrajectory Analysis:\n")
                f.write(f"  Total frames analyzed: {self.total_frames_analyzed}\n")
                f.write(f"  Occupancy threshold: {self.occupancy_threshold*100:.0f}%\n")

        print(f"✅ Summary saved to {summary_file}")

    def analyze_temporal_interactions(self, max_frames=1000, max_workers=4):
        """Analyze temporal dynamics of interactions during simulation."""
        print("⏰ Analyzing temporal interaction dynamics...")

        if not self.trajectory_file:
            print("❌ No trajectory file available for temporal analysis")
            return

        # Initialize temporal tracking
        self.temporal_interactions = {}
        self.interaction_timeline = {}
        self.chain_interaction_stats = {}
        self.frame_interactions = []

        # Get chain pairs for analysis
        chain_pairs = []
        for i, group1 in enumerate(self.chain_groups):
            for group2 in self.chain_groups[i+1:]:
                chain_pairs.append((group1, group2))

        print(f"🔗 Analyzing temporal dynamics for {len(chain_pairs)} chain pairs")

        # Load trajectory
        try:
            msys_model, cms_model = topo.read_cms(str(self.cms_file))
            tr = traj.read_traj(str(self.trajectory_file))

            total_frames = min(len(tr), max_frames)
            print(f"📊 Analyzing {total_frames} frames for temporal dynamics")

            # Analyze each frame
            for frame_idx in range(total_frames):
                frame = tr[frame_idx]

                # Get structure from frame - Desmond frames have different methods
                try:
                    # Try different methods to get structure from frame
                    if hasattr(frame, 'getStructure'):
                        frame_structure = frame.getStructure()
                    elif hasattr(frame, 'structure'):
                        frame_structure = frame.structure
                    elif hasattr(frame, 'get_structure'):
                        frame_structure = frame.get_structure()
                    else:
                        # For Desmond, we need to create a proper structure from the frame
                        # Use the original structure as template and update coordinates
                        frame_structure = self.structure.copy()

                        # Update atom coordinates from frame
                        if hasattr(frame, 'pos') and hasattr(frame, 'getPositions'):
                            try:
                                positions = frame.getPositions()
                                for i, atom in enumerate(frame_structure.atom):
                                    if i < len(positions):
                                        atom.xyz = positions[i]
                            except:
                                print(f"⚠️ Warning: Could not update coordinates for frame {frame_idx}")
                        elif hasattr(frame, 'getPositions'):
                            try:
                                positions = frame.getPositions()
                                for i, atom in enumerate(frame_structure.atom):
                                    if i < len(positions):
                                        atom.xyz = positions[i]
                            except:
                                print(f"⚠️ Warning: Could not update coordinates for frame {frame_idx}")
                        else:
                            print(f"⚠️ Warning: Could not extract coordinates from frame {frame_idx}, using original structure")

                except Exception as e:
                    print(f"⚠️ Warning: Could not get structure from frame {frame_idx}: {e}")
                    continue

                # Analyze interactions in this frame
                frame_interactions = self.analyze_frame(frame_structure)

                # Store frame interactions
                self.frame_interactions.append({
                    'frame': frame_idx,
                    'time': frame_idx * frame.getTimeStep() if hasattr(frame, 'getTimeStep') else frame_idx,
                    'interactions': frame_interactions
                })

                # Track temporal dynamics
                self._update_temporal_tracking(frame_idx, frame_interactions, chain_pairs)

                if frame_idx % 100 == 0:
                    print(f"   Processed frame {frame_idx}/{total_frames}")

            # Calculate temporal statistics
            self._calculate_temporal_statistics(chain_pairs)

            print(f"✅ Temporal analysis completed for {total_frames} frames")

        except Exception as e:
            print(f"❌ Error in temporal analysis: {e}")
            raise

    def _update_temporal_tracking(self, frame_idx, frame_interactions, chain_pairs):
        """Update temporal tracking data structures."""
        frame_time = frame_idx  # Can be converted to actual time if needed

        # Track interactions by chain pair
        for group1, group2 in chain_pairs:
            pair_key = f"{','.join(group1)}_vs_{','.join(group2)}"

            if pair_key not in self.temporal_interactions:
                self.temporal_interactions[pair_key] = {
                    'frames': [],
                    'interaction_types': defaultdict(list),
                    'residue_pairs': defaultdict(list)
                }

            # Filter interactions for this chain pair
            pair_interactions = []
            for interaction in frame_interactions:
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))

                # Check if interaction is between the chain pair
                chain1_in_group1 = any(chain in group1 for chain in [res1.split(':')[0] if ':' in res1 else ''])
                chain2_in_group2 = any(chain in group2 for chain in [res2.split(':')[0] if ':' in res2 else ''])
                chain1_in_group2 = any(chain in group2 for chain in [res1.split(':')[0] if ':' in res1 else ''])
                chain2_in_group1 = any(chain in group1 for chain in [res2.split(':')[0] if ':' in res2 else ''])

                if ((chain1_in_group1 and chain2_in_group2) or
                    (chain1_in_group2 and chain2_in_group1)):
                    pair_interactions.append(interaction)

            # Update temporal tracking
            self.temporal_interactions[pair_key]['frames'].append(frame_time)

            # Track by interaction type
            for interaction in pair_interactions:
                interaction_type = interaction['type']
                self.temporal_interactions[pair_key]['interaction_types'][interaction_type].append(frame_time)

                # Track specific residue pairs
                res_pair = f"{interaction.get('donor', interaction.get('residue1', ''))}-{interaction.get('acceptor', interaction.get('residue2', ''))}"
                self.temporal_interactions[pair_key]['residue_pairs'][res_pair].append(frame_time)

    def _calculate_temporal_statistics(self, chain_pairs):
        """Calculate temporal statistics for each chain pair."""
        print("📊 Calculating temporal statistics...")

        for group1, group2 in chain_pairs:
            pair_key = f"{','.join(group1)}_vs_{','.join(group2)}"

            if pair_key not in self.temporal_interactions:
                continue

            temporal_data = self.temporal_interactions[pair_key]
            total_frames = len(temporal_data['frames'])

            # Calculate statistics
            stats = {
                'total_frames': total_frames,
                'interaction_type_stats': {},
                'residue_pair_stats': {},
                'persistent_interactions': [],
                'transient_interactions': [],
                'interaction_events': []
            }

            # Analyze interaction types
            for interaction_type, frames in temporal_data['interaction_types'].items():
                presence_frames = len(frames)
                occupancy = presence_frames / total_frames if total_frames > 0 else 0

                stats['interaction_type_stats'][interaction_type] = {
                    'total_frames': presence_frames,
                    'occupancy': occupancy,
                    'first_frame': min(frames) if frames else None,
                    'last_frame': max(frames) if frames else None,
                    'duration': max(frames) - min(frames) + 1 if frames else 0
                }

            # Analyze residue pairs
            for res_pair, frames in temporal_data['residue_pairs'].items():
                presence_frames = len(frames)
                occupancy = presence_frames / total_frames if total_frames > 0 else 0

                stats['residue_pair_stats'][res_pair] = {
                    'total_frames': presence_frames,
                    'occupancy': occupancy,
                    'first_frame': min(frames) if frames else None,
                    'last_frame': max(frames) if frames else None,
                    'duration': max(frames) - min(frames) + 1 if frames else 0
                }

                # Classify as persistent or transient
                if occupancy >= 0.5:  # Present in >50% of frames
                    stats['persistent_interactions'].append(res_pair)
                else:
                    stats['transient_interactions'].append(res_pair)

            self.chain_interaction_stats[pair_key] = stats

    def print_temporal_analysis(self):
        """Print comprehensive temporal analysis results."""
        if not self.chain_interaction_stats:
            print("❌ No temporal analysis data available")
            return

        print("\n" + "="*80)
        print("⏰ TEMPORAL INTERACTION DYNAMICS ANALYSIS")
        print("="*80)

        for pair_key, stats in self.chain_interaction_stats.items():
            print(f"\n🔗 Chain Pair: {pair_key}")
            print(f"📊 Total Frames Analyzed: {stats['total_frames']}")

            # Interaction type statistics
            print(f"\n📈 Interaction Type Statistics:")
            for interaction_type, type_stats in stats['interaction_type_stats'].items():
                label = self.interaction_types[interaction_type]['label']
                occupancy_pct = type_stats['occupancy'] * 100
                duration = type_stats['duration']
                print(f"   {label}: {type_stats['total_frames']} frames ({occupancy_pct:.1f}% occupancy)")
                print(f"     Duration: {duration} frames (frames {type_stats['first_frame']}-{type_stats['last_frame']})")

            # Persistent vs transient interactions
            print(f"\n🔒 Persistent Interactions (≥50% occupancy): {len(stats['persistent_interactions'])}")
            for res_pair in stats['persistent_interactions'][:10]:  # Show first 10
                pair_stats = stats['residue_pair_stats'][res_pair]
                occupancy_pct = pair_stats['occupancy'] * 100
                print(f"   {res_pair}: {occupancy_pct:.1f}% occupancy")

            print(f"\n⚡ Transient Interactions (<50% occupancy): {len(stats['transient_interactions'])}")
            for res_pair in stats['transient_interactions'][:10]:  # Show first 10
                pair_stats = stats['residue_pair_stats'][res_pair]
                occupancy_pct = pair_stats['occupancy'] * 100
                print(f"   {res_pair}: {occupancy_pct:.1f}% occupancy")

            print("-" * 60)

    def save_temporal_analysis(self, output_prefix='temporal_analysis'):
        """Save temporal analysis results to files."""
        print(f"🔍 DEBUG: save_temporal_analysis called with output_prefix: {output_prefix}")

        if not self.chain_interaction_stats:
            print("❌ No temporal analysis data to save")
            print(f"🔍 DEBUG: chain_interaction_stats is empty: {self.chain_interaction_stats}")
            return

        print(f"🔍 DEBUG: Found {len(self.chain_interaction_stats)} chain pairs in stats")
        print(f"🔍 DEBUG: Chain pairs: {list(self.chain_interaction_stats.keys())}")

        print(f"💾 Saving temporal analysis results...")

        # Save detailed temporal statistics
        temporal_file = f"{output_prefix}_statistics.json"
        with open(temporal_file, 'w') as f:
            json.dump(self.chain_interaction_stats, f, indent=2, default=str)
        print(f"✅ Temporal statistics saved to {temporal_file}")

        # Save frame-by-frame interactions
        frame_file = f"{output_prefix}_frame_data.json"
        with open(frame_file, 'w') as f:
            json.dump(self.frame_interactions, f, indent=2, default=str)
        print(f"✅ Frame-by-frame data saved to {frame_file}")

        # Save temporal interaction data
        temporal_data_file = f"{output_prefix}_temporal_data.json"
        with open(temporal_data_file, 'w') as f:
            json.dump(self.temporal_interactions, f, indent=2, default=str)
        print(f"✅ Temporal interaction data saved to {temporal_data_file}")

        # Create summary CSV
        summary_file = f"{output_prefix}_summary.csv"
        with open(summary_file, 'w') as f:
            f.write("Chain_Pair,Interaction_Type,Total_Frames,Occupancy_Percent,Duration,First_Frame,Last_Frame\n")

            for pair_key, stats in self.chain_interaction_stats.items():
                for interaction_type, type_stats in stats['interaction_type_stats'].items():
                    occupancy_pct = type_stats['occupancy'] * 100
                    f.write(f"{pair_key},{interaction_type},{type_stats['total_frames']},"
                           f"{occupancy_pct:.2f},{type_stats['duration']},"
                           f"{type_stats['first_frame']},{type_stats['last_frame']}\n")

        print(f"✅ Summary CSV saved to {summary_file}")

        # Create temporal analysis plots
        print(f"🔍 DEBUG: About to call create_temporal_plots with output_prefix: {output_prefix}")
        self.create_temporal_plots(output_prefix)
        print(f"🔍 DEBUG: Finished create_temporal_plots call")

    def create_temporal_plots(self, output_prefix='temporal_analysis'):
        """Create comprehensive temporal analysis plots."""
        print(f"🔍 DEBUG: create_temporal_plots called with output_prefix: {output_prefix}")

        if not self.chain_interaction_stats:
            print("❌ No temporal analysis data available for plotting")
            print(f"🔍 DEBUG: chain_interaction_stats is empty: {self.chain_interaction_stats}")
            return

        print(f"🔍 DEBUG: Found {len(self.chain_interaction_stats)} chain pairs for plotting")
        print(f"🔍 DEBUG: Chain pairs: {list(self.chain_interaction_stats.keys())}")

        print("📊 Creating temporal analysis plots...")

        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Interaction Type Timeline Plot
        ax1 = plt.subplot(3, 2, 1)
        self._plot_interaction_timeline(ax1)

        # 2. Occupancy Heatmap
        ax2 = plt.subplot(3, 2, 2)
        self._plot_occupancy_heatmap(ax2)

        # 3. Interaction Duration Distribution
        ax3 = plt.subplot(3, 2, 3)
        self._plot_duration_distribution(ax3)

        # 4. Persistent vs Transient Interactions
        ax4 = plt.subplot(3, 2, 4)
        self._plot_persistent_vs_transient(ax4)

        # 5. Frame-by-Frame Interaction Count
        ax5 = plt.subplot(3, 2, 5)
        self._plot_frame_interaction_count(ax5)

        # 6. Chain Pair Comparison
        ax6 = plt.subplot(3, 2, 6)
        self._plot_chain_pair_comparison(ax6)

        plt.tight_layout(pad=3.0)

        # Save the comprehensive plot
        plot_file = f"{output_prefix}_temporal_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Temporal plots saved to {plot_file}")
        plt.close()

        # Create individual detailed plots
        self._create_detailed_temporal_plots(output_prefix)

    def _plot_interaction_timeline(self, ax):
        """Plot interaction timeline showing when different types appear/disappear."""
        ax.set_title('Interaction Timeline by Type', fontsize=14, fontweight='bold')

        colors = ['#3498db', '#27ae60', '#9b59b6', '#e67e22', '#f1c40f', '#e74c3c']
        interaction_types = list(self.interaction_types.keys())

        for i, (pair_key, stats) in enumerate(self.chain_interaction_stats.items()):
            for j, interaction_type in enumerate(interaction_types):
                if interaction_type in stats['interaction_type_stats']:
                    type_stats = stats['interaction_type_stats'][interaction_type]
                    if type_stats['first_frame'] is not None and type_stats['last_frame'] is not None:
                        # Plot horizontal line for interaction duration
                        ax.hlines(y=f"{pair_key}_{interaction_type}",
                                xmin=type_stats['first_frame'],
                                xmax=type_stats['last_frame'],
                                colors=colors[j % len(colors)],
                                linewidth=3,
                                alpha=0.8,
                                label=f"{self.interaction_types[interaction_type]['label']}")

        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Chain Pair - Interaction Type', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_occupancy_heatmap(self, ax):
        """Plot occupancy heatmap for different interaction types."""
        ax.set_title('Interaction Type Occupancy Heatmap', fontsize=14, fontweight='bold')

        # Prepare data for heatmap
        interaction_types = list(self.interaction_types.keys())
        chain_pairs = list(self.chain_interaction_stats.keys())

        occupancy_data = []
        labels = []

        for pair_key in chain_pairs:
            stats = self.chain_interaction_stats[pair_key]
            row = []
            for interaction_type in interaction_types:
                if interaction_type in stats['interaction_type_stats']:
                    occupancy = stats['interaction_type_stats'][interaction_type]['occupancy']
                    row.append(occupancy)
                else:
                    row.append(0.0)
            occupancy_data.append(row)
            labels.append(pair_key)

        if occupancy_data:
            im = ax.imshow(occupancy_data, cmap='YlOrRd', aspect='auto')

            # Add text annotations
            for i in range(len(occupancy_data)):
                for j in range(len(occupancy_data[0])):
                    text = ax.text(j, i, f'{occupancy_data[i][j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)

            ax.set_xticks(range(len(interaction_types)))
            ax.set_xticklabels([self.interaction_types[it]['label'] for it in interaction_types],
                              rotation=45, ha='right')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)

            plt.colorbar(im, ax=ax, label='Occupancy')

    def _plot_duration_distribution(self, ax):
        """Plot distribution of interaction durations."""
        ax.set_title('Interaction Duration Distribution', fontsize=14, fontweight='bold')

        all_durations = []
        duration_labels = []

        for pair_key, stats in self.chain_interaction_stats.items():
            for interaction_type, type_stats in stats['interaction_type_stats'].items():
                if type_stats['duration'] > 0:
                    all_durations.append(type_stats['duration'])
                    duration_labels.append(f"{pair_key}_{interaction_type}")

        if all_durations:
            ax.hist(all_durations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Duration (frames)', fontsize=12)
            ax.set_ylabel('Number of Interactions', fontsize=12)
            ax.grid(True, alpha=0.3)

    def _plot_persistent_vs_transient(self, ax):
        """Plot persistent vs transient interactions."""
        ax.set_title('Persistent vs Transient Interactions', fontsize=14, fontweight='bold')

        persistent_counts = []
        transient_counts = []
        chain_labels = []

        for pair_key, stats in self.chain_interaction_stats.items():
            persistent_counts.append(len(stats['persistent_interactions']))
            transient_counts.append(len(stats['transient_interactions']))
            chain_labels.append(pair_key)

        if persistent_counts:
            x = np.arange(len(chain_labels))
            width = 0.35

            ax.bar(x - width/2, persistent_counts, width, label='Persistent (≥50%)',
                   color='#27ae60', alpha=0.8)
            ax.bar(x + width/2, transient_counts, width, label='Transient (<50%)',
                   color='#e74c3c', alpha=0.8)

            ax.set_xlabel('Chain Pairs', fontsize=12)
            ax.set_ylabel('Number of Interactions', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(chain_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_frame_interaction_count(self, ax):
        """Plot total interaction count per frame."""
        ax.set_title('Total Interactions per Frame', fontsize=14, fontweight='bold')

        if not self.frame_interactions:
            return

        frames = [fi['frame'] for fi in self.frame_interactions]
        interaction_counts = [len(fi['interactions']) for fi in self.frame_interactions]

        ax.plot(frames, interaction_counts, linewidth=2, color='#3498db', alpha=0.8)
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Number of Interactions', fontsize=12)
        ax.grid(True, alpha=0.3)

    def _plot_chain_pair_comparison(self, ax):
        """Plot comparison of interaction types between chain pairs."""
        ax.set_title('Interaction Type Comparison by Chain Pair', fontsize=14, fontweight='bold')

        interaction_types = list(self.interaction_types.keys())
        chain_pairs = list(self.chain_interaction_stats.keys())

        if not chain_pairs:
            return

        # Prepare data
        data = []
        for interaction_type in interaction_types:
            row = []
            for pair_key in chain_pairs:
                stats = self.chain_interaction_stats[pair_key]
                if interaction_type in stats['interaction_type_stats']:
                    count = stats['interaction_type_stats'][interaction_type]['total_frames']
                    row.append(count)
                else:
                    row.append(0)
            data.append(row)

        if data:
            x = np.arange(len(chain_pairs))
            width = 0.15

            for i, interaction_type in enumerate(interaction_types):
                ax.bar(x + i*width, data[i], width,
                       label=self.interaction_types[interaction_type]['label'],
                       alpha=0.8)

            ax.set_xlabel('Chain Pairs', fontsize=12)
            ax.set_ylabel('Total Frames with Interaction', fontsize=12)
            ax.set_xticks(x + width * (len(interaction_types) - 1) / 2)
            ax.set_xticklabels(chain_pairs, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

    def _create_detailed_temporal_plots(self, output_prefix):
        """Create individual detailed temporal plots."""
        # 1. Detailed timeline plot
        self._create_detailed_timeline_plot(output_prefix)

        # 2. Residue pair occupancy plot
        self._create_residue_occupancy_plot(output_prefix)

    def _create_detailed_timeline_plot(self, output_prefix):
        """Create detailed timeline plot showing all interactions."""
        if not self.frame_interactions:
            return

        fig, ax = plt.subplots(figsize=(16, 10))

        # Get all unique interactions
        all_interactions = set()
        for frame_data in self.frame_interactions:
            for interaction in frame_data['interactions']:
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))
                interaction_type = interaction['type']
                all_interactions.add(f"{res1}-{res2}-{interaction_type}")

        # Create timeline for each interaction
        interaction_timeline = {interaction: [] for interaction in all_interactions}

        for frame_data in self.frame_interactions:
            frame = frame_data['frame']
            frame_interactions = set()

            for interaction in frame_data['interactions']:
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))
                interaction_type = interaction['type']
                interaction_key = f"{res1}-{res2}-{interaction_type}"
                frame_interactions.add(interaction_key)

            for interaction in all_interactions:
                if interaction in frame_interactions:
                    interaction_timeline[interaction].append(frame)

        # Plot timeline
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_interactions)))

        for i, (interaction, frames) in enumerate(interaction_timeline.items()):
            if frames:
                ax.hlines(y=i, xmin=min(frames), xmax=max(frames),
                         colors=colors[i], linewidth=2, alpha=0.8)

        ax.set_title('Detailed Interaction Timeline', fontsize=16, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=14)
        ax.set_ylabel('Interactions', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Save plot
        timeline_file = f"{output_prefix}_detailed_timeline.png"
        plt.savefig(timeline_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Detailed timeline plot saved to {timeline_file}")

    def _create_residue_occupancy_plot(self, output_prefix):
        """Create residue pair occupancy plot."""
        if not self.chain_interaction_stats:
            return

        fig, ax = plt.subplots(figsize=(14, 10))

        # Get top residue pairs by occupancy
        all_residue_stats = []
        for pair_key, stats in self.chain_interaction_stats.items():
            for res_pair, res_stats in stats['residue_pair_stats'].items():
                all_residue_stats.append({
                    'pair': res_pair,
                    'occupancy': res_stats['occupancy'],
                    'chain_pair': pair_key
                })

        # Sort by occupancy and take top 20
        all_residue_stats.sort(key=lambda x: x['occupancy'], reverse=True)
        top_residues = all_residue_stats[:20]

        if top_residues:
            residues = [item['pair'] for item in top_residues]
            occupancies = [item['occupancy'] * 100 for item in top_residues]
            colors = ['#27ae60' if occ >= 50 else '#e74c3c' for occ in occupancies]

            bars = ax.barh(range(len(residues)), occupancies, color=colors, alpha=0.8)
            ax.set_yticks(range(len(residues)))
            ax.set_yticklabels(residues)
            ax.set_xlabel('Occupancy (%)', fontsize=14)
            ax.set_title('Top 20 Residue Pair Occupancies', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels on bars
            for i, (bar, occ) in enumerate(zip(bars, occupancies)):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{occ:.1f}%', va='center', fontsize=10)

        # Save plot
        occupancy_file = f"{output_prefix}_residue_occupancy.png"
        plt.savefig(occupancy_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Residue occupancy plot saved to {occupancy_file}")

def main():
    """Main function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Protein-Protein Interaction Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic A-B chain analysis
  python ppi_analyzer.py --cms structure.cms

  # Custom chain groups
  python ppi_analyzer.py --cms structure.cms --chain-groups A,B C,D

  # Trajectory analysis with occupancy
  python ppi_analyzer.py --cms structure.cms --traj trajectory_dir --max-frames 500 --occupancy-threshold 0.3

  # Trajectory analysis with temporal dynamics (default)
  python ppi_analyzer.py --cms structure.cms --traj trajectory_dir --max-frames 1000

  # Structure-only analysis with custom output
  python ppi_analyzer.py --cms structure.cms --output my_analysis
        """
    )

    parser.add_argument('--cms', required=True,
                       help='CMS structure file')
    parser.add_argument('--traj',
                       help='Trajectory file/directory (optional)')
    parser.add_argument('--chain-groups', nargs='+',
                       help='Chain groups to analyze (e.g., A,B C,D for A-B vs C-D)')
    parser.add_argument('--output', default='ppi_analysis',
                       help='Output file prefix (default: ppi_analysis)')
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Maximum number of trajectory frames to analyze (default: 1000)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of worker threads for parallel processing (default: 4)')
    parser.add_argument('--occupancy-threshold', type=float, default=0.2,
                       help='Minimum occupancy threshold (0.0-1.0, default: 0.2)')
    parser.add_argument('--temporal-analysis', action='store_true',
                       help='Perform detailed temporal analysis of interaction dynamics')

    args = parser.parse_args()

    # Parse chain groups
    chain_groups = None
    if args.chain_groups:
        chain_groups = [group.split(',') for group in args.chain_groups]

    # Create analyzer
    analyzer = ComprehensivePPIAnalyzer(
        cms_file=args.cms,
        trajectory_file=args.traj,
        chain_groups=chain_groups,
        occupancy_threshold=args.occupancy_threshold
    )

    try:
        # Run analysis
        analyzer.validate_files()
        analyzer.load_structure()
        analyzer.extract_residue_info()

        # Analyze interactions
        if args.traj:
            interactions = analyzer.analyze_trajectory_with_occupancy(max_frames=args.max_frames, max_workers=args.max_workers)

            # Perform temporal analysis by default when trajectory is provided
            print("\n" + "="*60)
            print("⏰ PERFORMING TEMPORAL INTERACTION ANALYSIS (DEFAULT)")
            print("="*60)
            print(f"🔍 DEBUG: Running temporal analysis by default with trajectory")
            analyzer.analyze_temporal_interactions(max_frames=args.max_frames, max_workers=args.max_workers)
            analyzer.print_temporal_analysis()
            analyzer.save_temporal_analysis(f"{args.output}_temporal")

            # Also run if explicitly requested (for backward compatibility)
            if args.temporal_analysis:
                print(f"🔍 DEBUG: Temporal analysis flag is also set: {args.temporal_analysis}")
        else:
            interactions = analyzer.analyze_frame(analyzer.structure)

        # Create visualization
        if interactions:
            G = analyzer.create_network_graph(interactions)
            analyzer.create_network_diagram(G, interactions, f"{args.output}_network.png")
            analyzer.print_summary(interactions)
            analyzer.print_occupancy_summary(interactions)
            analyzer.save_results(interactions, args.output)

            print(f"\n🎉 Analysis completed successfully!")
            print(f"📊 Found {len(interactions)} interactions")
            print(f"📁 Output files:")
            print(f"   Network diagram: {args.output}_network.png")
            print(f"   Interactions data: {args.output}_interactions.json")
            print(f"   Summary report: {args.output}_summary.txt")

            if args.traj:
                print(f"   Temporal statistics: {args.output}_temporal_statistics.json")
                print(f"   Frame-by-frame data: {args.output}_temporal_frame_data.json")
                print(f"   Temporal summary: {args.output}_temporal_summary.csv")
                print(f"   Comprehensive temporal plots: {args.output}_temporal_temporal_plots.png")
                print(f"   Detailed timeline: {args.output}_temporal_detailed_timeline.png")
                print(f"   Residue occupancy: {args.output}_temporal_residue_occupancy.png")
        else:
            print("⚠️ No significant interactions found")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()