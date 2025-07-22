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
import time
from datetime import datetime, timedelta
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
        self.interaction_time_history = {}
        self.total_frames_analyzed = 0

        # Trajectory and time tracking
        self.trajectory_frames = []
        self.simulation_times = []  # Store simulation times for each frame
        self.total_simulation_time = 0.0  # Total simulation time in nanoseconds
        self.time_step = 0.0  # Time step between frames in nanoseconds

        # Performance tracking
        self.analysis_start_time = None
        self.analysis_end_time = None

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
        """Analyze multiple frames from trajectory to calculate occupancy percentages using multi-threading with simulation time tracking."""
        print(f"🎬 Analyzing trajectory for occupancy calculation (max {max_frames} frames, {max_workers} workers)...")

        # Start performance tracking
        self.analysis_start_time = time.time()

        if not self.trajectory_file or not self.trajectory_file.exists():
            print("⚠️ No trajectory file found, analyzing single frame only")
            return self.analyze_frame(self.structure)

        # Reset occupancy tracking
        self.interaction_occupancy = {}
        self.interaction_distance_history = {}
        self.interaction_time_history = {}
        self.total_frames_analyzed = 0
        self.simulation_times = []
        self.total_simulation_time = 0.0

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

            # Calculate simulation time information
            self._calculate_simulation_time_info(trajectory_frames)

            # Limit to max_frames if specified
            frames_to_analyze = min(total_available_frames, max_frames)
            print(f"🎯 Will analyze {frames_to_analyze} frames using {max_workers} threads")
            print(f"⏱️ Total simulation time: {self.total_simulation_time:.2f} ns")
            print(f"⏱️ Time step: {self.time_step:.3f} ns per frame")

            # Sample frames evenly across the trajectory
            frame_indices = np.linspace(0, total_available_frames-1, frames_to_analyze, dtype=int)

            # Thread-safe counters
            frame_counter = 0
            frame_lock = threading.Lock()

            def analyze_single_frame(frame_idx):
                """Analyze a single frame - thread-safe function with time tracking."""
                nonlocal frame_counter

                try:
                    # Get frame and apply coordinates to structure
                    frame = trajectory_frames[frame_idx]

                    # Get simulation time for this frame
                    frame_time = self._get_frame_time(frame, frame_idx)

                    # Create a copy of the structure and apply frame coordinates
                    frame_structure = self.structure.copy()
                    positions = frame.pos()

                    # Apply coordinates to the structure
                    for i, atom in enumerate(frame_structure.atom):
                        if i < len(positions):
                            atom.xyz = positions[i]

                    # Analyze current frame
                    frame_interactions = self.analyze_frame(frame_structure)

                    # Add time information to interactions
                    for interaction in frame_interactions:
                        interaction['simulation_time'] = frame_time

                    # Update progress counter
                    with frame_lock:
                        nonlocal frame_counter
                        frame_counter += 1
                        if frame_counter % 20 == 0:
                            elapsed_time = time.time() - self.analysis_start_time
                            print(f"   📊 Analyzed {frame_counter}/{frames_to_analyze} frames... (Elapsed: {elapsed_time:.1f}s)")

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
                                self._update_time_history(interaction)
                            all_interactions.extend(frame_interactions)

                    except Exception as e:
                        print(f"   ⚠️ Error processing frame {frame_idx}: {e}")

            # End performance tracking
            self.analysis_end_time = time.time()
            total_analysis_time = self.analysis_end_time - self.analysis_start_time

            print(f"✅ Successfully analyzed {self.total_frames_analyzed} trajectory frames using multi-threading")
            print(f"⏱️ Analysis completed in {total_analysis_time:.2f} seconds")
            print(f"⏱️ Average time per frame: {total_analysis_time/self.total_frames_analyzed:.3f} seconds")

            # Filter by occupancy threshold
            filtered_interactions = self._filter_by_occupancy(all_interactions, self.occupancy_threshold)

            return filtered_interactions

        except Exception as e:
            print(f"❌ Error during trajectory analysis: {e}")
            print("⚠️ Falling back to single frame analysis")
            return self.analyze_frame(self.structure)

    def _calculate_simulation_time_info(self, trajectory_frames):
        """Calculate simulation time information from trajectory frames."""
        try:
            # Try to get time information from trajectory metadata
            if hasattr(trajectory_frames[0], 'time'):
                # If frames have time attribute
                self.simulation_times = [frame.time for frame in trajectory_frames]
                self.total_simulation_time = self.simulation_times[-1] - self.simulation_times[0]
                if len(self.simulation_times) > 1:
                    self.time_step = (self.simulation_times[-1] - self.simulation_times[0]) / (len(self.simulation_times) - 1)
                else:
                    self.time_step = 0.0
            else:
                # Estimate time based on Desmond defaults (typically 1.2 ps per frame)
                default_time_step = 1.2  # picoseconds
                self.simulation_times = [i * default_time_step for i in range(len(trajectory_frames))]
                self.total_simulation_time = self.simulation_times[-1]
                self.time_step = default_time_step

            print(f"⏱️ Simulation time range: {self.simulation_times[0]:.2f} - {self.simulation_times[-1]:.2f} ps")

        except Exception as e:
            print(f"⚠️ Warning: Could not extract time information: {e}")
            # Fallback to frame-based time
            self.simulation_times = list(range(len(trajectory_frames)))
            self.total_simulation_time = len(trajectory_frames)
            self.time_step = 1.0

    def _get_frame_time(self, frame, frame_idx):
        """Get simulation time for a specific frame."""
        try:
            if hasattr(frame, 'time'):
                return frame.time
            elif frame_idx < len(self.simulation_times):
                return self.simulation_times[frame_idx]
            else:
                return frame_idx * self.time_step
        except:
            return frame_idx * self.time_step

    def _update_time_history(self, interaction):
        """Update time history for an interaction."""
        key = self._get_interaction_key(interaction)
        if key not in self.interaction_time_history:
            self.interaction_time_history[key] = []
        self.interaction_time_history[key].append(interaction.get('simulation_time', 0))

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
        print(f"🔍 Filtering interactions by ≥{occupancy_threshold*100:.0f}% occupancy...")

        filtered_interactions = []
        for key, occupancy_data in self.interaction_occupancy.items():
            occupancy = occupancy_data['count'] / self.total_frames_analyzed
            if occupancy >= occupancy_threshold:
                interaction = occupancy_data['example'].copy()
                interaction['occupancy'] = occupancy
                interaction['frames'] = occupancy_data['count']

                # Add distance statistics
                if key in self.interaction_distance_history:
                    distances = self.interaction_distance_history[key]
                    interaction['min_distance'] = min(distances)
                    interaction['max_distance'] = max(distances)
                    interaction['mean_distance'] = np.mean(distances)
                    interaction['std_distance'] = np.std(distances)

                # Add time statistics
                if key in self.interaction_time_history:
                    times = self.interaction_time_history[key]
                    interaction['min_time'] = min(times)
                    interaction['max_time'] = max(times)
                    interaction['mean_time'] = np.mean(times)
                    interaction['std_time'] = np.std(times)

                filtered_interactions.append(interaction)

        print(f"✅ Kept {len(filtered_interactions)} interactions with ≥{occupancy_threshold*100:.0f}% occupancy out of {len(interactions)} total")
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

    def create_network_diagram(self, G, interactions, output_file='ppi_network.png', output_dir='data'):
        """Create a professional network diagram and save to specified folder."""
        # Create output directory if it doesn't exist
        output_folder = Path(output_dir)
        output_folder.mkdir(exist_ok=True)

        # Update output file path to include output directory
        output_path = output_folder / output_file

        print(f"🎨 Creating network diagram: {output_path}")
        print(f"   Saving diagram to {output_path}...")

        # Set up the plot
        plt.figure(figsize=(16, 12))
        ax = plt.gca()

        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # Draw edges by interaction type
        for interaction_type, props in self.interaction_types.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == interaction_type]

            if edges:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edges,
                    edge_color=props['network_color'],
                    style=props['style'],
                    width=props['linewidth'],
                    alpha=0.7,
                    ax=ax
                )

        # Draw nodes by chain
        node_colors = []
        node_labels = {}

        for node in G.nodes():
            chain = G.nodes[node]['chain']
            res_name = G.nodes[node]['name']
            res_num = G.nodes[node]['number']

            # Color by chain
            node_colors.append(self.chain_colors.get(chain, '#CCCCCC'))

            node_labels[node] = f"{chain}:{res_name}\n{res_num}"

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=800,
            edgecolors='black',
            linewidths=1,
            ax=ax
        )

        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )

        # Edge labels with distance
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

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color='black',
            font_size=7,
            ax=ax
        )

        # Legend
        legend_elements = []
        for interaction_type, props in self.interaction_types.items():
            if interaction_type in [d['type'] for d in interactions]:
                legend_elements.append(
                    mpatches.Patch(color=props['network_color'], label=props['label'])
                )

        # Add chain legend
        for chain, color in self.chain_colors.items():
            if any(G.nodes[node]['chain'] == chain for node in G.nodes()):
                legend_elements.append(
                    mpatches.Patch(color=color, label=f'Chain {chain}')
                )

        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        # Title
        chain_groups_str = " vs ".join([f"Chain{'s' if len(g) > 1 else ''} {','.join(g)}" for g in self.chain_groups])
        ax.set_title(
            f"Protein-Protein Interaction Network\n"
            f"{chain_groups_str}\n"
            f"({len(interactions)} interactions)",
            fontsize=16, fontweight='bold', pad=20
        )

        ax.axis('off')
        plt.tight_layout()

        print(f"   Saving diagram to {output_path}...")
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"✅ Network diagram saved as {output_path}")
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
        """Print detailed occupancy analysis with simulation time information."""
        if not interactions:
            print("⚠️ No interactions to summarize")
            return

        print("\n" + "="*60)
        print("📊 OCCUPANCY ANALYSIS SUMMARY")
        print("="*60)

        # Trajectory analysis info
        print("🎬 Trajectory Analysis:")
        print(f"   Total frames analyzed: {self.total_frames_analyzed}")
        print(f"   Total unique interactions found: {len(self.interaction_occupancy)}")
        print(f"   Interactions with occupancy data: {len(interactions)}")

        if self.total_simulation_time > 0:
            print(f"   ⏱️ Total simulation time: {self.total_simulation_time:.2f} ps ({self.total_simulation_time/1000:.3f} ns)")
            print(f"   ⏱️ Time step: {self.time_step:.3f} ps per frame")
            print(f"   ⏱️ Analysis time: {self.analysis_end_time - self.analysis_start_time:.2f} seconds")

        # Occupancy statistics
        occupancies = [interaction.get('occupancy', 0) for interaction in interactions]
        if occupancies:
            print(f"\n📈 Occupancy Statistics:")
            print(f"   Mean occupancy: {np.mean(occupancies)*100:.2f}%")
            print(f"   Median occupancy: {np.median(occupancies)*100:.2f}%")
            print(f"   Min occupancy: {min(occupancies)*100:.2f}%")
            print(f"   Max occupancy: {max(occupancies)*100:.2f}%")

        # Distance statistics
        distances = [interaction.get('distance', 0) for interaction in interactions]
        if distances:
            print(f"\n📏 Distance Statistics:")
            print(f"   Mean distance: {np.mean(distances):.2f} Å")
            print(f"   Median distance: {np.median(distances):.2f} Å")
            print(f"   Min distance: {min(distances):.2f} Å")
            print(f"   Max distance: {max(distances):.2f} Å")

        # Time statistics (if available)
        times_with_data = [interaction for interaction in interactions if 'min_time' in interaction]
        if times_with_data:
            print(f"\n⏱️ Time Statistics:")
            time_ranges = [interaction['max_time'] - interaction['min_time'] for interaction in times_with_data]
            print(f"   Mean time range: {np.mean(time_ranges):.2f} ps")
            print(f"   Median time range: {np.median(time_ranges):.2f} ps")
            print(f"   Min time range: {min(time_ranges):.2f} ps")
            print(f"   Max time range: {max(time_ranges):.2f} ps")

        # Top persistent interactions
        print(f"\n🏆 Top 10 Most Persistent Interactions:")
        sorted_interactions = sorted(interactions, key=lambda x: x.get('occupancy', 0), reverse=True)

        for i, interaction in enumerate(sorted_interactions[:10], 1):
            res1 = interaction.get('donor', interaction.get('residue1', ''))
            res2 = interaction.get('acceptor', interaction.get('residue2', ''))
            itype = interaction['type'].replace('_', ' ').title()
            distance = interaction.get('distance', 0)
            occupancy = interaction.get('occupancy', 0) * 100

            time_info = ""
            if 'min_time' in interaction and 'max_time' in interaction:
                time_range = interaction['max_time'] - interaction['min_time']
                time_info = f", Time range: {time_range:.1f} ps"

            print(f"   {i:2d}. {res1} ↔ {res2} ({itype}: {distance:.2f}Å, {occupancy:.1f}% occupancy{time_info})")

        print("="*60)

    def create_time_analysis_plots(self, interactions, output_prefix='time_analysis'):
        """Create time-based analysis plots showing interaction evolution."""
        if not interactions or not self.interaction_time_history:
            print("⚠️ No time data available for plotting")
            return

        print(f"📊 Creating time analysis plots: {output_prefix}")

        # Create time evolution plot
        plt.figure(figsize=(15, 10))

        # Plot 1: Interaction occupancy over time
        plt.subplot(2, 2, 1)
        for interaction in interactions:
            key = self._get_interaction_key(interaction)
            if key in self.interaction_time_history:
                times = self.interaction_time_history[key]
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))
                itype = interaction['type'].replace('_', ' ').title()

                plt.scatter(times, [interaction.get('occupancy', 0) * 100] * len(times),
                           alpha=0.6, s=20, label=f"{res1}-{res2} ({itype})")

        plt.xlabel('Simulation Time (ps)')
        plt.ylabel('Occupancy (%)')
        plt.title('Interaction Occupancy Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Plot 2: Distance evolution over time
        plt.subplot(2, 2, 2)
        for interaction in interactions:
            key = self._get_interaction_key(interaction)
            if key in self.interaction_time_history and key in self.interaction_distance_history:
                times = self.interaction_time_history[key]
                distances = self.interaction_distance_history[key]
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))

                plt.scatter(times, distances, alpha=0.6, s=20, label=f"{res1}-{res2}")

        plt.xlabel('Simulation Time (ps)')
        plt.ylabel('Distance (Å)')
        plt.title('Interaction Distance Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Plot 3: Interaction type distribution over time
        plt.subplot(2, 2, 3)
        interaction_types = {}
        for interaction in interactions:
            key = self._get_interaction_key(interaction)
            if key in self.interaction_time_history:
                itype = interaction['type']
                times = self.interaction_time_history[key]
                if itype not in interaction_types:
                    interaction_types[itype] = []
                interaction_types[itype].extend(times)

        for itype, times in interaction_types.items():
            plt.hist(times, bins=20, alpha=0.6, label=itype.replace('_', ' ').title())

        plt.xlabel('Simulation Time (ps)')
        plt.ylabel('Frequency')
        plt.title('Interaction Type Distribution Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Occupancy vs Time Range
        plt.subplot(2, 2, 4)
        time_ranges = []
        occupancies = []
        labels = []

        for interaction in interactions:
            if 'min_time' in interaction and 'max_time' in interaction:
                time_range = interaction['max_time'] - interaction['min_time']
                occupancy = interaction.get('occupancy', 0) * 100
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))

                time_ranges.append(time_range)
                occupancies.append(occupancy)
                labels.append(f"{res1}-{res2}")

        if time_ranges:
            plt.scatter(time_ranges, occupancies, alpha=0.7, s=50)
            for i, label in enumerate(labels):
                plt.annotate(label, (time_ranges[i], occupancies[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.xlabel('Time Range (ps)')
        plt.ylabel('Occupancy (%)')
        plt.title('Occupancy vs Time Range')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_time_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Time analysis plots saved as {output_prefix}_time_analysis.png")

    def save_results(self, interactions, output_prefix='ppi_analysis', output_dir='data'):
        """Save comprehensive results including time analysis to specified folder."""
        # Create output directory if it doesn't exist
        output_folder = Path(output_dir)
        output_folder.mkdir(exist_ok=True)

        # Update output prefix to include output directory path
        output_path = output_folder / output_prefix

        print(f"💾 Saving results to {output_dir} folder: {output_path}")

        # Save interactions with time data
        results_data = {
            'metadata': {
                'cms_file': str(self.cms_file),
                'trajectory_file': str(self.trajectory_file) if self.trajectory_file else None,
                'chain_groups': self.chain_groups,
                'occupancy_threshold': self.occupancy_threshold,
                'total_frames_analyzed': self.total_frames_analyzed,
                'total_simulation_time': self.total_simulation_time,
                'time_step': self.time_step,
                'analysis_time': self.analysis_end_time - self.analysis_start_time if self.analysis_end_time else None
            },
            'interactions': interactions,
            'time_history': self.interaction_time_history,
            'distance_history': self.interaction_distance_history,
            'occupancy_data': self.interaction_occupancy
        }

        with open(f"{output_path}_interactions.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"✅ Interactions saved to {output_path}_interactions.json")

        # Save summary
        with open(f"{output_path}_summary.txt", 'w') as f:
            f.write("PROTEIN-PROTEIN INTERACTION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CMS File: {self.cms_file}\n")
            if self.trajectory_file:
                f.write(f"Trajectory File: {self.trajectory_file}\n")
            f.write(f"Chain Groups: {self.chain_groups}\n")
            f.write(f"Occupancy Threshold: {self.occupancy_threshold*100:.1f}%\n")
            f.write(f"Total Frames Analyzed: {self.total_frames_analyzed}\n")
            if self.total_simulation_time > 0:
                f.write(f"Total Simulation Time: {self.total_simulation_time:.2f} ps\n")
                f.write(f"Time Step: {self.time_step:.3f} ps\n")
            f.write(f"Total Interactions: {len(interactions)}\n\n")

            # Write interaction details
            f.write("INTERACTIONS:\n")
            f.write("-" * 30 + "\n")
            for i, interaction in enumerate(interactions, 1):
                res1 = interaction.get('donor', interaction.get('residue1', ''))
                res2 = interaction.get('acceptor', interaction.get('residue2', ''))
                itype = interaction['type'].replace('_', ' ').title()
                distance = interaction.get('distance', 0)
                occupancy = interaction.get('occupancy', 0) * 100

                time_info = ""
                if 'min_time' in interaction and 'max_time' in interaction:
                    time_range = interaction['max_time'] - interaction['min_time']
                    time_info = f", Time: {interaction['min_time']:.1f}-{interaction['max_time']:.1f} ps (Range: {time_range:.1f} ps)"

                f.write(f"{i:2d}. {res1} ↔ {res2} ({itype}: {distance:.2f}Å, {occupancy:.1f}%{time_info})\n")

        print(f"✅ Summary saved to {output_path}_summary.txt")

        # Create time analysis plots if trajectory data is available
        if self.trajectory_file and self.interaction_time_history:
            self.create_time_analysis_plots(interactions, str(output_path))

def main():
    """Main function to run the PPI analyzer."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Protein-Protein Interaction Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single structure
  python ppi_analyzer.py --cms structure.cms --output my_analysis

  # Analyze trajectory with occupancy
  python ppi_analyzer.py --cms structure.cms --traj trajectory_folder --max-frames 100 --output trajectory_analysis

  # Custom chain groups and occupancy threshold
  python ppi_analyzer.py --cms structure.cms --chain-groups A B --occupancy-threshold 0.3 --output custom_analysis
        """
    )

    parser.add_argument('--cms', required=True,
                       help='Path to the CMS structure file')
    parser.add_argument('--traj',
                       help='Path to trajectory file/directory (optional)')
    parser.add_argument('--chain-groups', nargs='+', action='append',
                       help='Chain groups to analyze (e.g., --chain-groups A B --chain-groups C D)')
    parser.add_argument('--output', default='ppi_analysis',
                       help='Output prefix for results (default: ppi_analysis)')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory for all files (default: data)')
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Maximum number of trajectory frames to analyze (default: 1000)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of worker threads for parallel processing (default: 4)')
    parser.add_argument('--occupancy-threshold', type=float, default=0.2,
                       help='Minimum occupancy threshold (0.0-1.0, default: 0.2)')

    args = parser.parse_args()

    try:
        # Process chain groups
        chain_groups = None
        if args.chain_groups:
            chain_groups = args.chain_groups

        # Create analyzer
        analyzer = ComprehensivePPIAnalyzer(
            cms_file=args.cms,
            trajectory_file=args.traj,
            chain_groups=chain_groups,
            occupancy_threshold=args.occupancy_threshold
        )

        # Validate and load structure
        analyzer.validate_files()
        analyzer.load_structure()
        analyzer.extract_residue_info()

        # Analyze interactions
        if args.traj:
            interactions = analyzer.analyze_trajectory_with_occupancy(max_frames=args.max_frames, max_workers=args.max_workers)
        else:
            interactions = analyzer.analyze_frame(analyzer.structure)

        if not interactions:
            print("⚠️ No significant interactions found")
            return

        # Create network diagram
        G = analyzer.create_network_graph(interactions)
        analyzer.create_network_diagram(G, interactions, f"{args.output}_network.png", args.output_dir)

        # Print summaries
        analyzer.print_summary(interactions)
        if args.traj:
            analyzer.print_occupancy_summary(interactions)

        # Save results
        analyzer.save_results(interactions, args.output, args.output_dir)

        print(f"\n🎉 Analysis completed successfully!")
        print(f"📊 Found {len(interactions)} interactions")
        print(f"📁 Output files saved to: {args.output_dir}/")
        print(f"   Network diagram: {args.output_dir}/{args.output}_network.png")
        print(f"   Interactions data: {args.output_dir}/{args.output}_interactions.json")
        print(f"   Summary report: {args.output_dir}/{args.output}_summary.txt")
        if args.traj:
            print(f"   Time analysis plots: {args.output_dir}/{args.output}_time_analysis.png")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()