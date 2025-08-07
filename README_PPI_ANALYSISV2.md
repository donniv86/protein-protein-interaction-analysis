# ppi_analysisV2.py - Enhanced Protein-Protein Interaction Analysis

## Overview
An advanced tool for analyzing, visualizing, and characterizing protein-protein interactions from molecular structures and trajectories. This V2 version offers improved visualization and properly handles A & L chain naming conventions.

## Features
- Detects multiple interaction types (H-bonds, salt bridges, π-π, π-cation)
- Generates publication-quality network visualizations
- Supports molecular dynamics trajectory analysis
- Calculates interaction occupancy percentages
- Produces both PNG and SVG outputs

## Requirements
- Python 3.6+
- Schrodinger Python API (2025-2)
- numpy, matplotlib, networkx

## Quick Usage
```bash
# Basic analysis with default chains (A & L)
python ppi_analysisV2.py --cms structure.cms

# With trajectory analysis
python ppi_analysisV2.py --cms structure.cms --traj trajectory_dir --occupancy-threshold 0.3

# Custom chain groups
python ppi_analysisV2.py --cms structure.cms --chain-groups A,B L,M
```

## Output
- Network diagrams showing protein-protein interactions
- Statistical summaries of interaction types
- JSON data files with detailed interaction information

## Documentation
For complete documentation, see [README_PPI_ANALYSISV2.md](./protein-protein-interaction-analysis/README_PPI_ANALYSISV2.md)

## License
Copyright Schrodinger, LLC. All rights reserved.
