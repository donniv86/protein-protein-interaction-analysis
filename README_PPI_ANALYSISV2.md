# Protein-Protein Interaction Analysis V2

## Overview

`ppi_analysisV2.py` is an enhanced version of the protein-protein interaction analysis tool designed to analyze, visualize, and characterize interactions between protein chains. This script offers improved visualization capabilities, better performance with large datasets, and more accurate chain naming conventions (A & L).

## Key Features

- **Advanced Interaction Detection**: Identifies hydrogen bonds, salt bridges, π-π stacking, π-cation interactions, and hydrophobic contacts
- **Professional Visualizations**: Generates publication-quality network diagrams with customizable styling
- **Trajectory Analysis**: Supports analysis of molecular dynamics trajectories with occupancy calculation
- **Chain-Specific Analysis**: Default chain names (A & L) for standard protein-protein interaction analysis
- **Multi-format Output**: Generates both PNG (for presentations) and SVG (vector graphics for publication)
- **Performance Optimizations**: Handles larger datasets efficiently with multi-threading support

## Installation Requirements

- Python 3.6+
- Schrodinger Python API 2025-2
- Required Python packages:
  - numpy
  - matplotlib
  - networkx
  - scipy (optional, for advanced features)

## Usage

### Basic Usage

```bash
python ppi_analysisV2.py --cms structure.cms --chain-groups A,B L,M
```

### With Trajectory Analysis

```bash
python ppi_analysisV2.py --cms structure.cms --traj trajectory_dir --occupancy-threshold 0.3
```

### Customizing Output

```bash
python ppi_analysisV2.py --cms structure.cms --output-prefix my_analysis --output-dir results
```

## Command Line Arguments

- `--cms`: Path to the Desmond CMS structure file (required)
- `--traj`: Path to the trajectory file or directory (optional)
- `--chain-groups`: Chain groups to analyze (e.g., A,B L,M). Default is A vs L
- `--occupancy-threshold`: Minimum occupancy threshold (default: 0.2)
- `--output-prefix`: Prefix for output files (default: ppi_analysis)
- `--output-dir`: Output directory name (default: ppi_results)
- `--debug`: Enable detailed debug logging
- `--max-frames`: Maximum number of trajectory frames to analyze
- `--skip-single-interactions`: Skip plots for interaction types with only one interaction

## Output Files

The script generates the following outputs in the specified output directory:

- `{prefix}_consolidated.png/.svg`: Network diagram showing all interaction types
- `{prefix}_hydrogen_bonds.png/.svg`: Network diagram for hydrogen bonds
- `{prefix}_salt_bridges.png/.svg`: Network diagram for salt bridges
- `{prefix}_pi_interactions.png/.svg`: Network diagram for π-π and π-cation interactions
- `{prefix}_data.json`: Raw interaction data in JSON format
- `{prefix}_statistics.txt`: Summary statistics of detected interactions

## Visualization Features

The network visualizations include:

- Residue nodes colored by amino acid type
- Chain-specific highlighting (Chain A in red, Chain L in light salmon)
- Interaction edges styled by interaction type
- Occupancy percentages shown for trajectory analysis
- Professional formatting with clear labels and legends

## Example

A typical workflow involves:

1. Prepare your protein structure (CMS file)
2. Run the analysis script
3. Examine the generated network diagrams
4. Review interaction statistics

## Improvements in V2

- Fixed chain naming to use A & L as default chains
- Enhanced network layout algorithms for better visualization
- Improved label placement to prevent overlaps
- Added SVG export for publication-quality figures
- Optimized performance for large trajectory analysis

## Troubleshooting

- If you encounter "Chain not found" errors, verify the chain names in your structure file
- For memory issues with large trajectories, use the `--max-frames` option
- Ensure proper Schrodinger environment is loaded (`~/schrod25-2.ve`)

## Contributing

To contribute to this project:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License

Copyright Schrodinger, LLC. All rights reserved.

## Contact

For questions and support, please contact:
- GitHub: [donniv86](https://github.com/donniv86/protein-protein-interaction-analysis)
