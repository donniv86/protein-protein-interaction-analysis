# Protein-Protein Interaction Analysis Pipeline

A comprehensive Python pipeline for analyzing and visualizing protein-protein interactions from molecular dynamics simulations using Schrödinger's Desmond.

## 🚀 Features

- **Advanced PPI Analysis**: Detect hydrogen bonds, salt bridges, π-π stacking, and π-cation interactions
- **Professional Visualizations**: High-quality network diagrams with perfect circular nodes and bidirectional arrows
- **Organized Output**: Structured output with separate folders for images, data, and logs
- **Scalable Vector Graphics**: SVG output for publication-ready figures
- **Interactive Analysis**: Configurable occupancy thresholds and chain group selection
- **Comprehensive Logging**: Detailed analysis logs with progress tracking

## 📋 Requirements

- Python 3.7+
- Schrödinger Suite 2025-2 or later
- Desmond molecular dynamics software
- Required Python packages:
  - NetworkX
  - Matplotlib
  - NumPy
  - Schrödinger Python API

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/donniv86/protein-protein-interaction-analysis.git
   cd protein-protein-interaction-analysis
   ```

2. **Set up Schrödinger environment**:
   ```bash
   # Activate Schrödinger virtual environment
   source /path/to/schrodinger.ve/bin/activate
   ```

3. **Verify installation**:
   ```bash
   python ppi_analysis.py --help
   ```

## 📖 Usage

### Basic Analysis

```bash
python ppi_analysis.py \
  --cms desmond_md_job_pro-pro/desmond_md_job_pro-pro-out.cms \
  --traj desmond_md_job_pro-pro/desmond_md_job_pro-pro_trj \
  --chain-groups A L \
  --occupancy-threshold 0.05 \
  --output-prefix my_analysis \
  --output-dir my_results
```

### Advanced Options

```bash
python ppi_analysis.py \
  --cms structure.cms \
  --traj trajectory_trj \
  --chain-groups A B C D \
  --occupancy-threshold 0.1 \
  --output-prefix detailed_analysis \
  --output-dir results \
  --max-frames 1000 \
  --debug \
  --skip-single-interactions
```

## 🔧 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cms` | str | **Required** | Path to Desmond .cms structure file |
| `--traj` | str | **Required** | Path to trajectory file or directory |
| `--chain-groups` | str | None | Chain groups to analyze (e.g., A,B C,D) |
| `--occupancy-threshold` | float | 0.2 | Minimum occupancy threshold |
| `--output-prefix` | str | ppi_analysis | Prefix for output files |
| `--output-dir` | str | ppi_results | Output directory name |
| `--max-frames` | int | None | Maximum frames to analyze |
| `--debug` | flag | False | Enable detailed logging |
| `--skip-single-interactions` | flag | False | Skip plots with only 1 interaction |

## 📁 Output Structure

The analysis creates an organized output directory structure:

```
output_directory/
├── 📸 images/
│   ├── prefix_consolidated.png          # All interactions combined
│   ├── prefix_consolidated.svg          # Vector version
│   ├── prefix_interactions_hydrogen_bond.png
│   ├── prefix_interactions_hydrogen_bond.svg
│   ├── prefix_interactions_salt-bridge.png
│   ├── prefix_interactions_salt-bridge.svg
│   ├── prefix_interactions_pi-pi.png
│   └── prefix_interactions_pi-pi.svg
├── 📊 data/
│   └── persistent_interactions.json     # Raw interaction data
└── 📝 logs/
    └── ppi_analysis.log                 # Analysis log
```

## 🎨 Visualization Features

### Network Diagrams
- **Perfect Circular Nodes**: Amino acid residues displayed as perfect circles
- **Bidirectional Arrows**: Interaction lines show arrows in both directions
- **Color-Coded Interactions**:
  - 🔵 Blue: Salt bridges
  - 🟣 Purple: Hydrogen bonds
  - 🟢 Green: π-π stacking
- **Occupancy Labels**: Percentage values on interaction lines
- **Professional Layout**: Optimized spacing and positioning

### Interactive Elements
- **Node Ordering**: Residues ordered by interaction count
- **Staggered Labels**: Edge labels positioned to avoid overlap
- **Scalable Output**: SVG format for vector editing
- **Publication Ready**: High-resolution PNG and SVG outputs

## 📊 Analysis Types

### 1. Hydrogen Bonds
- Backbone-backbone interactions
- Sidechain-sidechain interactions
- Backbone-sidechain interactions
- Self-interactions (intra-chain)

### 2. Salt Bridges
- Electrostatic interactions between charged residues
- Distance-based detection
- Occupancy calculation

### 3. π-π Stacking
- Aromatic residue interactions
- Geometric stacking detection
- Ring-ring interactions

### 4. π-Cation Interactions
- Aromatic-cation residue interactions
- Charge-pi interactions

## 🔍 Example Output

The pipeline generates professional network visualizations showing:

- **Chain Separation**: Clear left/right layout for different protein chains
- **Interaction Density**: Visual representation of interaction frequency
- **Occupancy Data**: Percentage values indicating interaction persistence
- **Residue Information**: Chain, amino acid type, and residue number
- **Legend**: Comprehensive interaction type legend

## 🐛 Troubleshooting

### Common Issues

1. **Schrödinger not found**:
   ```bash
   export SCHRODINGER=/path/to/schrodinger
   source $SCHRODINGER/schrodinger.ve/bin/activate
   ```

2. **Missing trajectory files**:
   - Ensure trajectory files are in the correct format
   - Check file permissions and paths

3. **No interactions detected**:
   - Lower the occupancy threshold
   - Check chain group specifications
   - Verify trajectory quality

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
python ppi_analysis.py --debug [other options]
```

## 📈 Performance

- **Scalable**: Handles large trajectories efficiently
- **Memory Optimized**: Processes frames incrementally
- **Parallel Processing**: Utilizes multiple CPU cores when available
- **Configurable**: Adjustable frame limits and thresholds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Schrödinger Inc.**: For providing the Desmond molecular dynamics software
- **NetworkX**: For network analysis and visualization capabilities
- **Matplotlib**: For high-quality plotting and visualization

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/donniv86/protein-protein-interaction-analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/donniv86/protein-protein-interaction-analysis/discussions)

---

**Made with ❤️ for the computational biology community**