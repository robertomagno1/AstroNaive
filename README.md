# AstroNaive

# 🌌 StellarDensity: Nonparametric Bayesian Classification of Celestial Objects
A robust implementation of nonparametric Naive Bayes with UMAP visualization for classifying stars, quasars, and white dwarfs from photometric data.

![UMAP Visualization Example](docs/umap_visualization.png)  
*(Example UMAP projection of celestial object features)*

## 📖 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features
- **Nonparametric Density Estimation**: Kernel density estimation with adaptive bandwidths
- **Dimensionality Visualization**: UMAP projections for feature space exploration
- **Independence Validation**: Mutual information and correlation analysis
- **Multi-class Classification**: Handles stars (1), quasars (2), and white dwarfs (3)
- **Production-ready**: Includes unit tests and CI/CD integration

## 💻 Installation
```bash
# Clone repository
git clone https://github.com/yourusername/stellardensity.git
cd stellardensity

# Install R dependencies
install.packages(c("umap", "ks", "infotheo", "ggplot2", "caret"))  
