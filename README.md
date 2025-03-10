# Designing Molecules with Desired Properties

This repository contains the code, data preprocessing pipeline, and report for the project "Designing Molecules with Desired Properties". The project focuses on enhancing the prediction of molecular polarizability using deep learning techniques. Specifically, it addresses the limitation of the QM9 dataset—which reports polarizability as a scalar—by converting it into a full 3×3 tensor. The modified SchNet model is then trained to predict this tensorial polarizability, enabling more detailed analyses of anisotropic molecular responses.

## Project Overview

The project is built around two key datasets:

QM9: A benchmark dataset providing a wide range of molecular properties (e.g., rotational constants, dipole moments, HOMO/LUMO energies, isotropic_polarizability, etc.), where polarizability is originally given as a scalar.
QM7-X: A dataset that offers the full polarizability tensor for small organic molecules, capturing directional information.
Our approach converts the QM9 scalar polarizability into a 3×3 tensor using an isotropic diagonal matrix and adapts the SchNet model (implemented in SchNetPack) to predict tensorial polarizability. We validate our model using parity plots and difference plots, and we also include a user interface to query the dataset for the top five molecules that best match a user-specified polarizability value.

# Getting Started

## Requirements

- Python 3.8+
- PyTorch
- SchNetPack 2.1.1
- ASE 3.23
- Other dependencies (listed in requirements.txt)

# The notebook

The main Jupyter Notebook demonstrates:

- Data Preprocessing: Converting QM9's scalar polarizability to a 3×3 tensor.
- Model Training: Training the modified SchNet model with a custom Frobenius norm-based loss function.
- Evaluation: Generating parity plots and difference plots to validate model performance.
- Molecule Screening: Querying the model for molecules that match a target polarizability.
To run the notebook, simply open it in Jupyter and execute the cells sequentially.


# Citation

Here are the main references I referenced for the project

1. Ramakrishnan, R., Dral, P. O., Rupp, M. & von Lilienfeld, O. A. “Quantum chemistry structures and properties of 134 kilo molecules.” Sci. Data 1, 140022 (2014).

2. Hoja, J. et al. “QM7-X: A comprehensive dataset of quantum‐mechanical properties spanning the chemical space of small organic molecules.” (Preprint on arXiv:2006.15139v1).
