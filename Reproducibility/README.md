
# LRNN: Logic Rules Neural Network for Software Defect Prediction

This repository provides the implementation and supplementary materials for the paper:

"LRNN: A Knowledge-Based Neural Network for Tackling Both Within-Project and
Cross-Project Class-Imbalance Problems in Software Defect Prediction"

## Contents
- Source code of the LRNN model
- Scripts for rule learning and experiments
- Complete set of learned logic rules used for feature selection
- Instructions for reproducing the experiments

## Reproducibility
The logic rules used in the experiments are automatically learned from the
training data using RuleKit. The complete rule sets for all datasets are
provided in the `rules/` directory.

## Repository Structure
- `lrnn/`: core LRNN implementation
- `scripts/`: experiment and rule generation scripts
- `rules/`: learned logic rules for each dataset
- `data/`: dataset description
- `results/`: experimental outputs

## Requirements
See `requirements.txt` for dependencies.
