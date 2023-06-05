# 23ss-BinaryML
Practical in summer 2023: BinaryML

# Malware & Vulnerability Detection Model
This repository contains a machine learning transformer model for malware and vulnerability detection. The model is based on the BERT architecture. 
Working on malware detection model as of now, and is trained on a subset of the EMBER dataset. Vulnerability detection is yet to be implemented.

## Dataset

The EMBER dataset is a collection of features extracted from Windows portable executable (PE) files. The dataset includes various numerical features and sequences of API calls extracted from the binary files. The dataset is split into training, validation, and testing sets.

## Requirements

To run the code in this repository, you need the following dependencies:

- Python (>= 3.6)
- PyTorch (>= 1.7.0)
- Transformers (>= 4.0.0)
- NumPy (>= 1.19.0)
- scikit-learn (>= 0.23.0)
