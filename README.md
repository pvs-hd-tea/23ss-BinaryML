# 23ss-BinaryML
Practical in summer 2023: BinaryML

# Malware & Vulnerability Detection Model
This repository contains a machine learning transformer model for malware and vulnerability detection. The model is based on the BERT architecture. 
Working on malware detection model as of now, and is trained on a subset of the EMBER dataset. Vulnerability detection is yet to be implemented.

## Dataset

The EMBER dataset is a collection of features extracted from Windows portable executable (PE) files. The dataset includes various numerical features and sequences of API calls extracted from the binary files. The dataset is split into training, validation, and testing sets.

## Requirements

To run the code in this repository, you need the following dependencies:

-Python (version >= 3.6)
-PyTorch (version >= 1.7.0)
-Transformers library (version >= 4.4.2)
-scikit-learn library (version >= 0.24.1)
-numpy library (version >= 1.19.5)
-jsonlines library (version >= 2.0.0)

## Getting Started
1. Clone the repository to your local machine: git clone https://github.com/pvs-hd-tea/23ss-BinaryML.git
2. Install the required dependencies: pip install -r requirements.txt
3. Download the EMBER dataset from the following link and extract it to the data/ember2018 directory: https://github.com/elastic/ember
4. Run the malware_detection.py script to train and evaluate the model: python malware_detection.py
Note: You may need to modify the data_path variable in the script to point to the correct location of the EMBER dataset on your machine.

## Model Architecture
The malware detection model is based on the BERT architecture, which is a transformer-based model that has been pre-trained on a large corpus of text data. In this project, BERT is used to learn representations from the sequential features of malware samples.

The model consists of the following components:

BERT encoder: This component takes the input sequences and encodes them into dense representations. It uses the pre-trained BERT model as a feature extractor.

Linear layer: This component takes the encoded features and applies a linear transformation to reduce the dimensionality of the features.

Dropout layer: This component helps to regularize the model and prevent overfitting.

Classifier layer: This component takes the reduced-dimensional features and performs the final classification into the two classes: malicious and benign.

## Training and Evaluation
The model is trained using the EMBER dataset, which contains a large number of malware samples along with their labels. The dataset is split into training, validation, and testing sets. The model is trained on the training set, and the performance is evaluated on the validation set. The testing set is used to assess the final performance of the model.

During training, the model is optimized using the Adam optimizer and the cross-entropy loss function. The number of training epochs and other hyperparameters can be adjusted according to your requirements.

After training, the model is evaluated on the testing set, and metrics such as accuracy and loss are calculated to assess the performance of the model.
