# 23ss-BinaryML
Practical in summer 2023: BinaryML

# Malware & Vulnerability Detection Model
The main goal of this project is to combine and improve best machine learning approaches for detection of malware and vulnerabilities based on binaries.

# 1. Malware Detection Model (HRR Transformers)

## 1.1 Dataset

PE Malware Machine Learning Dataset: https://practicalsecurityanalytics.com/pe-malware-machine-learning-dataset/

## 1.2 Requirements

To run the code in this repository, you need the dependencies mentioned in the requirements.txt
You can install the dependencies using the following command:

```
pip install -r requirements.txt
```

If you're unable to install using requirements.txt, you can install the key dependencies manually using below commands:

```
pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.15+cuda11.cudnn82-cp39-none-manylinux2014_x86_64.whl
pip install flax==0.6.0
pip install optax==0.1.2
```
```
pip install tensorflow-cpu==2.8.0
pip install tensorflow-datasets==4.5.2
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
Finally, the library that implements the vector symbolic architecture called Holographic Reduced Representations (HRR) which is the key concept used to develop Hrr Transformer.
```
pip install hrr --upgrade
```

## 1.3 Getting Started
1. Clone the repository to your local machine: git clone https://github.com/pvs-hd-tea/23ss-BinaryML.git
2. Install the required dependencies.
3. Download the datasets from the mentioned links and extract it to the corresponding sub folder in the ```data``` directory (benign/malware/vulnerability).
4. You can load and visualize the data sets using ```dataset.py``` class.
4. Run the ```hrrformer_mgpu.py``` script to train and evaluate the model.
