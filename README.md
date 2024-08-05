# A multi-modal fusion model with enhanced feature representation for chronic kidney disease progression prediction 
 <img src="https://github.com/Qiaoyx97/FLEX/blob/main/pipeline.png" width="600" />

## Description
In this work we present FLEX，a multi-modal fusion model that integrates clinical data, proteomic data, metabolomic data, and pathology images across different scales and modalities, with a focus on advanced feature learning and representation. FLEX contains a Feature Encoding Trainer structure that can train feature encoding, thus achieving fusion of inter-feature and inter-modal. 

© This code is made available for non-commercial academic purposes.

## Clone the Repository
```
https://github.com/Qiaoyx97/FLEX.git
```

## Installation
The environment settings needed to run the code:
- Linux (Tested on Ubuntu 18.04.6)
- NVIDIA GPU (Tested on Nvidia A100)
- CUDA (11.3)
- Package
    - Python (3.6.9)
    - PyTorch (1.10.1+cu113)
    - TorchMetrics (0.8.2)
    - torchvision (0.11.2+cu113)
    - NumPy (1.19.0)
    - pandas (1.1.5)
    - scikit-learn (0.24.2)
    - tqdm (4.64.1)
    - Albumentations (1.3.0)
    - Imbalanced-learn (0.8.1)
    - opencv-python-headless (4.7.0.72)
    - openslide-python (1.1.2)
 
 ## Data Preparation
We have provided a test sample in [sample](https://github.com/Qiaoyx97/FLEX/tree/main/sample). image is a 1024×1024 `.npy` file. Clinical data, proteomics data and metabolomics data, are stored as `.csv` files. Folder structure for data storage:
```
sample/
    ├── images/
         └── image_files
                ├── slide_1.npy
                ├── slide_2.npy
                └── ...
     ├── clinical_file.csv
     ├── proteomics_file.csv
     └── metabolomics_file.csv
```
Details of preprocessing methods for image data are given in [preprocess].

