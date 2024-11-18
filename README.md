# ESA-Net: An efficient scale-aware network for small crop pest detection ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423018109))

## Introduction

This is an unofficial implementation of ESA-Net, a novel deep learning framework designed to address challenges in detecting small agricultural pests.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/santiagot122/ESA-Net.git
   cd ESA-Net
   ```

2. Create a Python virtual environment and install the required packages:

   ```bash
   python3 -m venv esanet_env
   source esanet_env/bin/activate
   ```

3. Install the required packages:

   ```bash
    pip install -r requirements.txt
   ```

## Dataset

The model uses the IP102 dataset, which contains 102 classes of agricultural pests.

## Training and Testing

1. Start the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open the `esanet-notebook.ipynb` file and run the cells to train and test the model.

## Results

| Model                                                                               | mAP (%)  | Precision (%) | Recall (%) |
| ----------------------------------------------------------------------------------- | -------- | ------------- | ---------- |
| [LMCP2020](https://www.sciencedirect.com/science/article/abs/pii/S0957417423018109) | 39.5     | 68.8          | 42.2       |
| **IP102**                                                                           | **87.3** | **88.2**      | **89.7**   |
