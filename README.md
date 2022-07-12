# CRA-Net: A Channel Recalibration Feature Pyramid Network for Detecting Small Pests ([paper](https://doi.org/10.1016/j.compag.2021.106518))

## Introduction

This is an unofficial implementation of CRA-Net, a novel deep learning framework designed to address challenges in detecting small agricultural pests.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/santiagot122/CRA-Net.git
   cd CRA-Net
   ```

2. Create a Python virtual environment and install the required packages:

   ```bash
   python3 -m venv cranet_env
   source cranet_env/bin/activate
   ```

3. Install the required packages:

   ```bash
    pip install -r requirements.txt
   ```

## Dataset

The model uses the IP102 dataset, which contains 102 classes of agricultural pests.

## Training

1. Preprocess the dataset:

   ```bash
   python preprocess.py --dataset_path /path/to/IP102 --output_path /path/to/preprocessed_data
   ```

2. Start the training process:

   ```bash
   python trainval_net.py --data_path /path/to/preprocessed_data --batch_size 16 --epochs 100
   ```

## Testing

Run the evaluation on the test set:

```bash
python test_net.py --data_path /path/to/preprocessed_data --model_path /path/to/saved_model
```

## Results

| Model                                                    | mAP (%)  | Precision (%) | Recall (%) |
| -------------------------------------------------------- | -------- | ------------- | ---------- |
| [LMPD2020](https://doi.org/10.1016/j.compag.2021.106518) | 49.2     | 38.7          | 56.9       |
| **IP102**                                                | **85.2** | **82.1**      | **83.3**   |

## Hardware configuration

- **GPU**: NVIDIA RTX 3080 Ti (12 GB VRAM)
- **RAM**: 32 GB
- **OS**: Ubuntu 20.04 LTS
