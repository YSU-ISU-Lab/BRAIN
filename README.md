This repository contains the official implementation for our paper:

**BRAINHash: Brain-inspired Region-Aligned Interaction Network for Unsupervised Cross-Modal Hashing**

## 🧠 Overview

This codebase includes two main components:

- **HIPPO**: The hippocampus-inspired teacher model (🚧 *will be available soon*).
- **BRAIN**: The final SNN-based framework for unsupervised cross-modal hashing.

## 📄 Paper

Coming soon...

## 📁 Datasets

Please refer to the datasets used in [UCCH TPAMI'23](https://github.com/penghu-cs/UCCH/tree/main) for reproduction.

- MS COCO
- NUS-WIDE
- MIRFlickr25K
- IAPR TC-12

## 🚀 Usage

- Running BRAIN

  To run the BRAIN model, use the following command:
  
  ```bash
  python BRAIN.py --data_name flickr --bit 16 --num_hiden_layers 3 2 4 2 2 1 --max_epochs 300 --train_batch_size 256 --log_name BRAIN --lr 0.0001 --resume vgg11_16_best_flickr_checkpoint.t7 --margin1 0.5 --margin2 0.5 --margin3 0.5 --time_enc1 0.06 --time_enc2 0.09

