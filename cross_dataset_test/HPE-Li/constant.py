#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:16:35 2024

@author: jackson-devworks
"""

experiment_config = {
    "mmfi_config": "/home/aip/Downloads/LKZ/wiflow_mmfi/config.yaml",
    "dataset_root": "/home/aip/DATA/MMFi_Clean",
    "noise_level": [0.0],
    "mode": 0,  # Mode 0: no denoiser layer, Mode 1: have AE denoiser layers, Mode 2: use traditional filter to denoise
    "epoch": 60,
    "checkpoint": "/home/aip/Downloads/LKZ/wiflow_mmfi/HPE-Li-ECCV2024-main/output",
}

denoiser_config = {
    "epoch": 20,
    "mode": 1,  # Mode 0: 1 stage AE, Mode 1: stacked AE
    "previous_encoder": "/home/aip/Downloads/LKZ/wiflow_mmfi/HPE-Li-ECCV2024-main/output/SPN",
    "checkpoint": "/home/aip/Downloads/LKZ/wiflow_mmfi/HPE-Li-ECCV2024-main/output/AWGN",
}
