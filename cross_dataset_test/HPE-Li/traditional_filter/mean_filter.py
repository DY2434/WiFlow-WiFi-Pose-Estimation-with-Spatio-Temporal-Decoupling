#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:49:32 2024

@author: jackson-devworks
"""

import numpy as np


def mean_filter(data, kernel_size=3):
    """
    Apply a mean filter to 4D WiFi CSI data (batch_size, num_channels, num_subcarriers, num_timesteps).

    Args:
        data (numpy array): Input 4D WiFi CSI data.
        kernel_size (int): Size of the mean filter kernel.

    Returns:
        numpy array: Denoised data after applying the mean filter.
    """
    batch_size, num_channels, num_subcarriers, num_timesteps = data.shape

    # Calculate the padding size
    pad_size = kernel_size // 2

    # Pad the data along the time axis (axis 3) while keeping the other dimensions intact
    padded_data = np.pad(data, ((0, 0), (0, 0), (0, 0), (pad_size, pad_size)), mode='edge')

    # Initialize the output data
    filtered_data = np.zeros_like(data)

    # Apply the mean filter across the time axis (axis 3)
    for i in range(num_timesteps):
        filtered_data[:, :, :, i] = np.mean(padded_data[:, :, :, i:i + kernel_size], axis=3)

    return filtered_data