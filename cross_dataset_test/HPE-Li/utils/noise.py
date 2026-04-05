#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:55:19 2024

@author: jackson-devworks
"""

import numpy as np


def add_awgn(signal, noise_level):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal.

    :param signal: Input signal (numpy array).
    :param noise_level: Noise level defined as the standard deviation of the noise as a proportion of the signal's dynamic range.
    :return: Noisy signal with AWGN added.
    """
    # Calculate the standard deviation based on the signal's dynamic range
    std_dev = noise_level * (np.max(signal) - np.min(signal))

    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=std_dev, size=signal.shape)
    
    # Add noise to the original signal
    noisy_signal = signal + noise
    
    return noisy_signal

def add_salt_and_pepper_noise(signal, noise_level):
    """
    Add salt and pepper noise to a signal.

    :param signal: Input signal (numpy array).
    :param noise_level: Proportion of pixels to be corrupted with salt and pepper noise.
    :return: Noisy signal with salt and pepper noise added.
    """
    noisy_signal = np.copy(signal)

    # Calculate number of salt and pepper pixels
    num_salt = np.floor(noise_level * signal.size * 0.5).astype(int)
    num_pepper = np.floor(noise_level * signal.size * 0.5).astype(int)

    # Add salt (white) noise
    salt_coords = [np.random.randint(0, dim, num_salt) for dim in signal.shape]
    noisy_signal[tuple(salt_coords)] = 1

    # Add pepper (black) noise
    pepper_coords = [np.random.randint(0, dim, num_pepper) for dim in signal.shape]
    noisy_signal[tuple(pepper_coords)] = 0

    return noisy_signal
