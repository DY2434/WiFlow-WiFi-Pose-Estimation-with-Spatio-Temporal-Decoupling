import numpy as np


def gaussian_filter(data, kernel_size=3):
    """
    Apply a Gaussian filter to 4D WiFi CSI data (batch_size, num_channels, num_subcarriers, num_timesteps).

    Args:
        data (numpy array): Input 4D WiFi CSI data.
        kernel_size (int): Size of the Gaussian filter kernel (must be odd).

    Returns:
        numpy array: Denoised data after applying the Gaussian filter.
    """
    batch_size, num_channels, num_subcarriers, num_timesteps = data.shape

    # Calculate the padding size
    pad_size = kernel_size // 2

    # Compute sigma as the standard deviation of the input data
    sigma = np.std(data)

    # Create the Gaussian kernel
    x = np.linspace(-pad_size, pad_size, kernel_size)
    gauss_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gauss_kernel /= np.sum(gauss_kernel)  # Normalize the kernel

    # Pad the data along the time axis (axis 3) while keeping the other dimensions intact
    padded_data = np.pad(data, ((0, 0), (0, 0), (0, 0), (pad_size, pad_size)), mode='edge')

    # Initialize the output data
    filtered_data = np.zeros_like(data)

    # Apply the Gaussian filter across the time axis (axis 3)
    for i in range(num_timesteps):
        filtered_data[:, :, :, i] = np.sum(padded_data[:, :, :, i:i + kernel_size] * gauss_kernel, axis=3)

    return filtered_data
