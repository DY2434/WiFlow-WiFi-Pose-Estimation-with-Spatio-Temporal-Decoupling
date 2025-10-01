"""
WiFi-based Human Pose Estimation Package
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .config import Config
from .dataset import PreprocessedCSIDataset, create_data_loaders

__all__ = [
    'Config',
    'PreprocessedCSIDataset',
    'create_data_loaders',
]
