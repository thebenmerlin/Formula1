"""
Data module for synthetic data generation and loading.
"""

from .generator import SpaTrackGenerator, main as generate_data

__all__ = ['SpaTrackGenerator', 'generate_data']
