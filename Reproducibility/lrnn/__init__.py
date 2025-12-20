"""
LRNN package initialization.

This package provides the core implementation of the
Logic-Rule Neural Network (LRNN), including model
definition, rule-based layers, and training utilities.
"""

from .lnn_layer import LNNLayer
from .model import LRNNModel

__all__ = [
    "LNNLayer",
    "LRNNModel",
]
