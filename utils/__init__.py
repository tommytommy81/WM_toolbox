"""
Shared helpers for the STWM MEG pipeline.

One module per notebook family:
    utils.sensor        -> notebooks/example_usage_sensor*.ipynb
    utils.source        -> notebooks/example_usage_source*.ipynb
    utils.connectivity  -> notebooks/example_usage_connectivity.ipynb
"""

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, 'config.yaml')


def load_config(config_path=CONFIG_PATH):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
