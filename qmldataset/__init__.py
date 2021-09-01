"""
Top-level package
"""

__version__ = "0.1.0"


from .experiments.default_experiment import run_default_experiment
from .experiments.custom_experiment import run_custom_experiment

__all__ = [
    'run_custom_experiment',
    'run_default_experiment'
]
