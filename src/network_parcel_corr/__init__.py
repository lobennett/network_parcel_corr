"""Network parcel correlation analysis package."""

from . import atlases
from . import core
from . import data
from . import io
from . import postprocessing
from .main import run_analysis

__version__ = '0.1.0'

__all__ = [
    'atlases',
    'core', 
    'data',
    'io',
    'postprocessing',
    'run_analysis',
]


def main() -> None:
    print('Hello from network_parcel_corr!')
