"""lctx-bench: a rigorous long-context evaluation harness.

Measures how model accuracy on synthetic recall and reasoning tasks degrades as
context length and information position vary, with confidence intervals,
explicit control conditions, and pluggable model backends.
"""

__version__ = "0.1.0"

from . import scoring, tasks  # noqa: F401  (importing registers tasks and scorers)
from .config import ExperimentConfig  # noqa: F401

__all__ = ["ExperimentConfig", "__version__"]
