"""Compatibility helper for the supplied legacy CareerMate pickle."""
import sys
import types
import numpy as np
from sklearn._loss import _loss

def install_pickle_compatibility() -> None:
    # The supplied artifact references target_column.dtype while unpickling.
    # Restore legacy serialization symbols without changing fitted parameters.
    if "target_column" not in sys.modules:
        module = types.ModuleType("target_column")
        module.dtype = np.dtype
        sys.modules["target_column"] = module
    # The supplied CareerMate artifact also references the Cython module by
    # its short name. Modern sklearn exposes it under sklearn._loss._loss.
    sys.modules.setdefault("_loss", _loss)
