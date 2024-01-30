import numpy as np

from typing import NamedTuple


class SNGPInfo(NamedTuple):
    decision: np.ndarray
    mean: np.ndarray
    deviation: np.ndarray
    variance: np.ndarray
