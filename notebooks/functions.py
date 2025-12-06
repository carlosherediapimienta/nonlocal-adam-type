import numpy as np
from typing import Tuple

## Functions for the analysis

def dL_quadratic(function_parameters: Tuple[float, float], theta: np.ndarray) -> np.ndarray:
    """
    Gradient of the quadratic function: L(theta) = lambda1 * theta[0]**2 + lambda2 * theta[1]**2.
    """
    return np.array([function_parameters[0] * theta[0], function_parameters[1] * theta[1]])