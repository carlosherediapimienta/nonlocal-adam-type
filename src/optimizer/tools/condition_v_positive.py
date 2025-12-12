import numpy as np
from typing import Callable

class ConditionVPositive:
    def __init__(self,  alpha: float):
        self.alpha = alpha       

    def __call__(self, beta2: float, r: np.ndarray, t: np.ndarray, v: np.ndarray) -> bool:
        viol = np.zeros_like(t, dtype=bool)
        if beta2 >= 0.5:
            return viol
        else:
            rho = np.sqrt(1.0 - 2.0*beta2)          
            q   = np.exp(-np.pi / rho)
            delta = np.pi * self.alpha / rho              
            W = int(np.ceil(delta / self.alpha))          
            for k in range(len(t)):
                lo = max(0, k - W)
                r_win = r[lo:k+1]
                rmin, rmax = r_win.min(), r_win.max()
                viol[k] = (rmin < q * rmax)
            return viol



