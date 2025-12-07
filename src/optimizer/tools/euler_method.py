"""
Explicit Euler methods for ODEs: y' = f(...) or y'' = f(...).
"""
from typing import Callable, Tuple, Any, Union, Literal
import numpy as np

DTYPE = np.float64

class EulerMethod:
    """
    Integrador Euler explícito para EDOs de 1º y 2º orden.
    
    - order=1: Forward Euler (primer orden)
    - order=2: Midpoint Euler (segundo orden)
    """
    
    def __init__(self, order: Literal[1, 2] = 1):
        if order not in [1, 2]:
            raise ValueError(f"order must be 1 or 2, got {order}")
        self.order = order

    def _forward(self,
                 alpha: float,
                 y0: Union[float, np.ndarray],
                 t_vec: np.ndarray,
                 rhs: Callable,
                 stats: Tuple[Any, ...]) -> np.ndarray:
        """Forward Euler: y_{n+1} = y_n + h * f(t_n, y_n)"""
        alpha = DTYPE(alpha)
        is_scalar = np.ndim(y0) == 0
        n = len(t_vec)
        
        # Inicializar
        if is_scalar:
            y = np.zeros(n, dtype=DTYPE)
            y[0] = DTYPE(y0)
        else:
            y = np.zeros((n, len(y0)), dtype=DTYPE)
            y[0] = np.asarray(y0, dtype=DTYPE)
        
        # Integrar
        for i in range(n - 1):
            dy = rhs(t_vec[i], y[i], i, *stats)
            y[i + 1] = y[i] + alpha * dy
        
        return y
    
    def solve(self,
                  alpha: float,
                  y0: Union[float, np.ndarray],
                  t_vec: np.ndarray,
                  rhs: Callable,
                  stats: Tuple[Any, ...] = ()) -> np.ndarray:
        """
        Integra y' = rhs(t, y, ...) sobre t_vec.
        
        Parameters
        ----------
        alpha : float
            Paso temporal.
        y0 : float or array
            Condición inicial (escalar para y', vector [y, y'] para y'').
        t_vec : ndarray
            Tiempos de evaluación.
        rhs : Callable
            Lado derecho: rhs(t, y, idx, *stats) -> dy/dt
        stats : tuple
            Parámetros extra para rhs.
        
        Returns
        -------
        y_hist : ndarray
            Solución en cada tiempo.
        """
        return self._forward(alpha, y0, t_vec, rhs, stats)
    

    