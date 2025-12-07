from typing import Callable
import numpy as np

class IntegrationQuadrature:
    """
    Integration quadrature class using Gauss-Legendre quadrature.
    
    Parameters
    ----------
    n : int
        Quadrature order (number of nodes/weights).
    tol : float, optional
        Threshold below which intervals are treated as zero-length. Default: 1e-12
    verbose : bool, optional
        If True, prints intermediate results. Default: False
    """
    def __init__(self, n: int, tol: float = 1e-12, verbose: bool = False):
        self.n = n
        self.tol = tol
        self.verbose = verbose
        self.xg, self.wg = np.polynomial.legendre.leggauss(n)
    
    def integrate(self, fun: Callable, lo: float, hi: float) -> float:
        """
        Integrate function over a single interval [lo, hi].
        
        Parameters
        ----------
        fun : Callable
            Function to integrate.
        lo : float
            Lower bound.
        hi : float
            Upper bound.
            
        Returns
        -------
        float
            Integral value.
        """
        # Ensure correct orientation and keep track of sign if bounds reversed
        sign = np.where(hi < lo, -1., 1.)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

        # Affine map from [-1, 1] to [lo, hi]
        mid = 0.5 * (hi + lo)
        half = 0.5 * (hi - lo)
        pts = mid + half * self.xg

        # Evaluate integrand at quadrature nodes
        vals = np.array([fun(pt) for pt in pts])

        # Weighted sum with Jacobian factor
        res = sign * half * np.sum(self.wg * vals, axis=0)
        
        if self.verbose:
            print(f"fixed_quad: a={lo} b={hi}  res={res}")

        # Treat near-zero-length intervals as zero to avoid spurious noise
        return np.where(np.abs(hi - lo) < self.tol, 0., res)
    