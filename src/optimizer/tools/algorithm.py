from __future__ import annotations
from typing import Callable, Tuple, Any
import numpy as np

# Use 64-bit floats by default
DTYPE = np.float64

class AlgorithmIDE:
    """
    Algorithm for Integro-Differential Equations (IDE) solver.
    
    Shared boilerplate for non-local ODE-like solvers:
      - time grid construction
      - explicit Euler time-stepping
      - fixed-point relaxation with smoothing
      - global error monitoring and tolerance/iteration controls
      - integration via Gauss-Legendre quadrature

    Subclasses MUST implement:
      * _build_stats(y) -> Tuple[Any, ...]
          Precompute any structures ("stats") needed to evaluate the RHS
          at all times, given the current iterate y(t).

      * _rhs(t, y_prev, idx, *stats) -> dy/dt
          Right-hand side function to advance one Euler step.
          `idx` is the integer time-index (0-based) corresponding to `t`.

    Attributes (key ones)
    ---------------------
    dL : Callable
        Additional functional/derivative used by specific non-local models.
    t : np.ndarray
        Time grid [t0, tf) with spacing `alpha` (tf is excluded).
    y0 : DTYPE
        Initial condition (scalar).
    alpha : DTYPE
        Time step size.
    lambda_ : DTYPE
        Optional model parameter exposed to subclasses.
    smoothing : DTYPE
        Current relaxation factor s in y_next = s*y_new + (1-s)*y_cur.
    global_tol : float
        Convergence tolerance for the global error metric.
    max_iteration : int
        Safety cap on fixed-point iterations.
    verbose : bool
        If True, prints progress every 20 iterations.
    integrator : IntegrationQuadrature
        Gauss-Legendre quadrature integrator instance.
    """
    def __init__(self,
                 dL: Callable,
                 rhs: Callable,
                 build_stats: Callable,
                 t_span: Tuple[float, float],
                 y0,
                 alpha: float,
                 lambda_: float = 0.,
                 verbose: bool = False,
                 quad_order: int = 1000):

        self.dL = dL
        self.rhs = rhs
        self.build_stats = build_stats
        # Scalar initial condition (force dtype/shape)
        self.y0 = np.asarray(y0, dtype=DTYPE).ravel()[0]
        self.alpha = DTYPE(alpha)
        self.lambda_ = DTYPE(lambda_)
        self.verbose = verbose

        self.t0, self.tf = t_span
        # Time grid is [t0, tf) with uniform spacing alpha (tf not included)
        self.t = np.arange(self.t0, self.tf, self.alpha, dtype=DTYPE)

        # ---------- relaxation (under-relaxed fixed-point iteration) ----
        self.smoothing = DTYPE(0.5)    # starting smoothing factor
        self.smooth_max = DTYPE(0.9999) # hard cap to avoid overshoot
        # Monotone schedule of candidate smoothing increases
        self.increments = np.linspace(self.smoothing,
                                      self.smooth_max,
                                      1_000, dtype=DTYPE)
        self._max_inc_hit = False         # flag once the max is reached

        # ---------- stopping criteria / safeguards ----------------------
        self.global_tol = 1e-4
        self.max_iteration = int(3e3)
    
    # ---------------------------------------------------------------
    @staticmethod
    def _integrate(alpha: DTYPE,
                   y0: DTYPE,
                   t_vec: np.ndarray,
                   rhs: Callable[[DTYPE, DTYPE, int, Tuple[Any, ...]],DTYPE],
                   stats: Tuple[Any, ...]) -> np.ndarray:
        """
        Explicit Euler integrator over `t_vec`.

        Parameters
        ----------
        alpha : DTYPE
            Time step size.
        y0 : DTYPE
            Initial value at t_vec[0].
        t_vec : np.ndarray
            Monotone time grid.
        rhs : Callable
            Function (t, y_prev, idx, *stats) -> dy/dt.
        stats : tuple
            Precomputed data built from the current iterate y(t).
        Returns
        -------
        y_hist : np.ndarray
            Values of y at all points in t_vec (same length as t_vec).
        """
        y_hist = np.zeros(len(t_vec), dtype=DTYPE)
        y_hist[0] = y0
        
        for i in range(len(t_vec) - 1):
            t = t_vec[i]
            y_prev = y_hist[i]
            y_hist[i + 1] = y_prev + alpha * rhs(t, y_prev, i, *stats)
        
        return y_hist

    # ------------- error & smoothing -----------------------------------
    @staticmethod
    def _global_error(a: np.ndarray, b: np.ndarray):
        """Global error metric: Euclidean norm over the full time series."""
        return np.linalg.norm(a - b)

    @staticmethod
    def _mix(s: DTYPE, a: np.ndarray, b: np.ndarray):
        """Convex combination for relaxation: s*a + (1-s)*b."""
        return s * a + (1. - s) * b

    # -------------------------------------------------------------------
    def _step(self, y_current: np.ndarray) -> np.ndarray:
        """Run one full pass of the explicit Euler integrator."""
        stats = self.build_stats(y_current)
        y_next = self._integrate(self.alpha, self.y0, self.t, self.rhs, stats)
        err = self._global_error(y_current, y_next)
        return y_next, err

    # -------------------------------------------------------------------
    def solve(self):
        """
        Fixed-point outer loop with under-relaxation.

        Algorithm sketch
        ----------------
        1) Build a purely local baseline solution (no integral term)
           by integrating y' = f(t, y).
        2) Given the current iterate y_cur, build stats and integrate the
           full non-local RHS once to get y_new.
        3) Relax: y_relax = mix(smoothing, y_cur, y_new).
        4) Repeat until the global error ||y_relax - y_new|| < global_tol
           or safety limits are hit. The smoothing factor can increase
           when the error rises (simple backoff strategy).
        """
        self.iteration = 0

        # Baseline solution without the non-local term
        y_cur = self._integrate(self.alpha, self.y0, self.t, self.rhs)
        y_new, err = self._step(y_cur)
        if self.verbose:
            print(f"Iter {self.iteration} – err {err}")

        last = err
        while err > self.global_tol:
            # Under-relaxed update
            y_relax = self._mix(self.smoothing, y_cur, y_new)

            # Re-integrate with stats from the relaxed iterate
            y_new, err = self._step(y_relax)

            # ---------- safety / control logic ----------
            if np.isnan(err) or np.isinf(err):
                print("Divergence (NaN / Inf). Abort.")
                break
            
            # If error worsens, increase smoothing factor (toward 1)
            if err > last:
                if self._max_inc_hit:
                    print("Maximum smoothing achieved. Stop.")
                    break
                try:
                    nxt = self.increments[
                        np.searchsorted(self.increments,
                                        self.smoothing, side="right")]
                except IndexError:
                    nxt = self.smooth_max
                    self._max_inc_hit = True
                self.smoothing = np.minimum(self.smooth_max, nxt)
            last = err
            y_cur = y_relax
            self.iteration += 1

            if self.iteration % 20 == 0:
                print(f"Iter {self.iteration} – err {err}")

            if self.iteration >= self.max_iteration:
                print("Max iterations. Stop.")
                break

        print(f"Last iter {self.iteration} – err {err}")
        self.y = y_new
        self.global_error = err
        return self.t, y_new