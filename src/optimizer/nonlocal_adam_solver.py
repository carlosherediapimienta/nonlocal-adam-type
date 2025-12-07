import numpy as np
from typing import Callable, Tuple
from scipy.interpolate import interp1d as scipy_interp1d
from .tools.algorithm import AlgorithmIDE, DTYPE
from .tools.integration import IntegrationQuadrature
from .tools.kernels_definition import K_beta_first_order, K_beta_second_order

class NonlocalSolverMomentumAdam:
    """
    Continuous-time Adam-like nonlocal dynamics for a scalar parameter θ(t):

        dot theta(t) = - alpha(t) hatm(t) / ( sqrt{hatv(t) + eps(t)} )

    with hatm, hatv defined by exponential kernels K_1, K_2
           K_a(t) = (1 - beta_a)/alpha exp(-(1 - β_a) t / alpha),   a in {1,2}.

    Notes
    -----
    - Uses AlgorithmIDE for grid construction, Euler stepping, relaxation.
    - Uses IntegrationQuadrature for Gauss-Legendre quadrature.
    - alpha(t) and eps(t) are bias-correction factors (analogs to discrete Adam).
    """
    def __init__(self, 
                 dL: Callable,
                 t_span: Tuple[float, float],
                 y0: float,
                 alpha: float,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps_base: float = 1e-8,
                 verbose: bool = False,
                 quad_order: int = 1000):
        
        self.dL = dL
        self.beta1, self.beta2 = map(DTYPE, betas)
        self.eps_base = DTYPE(eps_base)
        self.verbose = verbose
        
        y0_arr = np.asarray(y0, dtype=DTYPE)
        if y0_arr.ndim == 0 or len(y0_arr) == 1:
            # First order equation
            self.equation_order = 1
            self.y0 = y0_arr.ravel()[0]
        elif len(y0_arr) == 2:
            # Second order equation
            self.equation_order = 2
            self.y0 = y0_arr
        else:
            raise ValueError(f"y0 debe ser escalar o vector de 2 elementos, got shape {y0_arr.shape}")
        
        self.t0, self.tf = t_span
        self.alpha = DTYPE(alpha)
        self.t = np.arange(self.t0, self.tf, self.alpha, dtype=DTYPE)
        
        # Setup integrator for quadrature computations
        self.integrator = IntegrationQuadrature(n=quad_order, tol=1e-12, verbose=verbose)

        if self.equation_order == 1:
            self.K1 = lambda s: K_beta_first_order(s, self.beta1, self.alpha)
            self.K2 = lambda s: K_beta_first_order(s, self.beta2, self.alpha)
            if verbose:
                print(f"First order equation --> using exponential kernels")
        else:
            self.K1 = lambda s: K_beta_second_order(s, self.beta1, self.alpha)
            self.K2 = lambda s: K_beta_second_order(s, self.beta2, self.alpha)
            if verbose:
                print(f"Second order equation --> using sinh/sin kernels")
        
        
        # Bias-correction factors (continuous-time analogs)
        self._alpha_t = lambda t: np.where(
            t <= 1e-12,
            1.,
            np.sqrt(1. - self.beta2 ** (t / self.alpha)) / (1. - self.beta1 ** (t / self.alpha))
        )
        self._eps_t = lambda t: np.where(
            t <= 1e-12,
            self.eps_base,                                   
            self.eps_base * np.sqrt(1. - self.beta2 ** (t / self.alpha))
        )

        # Lambda parameters for kernels
        self.lam1 = (1. - self.beta1) / self.alpha
        self.lam2 = (1. - self.beta2) / self.alpha

        # Precompute weight matrices (for potential fast-matrix versions)
        dt = self.t[:, None] - self.t[None, :]
        self._tri = dt >= 0
        self._exp1 = np.exp(-self.lam1 * dt)
        self._exp2 = np.exp(-self.lam2 * dt)
        
        # Create the IDE solver with our specific RHS and stats builder
        self.solver = AlgorithmIDE(
            dL=self.dL,
            rhs=self._rhs,
            build_stats=self._build_stats,
            t_span=t_span,
            y0=y0,
            alpha=alpha,
            verbose=verbose,
            quad_order=quad_order
        )
        
    def _interp(self, y: np.ndarray):
        """Cubic interpolator over the current time grid."""
        return lambda t: scipy_interp1d(self.t, y, kind='cubic', 
                                        fill_value='extrapolate')(t)

    def _build_stats(self, y: np.ndarray) -> Tuple:
        """
        Precompute along the current iterate y(t):
          - m(t): exponential moving average of g(t)
          - v(t): exponential moving average of g(t)^2
          - sqrt(v(t)), alpha(t), eps(t): quantities needed by the RHS

        Returns
        -------
        (y, m, v_sqrt, a_t, eps_t)
        """
        if self.verbose:
            print(f"¿NaNs in y?: {np.isnan(y).any()}")

        interp = self._interp(y)

        def g_fun(tau):
            # g(tau) = dL(y(tau))
            return self.dL(interp(tau)) 
        
        def _moments_single(t, lam):
            """
            Compute (m(t), v(t)) for a single time t via GL quadrature.
            For very small t, short-circuit to (0, 0) to avoid boundary issues.
            """
            if t < 1e-12:
                return DTYPE(0.), DTYPE(0.)
            
            f_m = lambda tau: self.K1(t - tau) * g_fun(tau)
            f_v = lambda tau: self.K2(t - tau) * g_fun(tau)**2
            
            # Use the integrator instance
            m_k = self.integrator.integrate(f_m, 1e-12, t)
            v_k = self.integrator.integrate(f_v, 1e-12, t)
            
            if self.verbose:
                print(f"step values --> t={t}  m={m_k}  v={v_k}")
            return m_k, v_k

        # Vectorized evaluation over the solver grid
        moments = np.array([_moments_single(t) for t in self.t])
        m = moments[:, 0]
        v = moments[:, 1]

        if self.verbose:
            print(f"m[0]={m[0]:.3e}, v[0]={v[0]:.3e}")
            print(f"m[1]={m[1]:.3e}, v[1]={v[1]:.3e}")
            print(f"m[-1]={m[-1]:.3e}, v[-1]={v[-1]:.3e}")

        # Save timeseries for inspection/plotting
        self._last_m = np.stack((self.t, m), axis=1)
        self._last_v = np.stack((self.t, v), axis=1)

        v_sqrt = np.sqrt(v)
        a_t = np.array([self._alpha_t(t) for t in self.t])
        eps_t = np.array([self._eps_t(t) for t in self.t])

        return (y, m, v_sqrt, a_t, eps_t)

    def _rhs(self, t: float, y_prev: float, idx: int, 
             y_fix: np.ndarray, m: np.ndarray, v_sqrt: np.ndarray, 
             a_t: np.ndarray, eps_t: np.ndarray) -> float:
        """
        Right-hand side for the explicit Euler step:

            dot theta(t) = f(t, theta(t)) - alpha(t) m(t) / ( sqrt{v(t)} + eps(t) )

        Parameters
        ----------
        t : float
            Current time.
        y_prev : float
            Previous value.
        idx : int
            Index of `t` on the solver time grid.
        y_fix : np.ndarray
            Current iterate y(t).
        m : np.ndarray
            Samples of the first moment along the grid.
        v_sqrt : np.ndarray
            Samples of sqrt(second moment) along the grid.
        a_t : np.ndarray
            Bias-correction factor alpha(t) along the grid.
        eps_t : np.ndarray
            Scaled eps(t) along the grid.

        Returns
        -------
        float
            dy/dt - Instantaneous rate used by the Euler integrator.
        """
        # Interpolate y at the exact time t
        y_val = scipy_interp1d(self.t, y_fix, kind='cubic', 
                               fill_value='extrapolate')(t)

        # Denominator sqrt{v(t)} + eps(t) for stability
        denom = v_sqrt[idx] + eps_t[idx]

        # Combine local dynamics with normalized moment term
        return - a_t[idx] * (m[idx] / denom)
    
    def solve(self):
        """
        Solve the nonlocal Adam ODE using the IDE solver.
        
        Returns
        -------
        t : np.ndarray
            Time grid.
        y : np.ndarray
            Solution values at each time point.
        """
        return self.solver.solve()