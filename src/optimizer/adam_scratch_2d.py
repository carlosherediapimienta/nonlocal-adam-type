from typing import Callable, Tuple
import numpy as np

class AdamScratch2D:
    """
    Scalar Adam optimizer from scratch.

    Parameters
    ----------
    dL : Callable
        Gradient function dL(theta). 
    lr : float
        Base learning rate (alpha).
    beta1 : float
        Exponential decay rate for the first moment (0 < beta1 < 1).
    beta2 : float
        Exponential decay rate for the second moment (0 < beta2 < 1).
    epsilon : float
        Small constant in the denominator for numerical stability.
    epochs : int
        Number of iterations to run.

    Attributes
    ----------
    m, v : float
        First/second moment accumulators.
    iteration : int
        1-based iteration counter.
    theta_result, m_result, v_result : list[float]
        Per-epoch histories for analysis/plotting.
    """

    def __init__(self, dL: Callable, lr: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, epochs: int = 1000):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.dL = dL
        self.epochs = epochs
        self.global_error_tolerance = 1e-5  # placeholder; not used to stop
        self.__reset_state__()

    # Private method to reset the optimizer state and histories
    def __reset_state__(self):
        """Reset optimizer state and histories."""
        self.m = np.array([0.0, 0.0])
        self.v = np.array([0.0, 0.0])
        self.iteration = 1
        self.theta_result = []
        self.m_result = []
        self.v_result = []

    @staticmethod
    def __global_error__(theta_new: np.ndarray, theta_old: np.ndarray) -> float:
        """Absolute difference |theta_new - theta_old|; handy as a progress metric."""
        diff = theta_new - theta_old
        return np.linalg.norm(diff)

    # Public method to solve the optimization problem
    def solve(self, theta_initial: np.ndarray, function_parameters: Tuple[float, float]) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int]:
            """
            Run Adam for `epochs` steps starting from `theta_initial`.

            Parameters
            ----------
            theta_initial : list, tuple, or numpy.ndarray
                Initial values [theta1, theta2]
            function_parameters : Tuple[float, float]
                Parameters of the function to optimize.
            Notes
            -----
            - 2D implementation (always works with [theta1, theta2]).
            - Bias-corrected moments m, v are used.
            - If `weight_decay != 0`, apply decoupled weight decay before subtracting `update`.
            """
            self.__reset_state__()
            theta = np.array([float(theta_initial[0]), float(theta_initial[1])])

            while self.iteration <= self.epochs:
                # Log histories 
                self.theta_result.append([theta[0], theta[1]])
                self.m_result.append([self.m[0], self.m[1]])
                self.v_result.append([self.v[0], self.v[1]])

                theta_old = theta.copy()
                dL_value = np.array(self.dL(function_parameters, theta))

                # Adam moment updates
                self.m = self.beta1 * self.m + (1 - self.beta1) * dL_value
                self.v = self.beta2 * self.v + (1 - self.beta2) * (dL_value ** 2)

                # Bias corrections
                m_hat = self.m / (1 - self.beta1 ** self.iteration)
                v_hat = self.v / (1 - self.beta2 ** self.iteration)

                # Adam update (theta -= update)
                theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

                global_error = float(self.__global_error__(theta_new=theta, theta_old=theta_old))
                if self.iteration % 50 == 0:
                    print(f'Epoch: {self.iteration}, Error: {global_error:.6e}')

                self.iteration += 1

            print(f'Last epoch: {self.iteration-1}, Error: {global_error:.6e}')

            return self.theta_result, self.m_result, self.v_result, self.iteration