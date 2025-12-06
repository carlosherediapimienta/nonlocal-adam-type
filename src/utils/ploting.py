import matplotlib.pyplot as plt
import numpy as np
from typing import List


class AdamPlotter:
    """
    Clase sencilla para visualizar la evolución de theta, m y v durante la optimización con Adam.
    
    Parameters
    ----------
    theta_history : List[List[float]]
        Historial de valores de theta [theta1, theta2] por iteración.
    m_history : List[List[float]]
        Historial de valores del primer momento m [m1, m2] por iteración.
    v_history : List[List[float]]
        Historial de valores del segundo momento v [v1, v2] por iteración.
    """
    
    def __init__(self, theta_history: List[List[float]], 
                 m_history: List[List[float]], 
                 v_history: List[List[float]]):
        self.theta_history = np.array(theta_history)
        self.m_history = np.array(m_history)
        self.v_history = np.array(v_history)
        self.iterations = np.arange(len(theta_history))
    
    def __plot_theta__(self, figsize=(10, 6)):
        """Plotea la evolución de theta1 y theta2."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.iterations, self.theta_history[:, 0], label='θ₁', linewidth=2)
        ax.plot(self.iterations, self.theta_history[:, 1], label='θ₂', linewidth=2)
        
        ax.set_xlabel('Iteración', fontsize=12)
        ax.set_ylabel('Valor de θ', fontsize=12)
        ax.set_title('Evolución de θ durante la optimización', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def __plot_moments__(self, figsize=(12, 5)):
        """Plotea la evolución de los momentos m y v."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Primer momento m
        ax1.plot(self.iterations, self.m_history[:, 0], label='m₁', linewidth=2)
        ax1.plot(self.iterations, self.m_history[:, 1], label='m₂', linewidth=2)
        ax1.set_xlabel('Iteración', fontsize=11)
        ax1.set_ylabel('Valor de m', fontsize=11)
        ax1.set_title('Evolución del primer momento m', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Segundo momento v
        ax2.plot(self.iterations, self.v_history[:, 0], label='v₁', linewidth=2)
        ax2.plot(self.iterations, self.v_history[:, 1], label='v₂', linewidth=2)
        ax2.set_xlabel('Iteración', fontsize=11)
        ax2.set_ylabel('Valor de v', fontsize=11)
        ax2.set_title('Evolución del segundo momento v', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def __plot_all__(self, figsize=(15, 10)):
        """Plotea todas las variables en una sola figura."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Theta
        axes[0, 0].plot(self.iterations, self.theta_history[:, 0], label='θ₁', linewidth=2)
        axes[0, 0].plot(self.iterations, self.theta_history[:, 1], label='θ₂', linewidth=2)
        axes[0, 0].set_xlabel('Iteración', fontsize=11)
        axes[0, 0].set_ylabel('Valor de θ', fontsize=11)
        axes[0, 0].set_title('Evolución de θ', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Primer momento m
        axes[0, 1].plot(self.iterations, self.m_history[:, 0], label='m₁', linewidth=2)
        axes[0, 1].plot(self.iterations, self.m_history[:, 1], label='m₂', linewidth=2)
        axes[0, 1].set_xlabel('Iteración', fontsize=11)
        axes[0, 1].set_ylabel('Valor de m', fontsize=11)
        axes[0, 1].set_title('Evolución del primer momento m', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Segundo momento v
        axes[1, 0].plot(self.iterations, self.v_history[:, 0], label='v₁', linewidth=2)
        axes[1, 0].plot(self.iterations, self.v_history[:, 1], label='v₂', linewidth=2)
        axes[1, 0].set_xlabel('Iteración', fontsize=11)
        axes[1, 0].set_ylabel('Valor de v', fontsize=11)
        axes[1, 0].set_title('Evolución del segundo momento v', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Trayectoria en el espacio 2D (theta1 vs theta2)
        axes[1, 1].plot(self.theta_history[:, 0], self.theta_history[:, 1], 
                       linewidth=2, alpha=0.7)
        axes[1, 1].scatter(self.theta_history[0, 0], self.theta_history[0, 1], 
                          color='green', s=100, label='Inicio', zorder=5)
        axes[1, 1].scatter(self.theta_history[-1, 0], self.theta_history[-1, 1], 
                          color='red', s=100, label='Final', zorder=5)
        axes[1, 1].set_xlabel('θ₁', fontsize=11)
        axes[1, 1].set_ylabel('θ₂', fontsize=11)
        axes[1, 1].set_title('Trayectoria en el espacio θ', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    # Public method to plot the results
    def plot(self, which='all', figsize=None):
        """
        Método principal para plotear.
        
        Parameters
        ----------
        which : str
            Qué plotear: 'theta', 'moments', o 'all' (default: 'all')
        figsize : tuple, optional
            Tamaño de la figura. Si None, usa tamaños por defecto.
        """
        if figsize is None:
            figsize_map = {
                'theta': (10, 6),
                'moments': (12, 5),
                'all': (15, 10)
            }
            figsize = figsize_map.get(which, (10, 6))
        
        if which == 'theta':
            return self.__plot_theta__(figsize)
        elif which == 'moments':
            return self.__plot_moments__(figsize)
        elif which == 'all':
            return self.__plot_all__(figsize)
        else:
            raise ValueError(f"Opción '{which}' no válida. Usa 'theta', 'moments', o 'all'")
