from typing import Optional
import numpy as np

class FourierMMDFitting:
    """Store precomputed statistics for Fourier MMD drift detection."""
    
    def __init__(self, random_seed: Optional[int] = None, 
                 delta_stat: Optional[bool] = None, 
                 n_mode: Optional[int] = None):
        """
        Initialize with parameters for Fourier MMD.
        
        Args:
            random_seed: Seed for random number generation
            delta_stat: Whether to compute delta statistics
            n_mode: Number of Fourier modes
        """
        self._random_seed = random_seed  
        self._delta_stat = delta_stat    
        self._n_mode = n_mode           
        self._scale: Optional[np.ndarray] = None  
        self._a_ref: Optional[np.ndarray] = None 
        self._mean_mmd: Optional[float] = None   
        self._std_mmd: Optional[float] = None    
    
    def get_random_seed(self) -> Optional[int]:
        """Get the random seed."""
        return self._random_seed
    
    def set_random_seed(self, random_seed: int) -> None:
        """Set the random seed."""
        self._random_seed = random_seed
    
    def is_delta_stat(self) -> Optional[bool]:
        """Get the delta stat flag."""
        return self._delta_stat
    
    def set_delta_stat(self, delta_stat: bool) -> None:
        """Set the delta stat flag."""
        self._delta_stat = delta_stat
    
    def get_n_mode(self) -> Optional[int]:
        """Get the number of modes."""
        return self._n_mode
    
    def set_n_mode(self, n_mode: int) -> None:
        """Set the number of modes."""
        self._n_mode = n_mode
    
    def get_scale(self) -> Optional[np.ndarray]:
        """Get the scale array."""
        return self._scale
    
    def set_scale(self, scale: np.ndarray) -> None:
        """Set the scale array."""
        self._scale = scale
    
    def get_a_ref(self) -> Optional[np.ndarray]:
        """Get the reference Fourier coefficients."""
        return self._a_ref
    
    def set_a_ref(self, a_ref: np.ndarray) -> None:
        """Set the reference Fourier coefficients."""
        self._a_ref = a_ref
    
    def get_mean_mmd(self) -> Optional[float]:
        """Get the mean MMD value."""
        return self._mean_mmd
    
    def set_mean_mmd(self, mean_mmd: float) -> None:
        """Set the mean MMD value."""
        self._mean_mmd = mean_mmd
    
    def get_std_mmd(self) -> Optional[float]:
        """Get the standard deviation of MMD values."""
        return self._std_mmd
    
    def set_std_mmd(self, std_mmd: float) -> None:
        """Set the standard deviation of MMD values."""
        self._std_mmd = std_mmd
    
    def __str__(self) -> str:
        """String representation."""
        return (f"FourierMMDFitting{{randomSeed={self._random_seed}, "
                f"deltaStat={self._delta_stat}, n_mode={self._n_mode}}}")