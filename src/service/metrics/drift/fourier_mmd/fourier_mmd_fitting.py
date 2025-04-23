from typing import Optional, Dict, Any
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the FourierMMDFitting to a dictionary that can be serialized.
        
        Returns:
            Dictionary representation of the fitting
        """
        result = {
            "random_seed": self._random_seed,
            "delta_stat": self._delta_stat,
            "n_mode": self._n_mode,
            "mean_mmd": self._mean_mmd,
            "std_mmd": self._std_mmd
        }
        
        # Convert numpy arrays to lists for serialization
        if self._scale is not None:
            result["scale"] = self._scale.tolist()
        
        if self._a_ref is not None:
            result["a_ref"] = self._a_ref.tolist()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FourierMMDFitting':
        """
        Create a FourierMMDFitting instance from a dictionary.
        
        Args:
            data: Dictionary containing the fitting data
            
        Returns:
            A new FourierMMDFitting instance
        """
        instance = cls(
            random_seed=data.get("random_seed"),
            delta_stat=data.get("delta_stat"),
            n_mode=data.get("n_mode")
        )
        
        # Set mean and std
        if "mean_mmd" in data:
            instance.set_mean_mmd(data["mean_mmd"])
        
        if "std_mmd" in data:
            instance.set_std_mmd(data["std_mmd"])
        
        # Convert lists back to numpy arrays
        if "scale" in data:
            instance.set_scale(np.array(data["scale"]))
        
        if "a_ref" in data:
            instance.set_a_ref(np.array(data["a_ref"]))
        
        return instance