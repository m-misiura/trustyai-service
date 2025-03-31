from typing import Dict, Optional
from src.service.metrics.drift.ks_test.gk_sketch import GKSketch

class ApproxKSFitting:
    """Store precomputed sketches for ApproxKSTest."""
    
    def __init__(self, fit_sketches: Dict[str, GKSketch]):
        """Initialize with precomputed sketches."""
        self._fit_sketches = fit_sketches  
    
    def get_fit_sketches(self) -> Dict[str, GKSketch]:
        """Get the precomputed sketches."""
        return self._fit_sketches  
    
    def __str__(self) -> str:
        return f"ApproxKSFitting{{fitSketches={self._fit_sketches}}}"  