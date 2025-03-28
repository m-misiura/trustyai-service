import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Set
import logging
import math

from src.service.metrics.drift.ks_test.gk_sketch import GKSketch, GKException
from src.service.metrics.drift.ks_test.approx_ks_fitting import ApproxKSFitting
from src.service.metrics.drift.ks_test.ks_test import HypothesisTestResult

logger = logging.getLogger(__name__)

class ApproxKSTest:
    """
    Implements Approximate Kolmogorov-Smirnov Test using Greenwald-Khanna epsilon sketch as described in 
    A. Lall, "Data streaming algorithms for the kolmogorov-smirnov test" in 2015 IEEE International 
    Conference on Big Data (Big Data), 2015, pp. 95–104
    """
    
    def __init__(self, eps: float = 0.01, train_data: Optional[np.ndarray] = None, 
                 column_names: Optional[List[str]] = None, approx_ks_fitting: Optional[ApproxKSFitting] = None):
        """Initialize ApproxKSTest."""
        self.eps = eps
        
        if approx_ks_fitting is not None:
            self.train_gk_sketches = approx_ks_fitting.get_fit_sketches()
        elif train_data is not None and column_names is not None:
            # Precompute GKSketch of training data
            ks_fitting = self.precompute(train_data, column_names, eps)
            self.train_gk_sketches = ks_fitting.get_fit_sketches()
        else:
            raise ValueError("Either train_data and column_names or approx_ks_fitting must be provided")
    
    @staticmethod
    def precompute(train_data: np.ndarray, column_names: List[str], eps: float = 0.01) -> ApproxKSFitting:
        """
        Precompute GKSketches for training data.
        
        Args:
            train_data: Training data as numpy array
            column_names: Names of the columns
            eps: Epsilon parameter for sketch approximation
        
        Returns:
            ApproxKSFitting object with precomputed sketches
        """
        sketches = {}
        
        for i, col_name in enumerate(column_names):
            # Only process numerical columns
            if np.issubdtype(train_data[:, i].dtype, np.number):
                # Build epsilon sketch for given column
                sketch = GKSketch(eps)
                for val in train_data[:, i]:
                    if np.isfinite(val):  # Skip NaN and infinity values
                        sketch.insert(val)
                
                sketches[col_name] = sketch
        
        return ApproxKSFitting(sketches)
    
    def calculate(self, test_data: np.ndarray, column_names: List[str], 
                  alpha: float = 0.05) -> Dict[str, HypothesisTestResult]:
        """
        Calculate drift for each numerical column using approximate KS-test.
        
        Args:
            test_data: Test data as numpy array
            column_names: Names of the columns
            alpha: Significance level (default: 0.05)
        
        Returns:
            Dictionary of column names to HypothesisTestResult
        """
        # Input validation
        if not isinstance(test_data, np.ndarray):
            raise TypeError("Test data must be a numpy ndarray")
            
        if not column_names or len(column_names) == 0:
            raise ValueError("column_names cannot be empty")
            
        if test_data.shape[1] != len(column_names):
            raise ValueError(f"Column dimension mismatch: test_data has {test_data.shape[1]} columns, "
                             f"but column_names has {len(column_names)}")
            
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
            
        result = {}
        
        for i, col_name in enumerate(column_names):
            logger.debug(f"Processing approximate KS test for column {col_name}")
            
            # Skip non-numerical columns
            if not np.issubdtype(test_data[:, i].dtype, np.number):
                logger.debug(f"Skipping non-numerical column: {col_name}")
                continue
                
            # Check if column exists in training sketches
            if col_name not in self.train_gk_sketches:
                raise ValueError(f"Column {col_name} not found in precomputed sketches")
                
            # Get train sketch
            train_sketch = self.train_gk_sketches[col_name]
            
            # Skip if insufficient data
            if test_data.shape[0] < 2 or train_sketch.size() < 2:
                logger.debug(f"Insufficient data for column {col_name}")
                result[col_name] = HypothesisTestResult(0, 1.0, False)
                continue
                
            try:
                # Build sketch for test data
                test_sketch = GKSketch(self.eps)
                for val in test_data[:, i]:
                    if np.isfinite(val):  # Skip NaN and infinity values
                        test_sketch.insert(val)
                        
                # Compute KS distance
                d = self._compute_ks_distance(train_sketch, test_sketch)
                
                # Compute p-value
                p_value = self._compute_pvalue(d, train_sketch.get_numx(), test_sketch.get_numx())
                
                # Check if null hypothesis should be rejected
                reject = p_value <= alpha
                
                result[col_name] = HypothesisTestResult(d, p_value, reject)
                logger.debug(f"Approximate KS test result for {col_name}: "
                             f"statistic={d:.4f}, p-value={p_value:.4f}, drift detected={reject}")
                
            except Exception as e:
                logger.warning(f"Failed to calculate approximate KS test for column {col_name}: {str(e)}")
                result[col_name] = HypothesisTestResult(0, 1.0, False)
                
        return result
    
    def _compute_pvalue(self, d: float, numx: int, numx2: int) -> float:
        """
        Compute p-value for KS test statistic.
        
        This is equivalent to KolmogorovSmirnovTest.approximateP() in Apache Commons Math.
        """
        # Use scipy's KS test p-value calculation
        n = numx
        m = numx2
        
        # This follows the Apache Commons Math implementation
        if n == 0 or m == 0:
            return 1.0
            
        # Calculate p-value using asymptotic distribution
        # The formula comes from Stephens, M. A. "Use of the Kolmogorov-Smirnov, 
        # Cramer-Von Mises and Related Statistics Without Extensive Tables"
        nm = n * m / (n + m)
        z = d * math.sqrt(nm)
        
        # Use the Kolmogorov distribution
        return stats.kstwo.sf(z)
    
    def _compute_ks_distance(self, train_sketch: GKSketch, test_sketch: GKSketch) -> float:
        """
        Compute KS distance between two sketches using Lall's algorithm.
        
        Algorithm 2 in Lall's paper.
        """
        # Extract values from sketches
        train_sketch_values = [t.left for t in train_sketch.get_summary()]
        test_sketch_values = [t.left for t in test_sketch.get_summary()]
        
        # Merge values
        merged_sketches = set(train_sketch_values) | set(test_sketch_values)
        
        # Get sizes
        train_size = train_sketch.get_numx()
        test_size = test_sketch.get_numx()
        
        # Find maximum difference
        max_d = 0.0
        for v in merged_sketches:
            try:
                train_approx_rank = train_sketch.rank(v)
                test_approx_rank = test_sketch.rank(v)
                
                train_approx_prob = train_approx_rank / train_size
                test_approx_prob = test_approx_rank / test_size
                
                v_dist = abs(train_approx_prob - test_approx_prob)
                max_d = max(v_dist, max_d)
            except GKException as e:
                logger.warning(f"Error computing rank: {e.message}")
                continue
            
        return max_d
    
    @classmethod
    async def from_model_data(cls, model_data, reference_tag: Optional[str] = None, 
                             epsilon: float = 0.01) -> "ApproxKSTest":
        """Create ApproxKSTest instance from model data."""
        # Get reference data
        inputs, _, metadata = await model_data.data()
        input_names, _, metadata_names = await model_data.column_names()
        
        # Filter by reference tag if provided
        if reference_tag and metadata is not None:
            tag_column_indices = [
                i for i, name in enumerate(metadata_names)
                if name == "data_tag" or name.endswith(".data_tag")
            ]
            
            if tag_column_indices:
                mask = metadata[:, tag_column_indices[0]] == reference_tag
                inputs = inputs[mask] if mask.any() else inputs
                logger.info(f"Filtered reference data by tag '{reference_tag}': {np.sum(mask)} rows selected")
        
        # Create ApproxKSTest
        return cls(eps=epsilon, train_data=inputs, column_names=input_names)
    
    def __str__(self) -> str:
        """String representation."""
        return f"ApproxKSTest [eps={self.eps}, trainGKSketches={self.train_gk_sketches}]"