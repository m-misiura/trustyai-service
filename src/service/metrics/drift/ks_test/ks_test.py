import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from src.service.metrics.drift.hypothesis_test_result import HypothesisTestResult
import logging

logger = logging.getLogger(__name__)


class KSTest:
    """Kolmogorov-Smirnov test for drift detection."""
    
    def __init__(self):
        """Initialize KSTest."""
        pass
    
    def calculate(self, ref_data: np.ndarray, test_data: np.ndarray, 
                 column_names: List[str], alpha: float = 0.05) -> Dict[str, HypothesisTestResult]:
        """
        Calculate drift for each numerical column using KS-test
        
        Args:
            ref_data: Reference data as numpy array
            test_data: Test data as numpy array
            column_names: Names of the columns
            alpha: Significance level for hypothesis testing
            
        Returns:
            Dictionary mapping column names to drift results
        """
        # Input validation
        if not isinstance(ref_data, np.ndarray) or not isinstance(test_data, np.ndarray):
            raise TypeError("Data must be numpy ndarrays")
            
        if not column_names or len(column_names) == 0:
            raise ValueError("column_names cannot be empty")
            
        if ref_data.shape[1] != len(column_names) or test_data.shape[1] != len(column_names):
            raise ValueError(f"Column dimension mismatch: ref_data has {ref_data.shape[1]} columns, "
                             f"test_data has {test_data.shape[1]} columns, but column_names has {len(column_names)}")
            
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
            
        result = {}
        
        for i, col_name in enumerate(column_names):
            logger.debug(f"Processing KS test for column {col_name}")
            
            # Get column data
            ref_col = ref_data[:, i]
            test_col = test_data[:, i]
            
            # Skip non-numerical columns
            if not np.issubdtype(ref_col.dtype, np.number) or not np.issubdtype(test_col.dtype, np.number):
                logger.debug(f"Skipping non-numerical column: {col_name}")
                continue
                
            # Skip if insufficient data
            if len(ref_col) < 2 or len(test_col) < 2:
                logger.debug(f"Insufficient data for column {col_name}")
                result[col_name] = HypothesisTestResult(0, 1.0, False)
                continue
                
            try:
                # Filter out NaNs and infinities
                ref_col_clean = ref_col[np.isfinite(ref_col)]
                test_col_clean = test_col[np.isfinite(test_col)]
                
                if len(ref_col_clean) < 2 or len(test_col_clean) < 2:
                    logger.debug(f"Insufficient clean data for column {col_name} after filtering NaNs/Infs")
                    result[col_name] = HypothesisTestResult(0, 1.0, False)
                    continue
                
                # Calculate KS statistic and p-value
                ks_stat, p_value = stats.ks_2samp(ref_col_clean, test_col_clean)
                
                # Check if null hypothesis should be rejected
                reject = p_value <= alpha
                
                result[col_name] = HypothesisTestResult(ks_stat, p_value, reject)
                logger.debug(f"KS test result for {col_name}: statistic={ks_stat:.4f}, p-value={p_value:.4f}, drift detected={reject}")
                
            except Exception as e:
                logger.warning(f"Failed to calculate KS test for column {col_name}: {str(e)}")
                result[col_name] = HypothesisTestResult(0, 1.0, False)
                
        return result
    
    @classmethod
    async def from_model_data(cls, model_data, reference_tag: Optional[str] = None) -> "KSTest":
        """
        Create KSTest instance from model data
        
        Args:
            model_data: ModelData instance
            reference_tag: Optional tag to filter reference data
            
        Returns:
            KSTest instance
        """
        # KSTest doesn't need to precompute anything from reference data
        return cls()