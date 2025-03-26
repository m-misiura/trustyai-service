import pytest
import numpy as np
from typing import List, Tuple

from src.service.metrics.drift.kstest import KSTest, HypothesisTestResult

# Constants matching Java implementation
COL_SIZE = 4
SAMPLE_SIZE = 10000
RANDOM_SEED = 0
ALPHA = 0.05  # Significance level

class TestKSTest:
    """Test the KSTest drift detection implementation."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.dists = self.get_normal_distributions(RANDOM_SEED)
    
    def get_normal_distributions(self, random_seed: int) -> np.ndarray:
        """Produces 4 Normal distributions with various means and stds."""
        np.random.seed(random_seed)
        dists = np.zeros((COL_SIZE, SAMPLE_SIZE))
        
        # Generate samples matching the Java implementation
        dists[0] = np.random.normal(0.0, 1.0, SAMPLE_SIZE)  # N(0,1)
        dists[1] = np.random.normal(1.0, 1.0, SAMPLE_SIZE)  # N(1,1)
        dists[2] = np.random.normal(1.0, 1.0, SAMPLE_SIZE)  # N(1,1)
        dists[3] = np.random.normal(0.0, 2.0, SAMPLE_SIZE)  # N(0,2)
        
        return dists
    
    def generate_single_column(self, col: int) -> Tuple[np.ndarray, List[str]]:
        """Generate single column data from mock data."""
        data = self.dists[col].reshape(-1, 1)
        column_names = ['0']
        return data, column_names
    
    def generate_multi_column(self, col_idxs: List[int], col_prefix: str) -> Tuple[np.ndarray, List[str]]:
        """Generate multi-column data."""
        data = np.column_stack([self.dists[idx] for idx in col_idxs])
        column_names = [f"{col_prefix}{i}" for i in range(len(col_idxs))]
        return data, column_names
    
    def test_univariate_normal_distributions_no_shift(self):
        """Test KSTest on normal distributions with no shift."""
        # N(1,1) vs N(1,1)
        data1, names = self.generate_single_column(1)
        data2, _ = self.generate_single_column(2)
        
        ks = KSTest()
        result = ks.calculate(data1, data2, names, ALPHA)
        
        assert len(result) == 1
        assert result[names[0]].p_value >= 0.01
        assert result[names[0]].statistic > 0.0
        assert result[names[0]].reject_null == False
    
    def test_univariate_normal_distributions_mean_shift(self):
        """Test KSTest on normal distributions with mean shift."""
        # N(0,1) vs N(1,1)
        data1, names = self.generate_single_column(0)
        data2, _ = self.generate_single_column(1)
        
        ks = KSTest()
        result = ks.calculate(data1, data2, names, ALPHA)
        
        assert len(result) == 1
        assert result[names[0]].p_value <= 0.01
        assert result[names[0]].statistic > 0.0
        assert result[names[0]].reject_null == True
    
    def test_univariate_normal_distributions_variance_shift(self):
        """Test KSTest on normal distributions with variance shift."""
        # N(0,1) vs N(0,2)
        data1, names = self.generate_single_column(0)
        data2, _ = self.generate_single_column(3)
        
        ks = KSTest()
        result = ks.calculate(data1, data2, names, ALPHA)
        
        assert len(result) == 1
        assert result[names[0]].p_value <= 0.01
        assert result[names[0]].statistic > 0.0
        assert result[names[0]].reject_null == True
    
    def test_multi_column_ks_test(self):
        """Test KSTest on multiple columns."""
        # Set new random seed as in Java test
        self.dists = self.get_normal_distributions(34)
        
        # Create column indices: 0,1,2 vs 1,2,3
        idx1 = list(range(COL_SIZE - 1))  # [0,1,2]
        idx2 = list(range(1, COL_SIZE))   # [1,2,3]
        
        data1, names = self.generate_multi_column(idx1, "normal_dist")
        data2, _ = self.generate_multi_column(idx2, "normal_dist")
        
        ks = KSTest()
        result = ks.calculate(data1, data2, names, ALPHA)
        
        # Verify first column shows drift (N(0,1) vs N(1,1))
        assert result[names[0]].p_value <= 0.05
        assert result[names[0]].statistic > 0.0
        assert result[names[0]].reject_null == True
        
        # Second column should not show drift (N(1,1) vs N(1,1))
        assert result[names[1]].p_value >= 0.05
        assert result[names[1]].statistic > 0.0
        assert result[names[1]].reject_null == False
        
        # Third column should show drift (N(1,1) vs N(0,2))
        assert result[names[2]].p_value <= 0.05
        assert result[names[2]].statistic > 0.0
        assert result[names[2]].reject_null == True