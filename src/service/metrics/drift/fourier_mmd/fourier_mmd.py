import numpy as np
from typing import List, Optional, Any
import logging
import random
from scipy.stats import norm  # type: ignore

from src.service.metrics.drift.fourier_mmd.fourier_mmd_fitting import FourierMMDFitting
from src.service.metrics.drift.hypothesis_test_result import HypothesisTestResult

logger = logging.getLogger(__name__)

class FourierMMD:
    """
    FourierMMD identifies drift using a random Fourier approximation to a 
    Radial-Basis-Function Kernel Maximum Mean Discrepancy.
    
    References:
    .. [#0] `Ji Zhao, Deyu Meng, 'FastMMD: Ensemble of Circular Discrepancy for
    Efficient Two-Sample Test'
    <https://arxiv.org/abs/1405.2664>`_
    .. [#1] `Olivier Goudet, et al. 'Learning Functional Causal Models with
    Generative Neural Networks'
    <https://arxiv.org/abs/1709.05321>`_
    """
    
    def __init__(self, 
                 train_data: Optional[np.ndarray] = None,
                 column_names: Optional[List[str]] = None,
                 delta_stat: bool = False, 
                 n_test: int = 100, 
                 n_window: int = 100,
                 sig: float = 1.0, 
                 random_seed: int = 42, 
                 n_mode: int = 100,
                 epsilon: float = 1e-6, 
                 fourier_mmd_fitting: Optional[FourierMMDFitting] = None):
        """
        Initialize the FourierMMD drift detector.
        
        Args:
            train_data: Training data as numpy array
            column_names: Names of the columns
            delta_stat: If True, compute MMD score for Dx = x[t+1]-x[t]
            n_test: Number of MMD scores to compute
            n_window: Number of samples to compute a MMD score
            sig: Sigma, a scale parameter of the kernel
            random_seed: The seed for random number generation
            n_mode: Number of Fourier modes to approximate the kernel
            epsilon: Minimum value for standard deviation
            fourier_mmd_fitting: Pre-computed fitting data
        """
        self._normal_distribution = norm(0, 1)
        
        if fourier_mmd_fitting is not None:
            self._fit_stats = fourier_mmd_fitting
        elif train_data is not None and column_names is not None:
            self._fit_stats = self.precompute(train_data, column_names, delta_stat, 
                                           n_test, n_window, sig, random_seed, 
                                           n_mode, epsilon)
        else:
            raise ValueError("Either train_data and column_names or fourier_mmd_fitting must be provided")
    
    def get_normal_distribution(self) -> norm:
        """Get the normal distribution."""
        return self._normal_distribution
    
    def get_fit_stats(self)  -> FourierMMDFitting:
        """Get the fit statistics."""
        return self._fit_stats
    
    @staticmethod
    def precompute(data: np.ndarray, 
                  column_names: List[str], 
                  delta_stat: bool, 
                  n_test: int,
                  n_window: int, 
                  sig: float, 
                  random_seed: int, 
                  n_mode: int, 
                  epsilon: float) -> FourierMMDFitting:
        """
        Precompute the FourierMMD drift model statistics.
        
        Args:
            data: Training data
            column_names: Names of the columns
            delta_stat: If True, compute MMD score for Dx = x[t+1]-x[t]
            n_test: Number of MMD scores to compute
            n_window: Number of samples to compute a MMD score
            sig: Sigma, a scale parameter of the kernel
            random_seed: The seed for random number generation
            n_mode: Number of Fourier modes
            epsilon: Minimum value for standard deviation
            
        Returns:
            FourierMMDFitting object with precomputed statistics
        """
        # Create fitting object
        computed_stats = FourierMMDFitting(random_seed, delta_stat, n_mode)
        
        # Get only numeric columns
        numeric_data = data
        num_columns = numeric_data.shape[1]
        
        # Apply delta if requested
        if delta_stat:
            x_in = FourierMMD.delta(numeric_data, num_columns)
        else:
            x_in = numeric_data
            
        num_rows = x_in.shape[0]
        
        if num_rows == 0:
            raise ValueError("Dataframe is empty")
        
        # Calculate scales for each column
        std_values = np.std(x_in, axis=0)
        scale_array = np.array([max(std, epsilon) * sig for std in std_values])
        computed_stats.set_scale(scale_array)
        
        # Set the random seed for reproducibility
        np.random.seed(random_seed)
        rg = np.random.RandomState(random_seed)
        
        # Generate wave numbers and bias
        wave_num = FourierMMD.get_wave_num(num_columns, rg, n_mode)
        bias = FourierMMD.get_bias(rg, n_mode)
        
        # Determine number of data points to use
        ndata = min(n_window * n_test, num_rows)
        if ndata <= n_window:
            raise ValueError(f"n_window must be less than {ndata}")
        
        # Sample random rows
        idxs = list(range(num_rows))
        random.shuffle(idxs)
        idxs = idxs[:ndata]
        
        # Scale the data
        x1 = np.zeros((ndata, num_columns))
        for i, idx in enumerate(idxs):
            x1[i] = x_in[idx]
        
        x1_scaled = FourierMMD.get_x_scaled(num_columns, ndata, x1, scale_array)
        
        # Compute random Fourier coefficients
        a_ref = FourierMMD.random_fourier_coefficients(x1_scaled, wave_num, bias, ndata, n_mode)
        computed_stats.set_a_ref(a_ref)
        
        # Compute reference MMD scores
        sample_mmd = np.zeros(n_test)
        
        for i in range(n_test):
            # Select random window
            index_size = ndata - n_window
            idx = int(rg.random() * index_size)
            
            # Create windowed data
            x_windowed = np.zeros((n_window, num_columns))
            for r in range(idx, idx + n_window):
                for c in range(num_columns):
                    x_windowed[r - idx, c] = x_in[r, c]
            
            # Scale the windowed data
            for r in range(n_window):
                for c in range(num_columns):
                    x_windowed[r, c] /= scale_array[c]
            
            # Compute Fourier coefficients
            a_comp = FourierMMD.random_fourier_coefficients(x_windowed, wave_num, bias, n_window, n_mode)
            
            # Calculate MMD
            mmd = 0.0
            for c in range(n_mode):
                dist = a_ref[c] - a_comp[c]
                term = dist * dist
                mmd += term
            
            sample_mmd[i] = mmd
        
        # Filter out NaN values
        sample_mmd2 = [mmd for mmd in sample_mmd if not np.isnan(mmd)]
        
        if len(sample_mmd2) == 0:
            raise ValueError("sampleMMD2 length is zero")
        
        # Calculate mean and std
        sample_mmd2_no_nan = np.array(sample_mmd2)
        mean_mmd = np.mean(sample_mmd2_no_nan)
        std_mmd = np.std(sample_mmd2_no_nan)
        
        computed_stats.set_mean_mmd(float(mean_mmd))
        computed_stats.set_std_mmd(float(std_mmd))
        
        return computed_stats
    
    def calculate(self, data: np.ndarray, threshold: float, gamma: float) -> HypothesisTestResult:
        """
        Calculate drift relative to the precomputed data.
        
        Args:
            data: Test data
            threshold: If probability of "data MMD score > (mean_mmd+std_mmd*gamma)" 
                       is larger than threshold, we flag drift
            gamma: Sets threshold to flag drift
        
        Returns:
            HypothesisTestResult with drift information
        """
        # Get numeric columns
        numeric_data = data
        num_columns = numeric_data.shape[1]
        
        # Apply delta if needed
        if self._fit_stats.is_delta_stat():
            x_in = self.delta(numeric_data, num_columns)
        else:
            x_in = numeric_data
        
        # Set RNG to same seed for reproducibility
        np.random.seed(self._fit_stats.get_random_seed())
        rg = np.random.RandomState(self._fit_stats.get_random_seed()) 
        
        # Generate wave numbers and bias
        wave_num = self.get_wave_num(num_columns, rg, self._fit_stats.get_n_mode())
        bias = self.get_bias(rg, self._fit_stats.get_n_mode())
        
        num_rows = x_in.shape[0]
        
        # Get rows as list
        x_in_rows = []
        for r in range(num_rows):
            x_in_rows.append(x_in[r])
        
        # Scale the data
        x1 = self.get_x_scaled(num_columns, num_rows, x_in, self._fit_stats.get_scale())
        
        # Compute random Fourier coefficients
        a_comp = self.random_fourier_coefficients(x1, wave_num, bias, num_rows, 
                                                self._fit_stats.get_n_mode())
        
        # Calculate MMD
        mmd = 0.0
        for c in range(self._fit_stats.get_n_mode()):
            diff = self._fit_stats.get_a_ref()[c] - a_comp[c]
            term = diff * diff
            mmd += term
        
        # Calculate drift score
        drift_score = max((mmd - self._fit_stats.get_mean_mmd()) / self._fit_stats.get_std_mmd(), 0)
        
        # Calculate p-value using normal distribution
        cdf = self._normal_distribution.cdf(gamma - drift_score)
        p_value = 1.0 - cdf
        
        return HypothesisTestResult(drift_score, p_value, p_value > threshold)
    
    @staticmethod
    def delta(data: np.ndarray, num_columns: int) -> np.ndarray:
        """
        Calculate the delta between consecutive rows.
        
        Args:
            data: Input data
            num_columns: Number of columns
            
        Returns:
            Delta of the data
        """
        # Tail of data (all rows except first)
        x_in = data[1:].copy()
        
        for r in range(x_in.shape[0]):
            for c in range(num_columns):
                x_in[r, c] = x_in[r, c] - data[r, c]
                
        return x_in
    
    @staticmethod
    def get_wave_num(num_columns: int, rg: np.random.RandomState, n_mode: int) -> np.ndarray:
        """
        Generate wave numbers.
        
        Args:
            num_columns: Number of columns
            rg: Random number generator
            n_mode: Number of modes
            
        Returns:
            Wave number matrix
        """
        wave_num = np.zeros((num_columns, n_mode))
        for i in range(num_columns):
            for j in range(n_mode):
                wave_num[i, j] = rg.normal(0, 1)
                
        return wave_num
    
    @staticmethod
    def get_bias(rg: np.random.RandomState, n_mode: int) -> np.ndarray:
        """
        Generate bias.
        
        Args:
            rg: Random number generator
            n_mode: Number of modes
            
        Returns:
            Bias matrix
        """
        bias = np.zeros((1, n_mode))
        for i in range(n_mode):
            bias[0, i] = rg.random() * 2.0 * np.pi
            
        return bias
    
    @staticmethod
    def get_x_scaled(num_columns: int, ndata: int, x1: np.ndarray, 
                   scale_array: np.ndarray) -> np.ndarray:
        """
        Scale input data.
        
        Args:
            num_columns: Number of columns
            ndata: Number of data points
            x1: Input data
            scale_array: Scale factors
            
        Returns:
            Scaled data
        """
        x1_scaled = np.zeros((ndata, num_columns))
        
        for row in range(ndata):
            for col in range(num_columns):
                col_value = x1[row, col]
                scaled_col_value = col_value / scale_array[col]
                x1_scaled[row, col] = scaled_col_value
                
        return x1_scaled
    
    @staticmethod
    def random_fourier_coefficients(x: np.ndarray, wave_num: np.ndarray, 
                                  bias: np.ndarray, ndata: int, n_mode: int) -> np.ndarray:
        """
        Compute random Fourier coefficients.
        
        Args:
            x: Input data
            wave_num: Wave numbers
            bias: Bias values
            ndata: Number of data points
            n_mode: Number of modes
            
        Returns:
            Fourier coefficients
        """
        # Matrix multiplication
        product = np.matmul(x, wave_num)
        
        # Apply cosine
        r_cos = np.zeros((ndata, n_mode))
        for r in range(ndata):
            for c in range(n_mode):
                entry = product[r, c]
                new_entry = entry + bias[0, c]
                r_cos[r, c] = np.cos(new_entry)
        
        # Calculate average and scale
        a_ref = np.zeros(n_mode)
        multiplier = np.sqrt(2.0 / n_mode)
        
        for c in range(n_mode):
            sum_val = 0.0
            for r in range(ndata):
                sum_val += r_cos[r, c]
                
            a_ref[c] = (sum_val / ndata) * multiplier
            
        return a_ref
    
    @classmethod
    async def from_model_data(cls, model_data: Any, reference_tag: Optional[str] = None,
                            delta_stat: bool = False, n_test: int = 100, n_window: int = 100,
                            sig: float = 1.0, random_seed: int = 42, n_mode: int = 100,
                            epsilon: float = 1e-6) -> "FourierMMD":
        """
        Create FourierMMD instance from model data.
        
        Args:
            model_data: Model data object
            reference_tag: Optional tag to filter reference data
            delta_stat: Whether to use delta statistics
            n_test: Number of test iterations
            n_window: Window size
            sig: Sigma parameter
            random_seed: Random seed
            n_mode: Number of modes
            epsilon: Minimum standard deviation
            
        Returns:
            Configured FourierMMD instance
        """
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
        
        # Create FourierMMD
        return cls(train_data=inputs, column_names=input_names, delta_stat=delta_stat,
                 n_test=n_test, n_window=n_window, sig=sig, random_seed=random_seed,
                 n_mode=n_mode, epsilon=epsilon)