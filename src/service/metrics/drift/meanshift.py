import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MeanshiftResult:
    """Results of a Meanshift statistical test."""

    t_stat: float
    p_value: float
    reject_null: bool

    def to_dict(self) -> Dict[str, Union[float, bool]]:
        return {
            "tStat": self.t_stat,
            "pValue": self.p_value,
            "driftDetected": self.reject_null,
        }


class StatisticalSummaryValues:
    """Holder for column statistical properties."""

    def __init__(
        self,
        mean: float,
        variance: float,
        n: int,
        max_val: float,
        min_val: float,
        sum_val: float,
    ):
        self.mean = mean
        self.variance = variance
        self.n = n
        self.max = max_val
        self.min = min_val
        self.sum = sum_val
        self.standard_deviation = np.sqrt(variance) if variance is not None else None

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "StatisticalSummaryValues":
        """Create summary values from a numpy array."""
        # Check if non-numeric data is present
        if not np.issubdtype(arr.dtype, np.number):
            # Convert all numeric values to float, ignore non-numeric
            numeric_arr = np.array([x for x in arr if isinstance(x, (int, float))], dtype=float)
        else:
            numeric_arr = arr
            
        # Filter out NaNs and infinities
        clean_arr = numeric_arr[np.isfinite(numeric_arr)]
        if len(clean_arr) == 0:
            return cls(mean=0, variance=0, n=0, max_val=0, min_val=0, sum_val=0)

        # return cls(
        #     mean=np.mean(clean_arr),
        #     variance=np.var(clean_arr, ddof=1) if len(clean_arr) > 1 else 0,
        #     n=len(clean_arr),
        #     max_val=np.max(clean_arr),
        #     min_val=np.min(clean_arr),
        #     sum_val=np.sum(clean_arr),
        # )
         # Calculate using Java's approach (standard deviation first, then square)
        mean = np.mean(clean_arr)
        # Use biased standard deviation (divides by n) to match Java
        std_dev = np.std(clean_arr, ddof=0) 
        variance = std_dev**2  # Square to get variance
        
        return cls(
            mean=mean,
            variance=variance,
            n=len(clean_arr),
            max_val=np.max(clean_arr),
            min_val=np.min(clean_arr),
            sum_val=np.sum(clean_arr),
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "StatisticalSummaryValues":
        # Handle both dictionary and StatisticalSummaryValues objects
        if hasattr(data, 'dict'):  # It's a Pydantic model
            data = data.dict()
        
        return cls(
            mean=data.get("mean", 0),
            variance=data.get("variance", 0),
            n=data.get("n", 0),
            max_val=data.get("max", 0),
            min_val=data.get("min", 0),
            sum_val=data.get("sum", 0),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean,
            "variance": self.variance,
            "n": self.n,
            "max": self.max,
            "min": self.min,
            "sum": self.sum,
            "standardDeviation": self.standard_deviation,
        }


class Meanshift:
    """Meanshift drift detection algorithm."""

    def __init__(self, fit_stats: Optional[Dict[str, StatisticalSummaryValues]] = None):
        """
        Initialize with precomputed statistics or None

        Args:
            fit_stats: Dictionary mapping column names to their statistical summaries
        """
        self.fit_stats = fit_stats or {}

    @staticmethod
    def _calculate_t_stat(ref_stats: StatisticalSummaryValues, test_stats: StatisticalSummaryValues) -> float:
        """
        Calculate t-statistic using the same formula as Java implementation.
        
        Java uses (ref_mean - test_mean) while scipy uses (test_mean - ref_mean)
        """
        # Match Java's formula: t = (ref_mean - test_mean) / sqrt((ref_var/ref_n) + (test_var/test_n))
        numerator = ref_stats.mean - test_stats.mean
        denominator = np.sqrt((ref_stats.variance / ref_stats.n) + (test_stats.variance / test_stats.n))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator

    @staticmethod
    def _calculate_p_value(t_stat: float, df: int) -> float:
        """Calculate p-value for t-statistic using two-tailed test."""
        # Two-tailed test like Java implementation
        return stats.t.sf(abs(t_stat), df) * 2

    @classmethod
    async def from_model_data(
        cls, model_data, reference_tag: Optional[str] = None
    ) -> "Meanshift":
        """
        Create Meanshift instance from model data

        Args:
            model_data: ModelData instance
            reference_tag: Optional tag to filter reference data

        Returns:
            Meanshift instance fitted with reference data
        """
        # Load reference data
        inputs, _, metadata = await model_data.data()

        # Get column names before potentially filtering data
        input_names, _, metadata_names = await model_data.column_names()

        # Filter by reference tag if provided
        if reference_tag and metadata is not None:
            # Look for data_tag column in a more robust way
            tag_column_indices = [
                i
                for i, name in enumerate(metadata_names)
                if name == "data_tag" or name.endswith(".data_tag")
            ]

            if tag_column_indices:
                mask = metadata[:, tag_column_indices[0]] == reference_tag
                inputs = inputs[mask] if mask.any() else inputs
                logger.info(
                    f"Filtered reference data by tag '{reference_tag}': {np.sum(mask)} rows selected"
                )
            else:
                logger.warning(
                    f"Reference tag '{reference_tag}' provided but no data_tag column found"
                )

        if inputs is None or len(inputs) == 0:
            raise ValueError(
                f"No reference data found for model {model_data.model_name}"
            )

        # Calculate statistics for each column
        fit_stats = {}
        for i, col_name in enumerate(input_names):
            col_data = inputs[:, i]
            # Only process numerical columns
            if np.issubdtype(col_data.dtype, np.number):
                fit_stats[col_name] = StatisticalSummaryValues.from_array(col_data)
                logger.debug(f"Computed reference statistics for column {col_name}")

        return cls(fit_stats)

    @classmethod
    def from_fitting(cls, fitting: Dict[str, Dict]) -> "Meanshift":
        """
        Create Meanshift from precomputed statistics

        Args:
            fitting: Dictionary mapping column names to their statistical summary dictionaries

        Returns:
            Meanshift instance with loaded statistics
        """
        fit_stats = {
            col_name: StatisticalSummaryValues.from_dict(stats_dict)
            for col_name, stats_dict in fitting.items()
        }
        return cls(fit_stats)

    def calculate(
        self, test_data: np.ndarray, column_names: List[str], alpha: float = 0.05
    ) -> Dict[str, MeanshiftResult]:
        """
        Calculate drift for each numerical column

        Args:
            test_data: Array of test data with columns matching the fitted data
            column_names: Names of the columns in test_data
            alpha: Significance level for hypothesis testing

        Returns:
            Dictionary mapping column names to drift results
        """
        # Input validation
        if not isinstance(test_data, np.ndarray):
            raise TypeError("test_data must be a numpy ndarray")

        if not column_names or len(column_names) == 0:
            raise ValueError("column_names cannot be empty")

        if test_data.shape[1] != len(column_names):
            raise ValueError(
                f"Column dimension mismatch: test_data has {test_data.shape[1]} columns but column_names has {len(column_names)}"
            )

        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        result = {}

        for i, col_name in enumerate(column_names):
            logger.debug(f"Processing drift detection for column {col_name}")

            # First check if the column is numerical (before checking reference data)
            test_col = test_data[:, i]
            if not np.issubdtype(test_col.dtype, np.number):
                logger.debug(f"Skipping non-numerical column: {col_name}")
                continue

            # Then check if column is in reference data
            if col_name not in self.fit_stats:
                raise ValueError(
                    f"Passed data not compatible with the mean-shift fitting: no such column in fitting with name {col_name}."
            )

            # Get test column data
            test_col = test_data[:, i]
            if not np.issubdtype(test_col.dtype, np.number):
                logger.debug(f"Skipping non-numerical column: {col_name}")
                continue

            # Skip if insufficient data
            if len(test_col) < 2:
                logger.debug(
                    f"Insufficient data for column {col_name}: {len(test_col)} samples"
                )
                result[col_name] = MeanshiftResult(0, 1.0, False)
                continue

            # Calculate test statistics
            test_stats = StatisticalSummaryValues.from_array(test_col)

            # Calculate t-statistic and p-value using Java-equivalent implementation
            try:
                # Use our custom methods instead of scipy to match Java implementation
                t_stat = self._calculate_t_stat(self.fit_stats[col_name], test_stats)
                df = self.fit_stats[col_name].n + test_stats.n - 2  # Degrees of freedom
                p_value = self._calculate_p_value(t_stat, df)
                
                # Check if null hypothesis should be rejected
                reject = p_value <= alpha

                result[col_name] = MeanshiftResult(t_stat, p_value, reject)
                logger.debug(
                    f"Drift result for {col_name}: t-stat={t_stat:.4f}, p-value={p_value:.4f}, drift detected={reject}"
                )
                
            except Exception as e:
                logger.warning(
                    f"Failed to calculate t-test for column {col_name}: {str(e)}"
                )
                result[col_name] = MeanshiftResult(0, 1.0, False)

        return result