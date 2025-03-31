class HypothesisTestResult:
    """
    Result object for statistical hypothesis tests.
    
    A direct port of the Java class org.kie.trustyai.metrics.drift.HypothesisTestResult.
    """
    
    def __init__(self, stat_val: float, p_value: float, reject: bool):
        """
        Initialize the hypothesis test result.
        
        Args:
            stat_val: The test statistic value
            p_value: The p-value of the test
            reject: Whether the null hypothesis is rejected
        """
        # Use the underscore prefix to indicate these are "private" fields
        self._stat_val = stat_val
        self._p_value = p_value
        self._reject = reject
    
    def get_p_value(self) -> float:
        """Get the p-value of the test."""
        return self._p_value
    
    def is_reject(self) -> bool:
        """Check if the null hypothesis is rejected."""
        return self._reject
    
    def get_stat_val(self) -> float:
        """Get the test statistic value."""
        return self._stat_val