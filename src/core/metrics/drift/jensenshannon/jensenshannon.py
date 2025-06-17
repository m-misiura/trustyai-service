import numpy as np
from typing import List, Tuple, Union, Optional
#from scipy.stats import entropy  # type: ignore

from dataclasses import dataclass


# ╭───────────────────────────────────────────────────────────────────────────╮
# │                      JENSEN-SHANNON DRIFT RESULT CLASS                    │
# ╰───────────────────────────────────────────────────────────────────────────╯
@dataclass
class JensenShannonDriftResult:
    """Result of Jensen-Shannon drift calculation."""

    js_stat: float
    threshold: float
    reject: bool

    def __str__(self) -> str:
        return f"JensenShannonDriftResult: [js_stat={self.js_stat}, threshold={self.threshold}, reject={self.reject}]"

    def get_js_stat(self) -> float:
        return self.js_stat

    def get_threshold(self) -> float:
        return self.threshold

    def is_reject(self) -> bool:
        return bool(self.reject)

# KL divergence with safe p and q multiplication
def kl_divergence_arraywise(p: np.ndarray, q: np.ndarray, axis: Optional[int] = None, epsilon: float = 1e-15) -> np.ndarray:
    """
    Compute Kullback-Leibler divergence between two probability distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        axis: Axis along which to compute the sum. If None, sum over all axes
        epsilon: Small value to avoid division by zero
        
    Returns:
        KL divergence as a numpy array
    """
    p = np.asarray(p)
    q = np.asarray(q)

    if p.shape != q.shape:
        raise ValueError("p and q must have same shape")

    q_safe = np.where(q > 0, q, epsilon)

    mask_p_positive = p > 0
    mask_q_zero = (q == 0) & mask_p_positive

    ratio = np.zeros_like(p)
    ratio[mask_p_positive] = p[mask_p_positive] / q_safe[mask_p_positive]
    log_ratio = np.zeros_like(p)
    log_ratio[mask_p_positive] = np.log(ratio[mask_p_positive])

    # Compute p * log(p/q) for regular finite values
    kl_elements = p * log_ratio

    # Handle q=0 cases (set to infinity where needed)
    if np.any(mask_q_zero):
        kl_elements[mask_q_zero] = np.inf

    # Sum across specified axes
    result: np.ndarray = np.sum(kl_elements, axis=axis)
    return result





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                             JENSEN-SHANNON CLASS                          │
# ╰───────────────────────────────────────────────────────────────────────────╯
#
class JensenShannon:
    """
    Jensen-Shannon divergence
    """

    @staticmethod
    def _validate_tensor_inputs(
        references: np.ndarray, hypotheses: np.ndarray, threshold: Union[float, List[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate input tensors for Jensen-Shannon calculations.

        Args:
            references: The reference tensor
            hypotheses: The hypothesis tensor
            threshold: Threshold(s) for drift detection
        """

        if references is None or hypotheses is None:
            raise ValueError("Reference and hypothesis tensors cannot be None")

        if threshold is None:
            raise ValueError("Threshold cannot be None")

        references = np.asarray(references, dtype=np.float64)
        hypotheses = np.asarray(hypotheses, dtype=np.float64)

        
        if references.size == 0 or hypotheses.size == 0:
            raise ValueError("Input tensors cannot be empty")

        
        if np.isnan(references).any() or np.isnan(hypotheses).any():
            raise ValueError("Input tensors contain NaN values")

        if np.isinf(references).any() or np.isinf(hypotheses).any():
            raise ValueError("Input tensors contain infinity values")

        # Dimensions match
        if references.shape != hypotheses.shape:
            raise ValueError(
                f"Dimensions of references {references.shape} and hypotheses {hypotheses.shape} do not match"
            )

        
        if len(references.shape) < 2:
            raise ValueError(f"Input tensors must have at least 2 dimensions, got shape {references.shape}")

        n_channels = references.shape[1]
        if n_channels <= 0:
            raise ValueError(f"Number of channels must be positive, got {n_channels}")

        return references, hypotheses

    @staticmethod
    def jensen_shannon_divergence(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculates the Jensen-Shannon divergence between two probability distributions.
        """
        try:
            # midpoint distribution m (np arrays handles elemet division)
            m = (p1 + p2) / 2.0

            # KL divergences
            kl_div1 = float(kl_divergence_arraywise(p1, m))
            kl_div2 = float(kl_divergence_arraywise(p2, m))

            # Return JS divergence
            return (kl_div1 + kl_div2) / 2.0

        except Exception as e:
            raise ValueError(f"Error calculating Jensen-Shannon divergence: {str(e)}")

    @staticmethod
    def calculate(
        references: np.ndarray, hypothesis: np.ndarray, threshold: float, normalize: bool = False
    ) -> JensenShannonDriftResult:
        """
        Calculates the average Jensen-Shannon divergence over channel values between reference and hypothesis tensors.

        Args:
            references: The reference tensor (first dimension is samples, second is channels)
            hypotheses: The hypothesis tensor (must have same dimensions as references)
            threshold: Threshold to determine whether the hypothesis is different from reference
            normalize: Whether to normalize via the reference tensor size
        """
        try:
            # Validate inputs using _validate_tensor_inputs
            references, hypothesis = JensenShannon._validate_tensor_inputs(references, hypothesis, threshold)
            n_channels = references.shape[1]
            normalization_factor = references.size

            js_stat = 0.0

            for channel in range(n_channels):
                reference_slice = references[:, channel].flatten()
                hypothesis_slice = hypothesis[:, channel].flatten()
                js_stat += JensenShannon.jensen_shannon_divergence(reference_slice, hypothesis_slice)

                # # More closely emulate Java's getFromSecondAxis behavior
                # ref_shape = references.shape
                # # Extract channel data in same layout as Java
                # reference_slice = references[:, channel].reshape(ref_shape[0], -1)
                # hypothesis_slice = hypothesis[:, channel].reshape(ref_shape[0], -1)

                # # Now flatten each batch item and concatenate them to match Java's behavior
                # ref_flattened = []
                # hyp_flattened = []
                # for i in range(ref_shape[0]):
                #     ref_flattened.extend(reference_slice[i].flatten())
                #     hyp_flattened.extend(hypothesis_slice[i].flatten())

                # js_stat = JensenShannon.jensen_shannon_divergence(np.asarray(ref_flattened), np.asarray(hyp_flattened))

            if normalize:
                js_stat /= normalization_factor

            reject = js_stat > threshold
            return JensenShannonDriftResult(js_stat, threshold, reject)

        except (ValueError, TypeError) as e:
            raise type(e)(f"Error in Jensen-Shannon calculation: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in Jensen-Shannon calculation: {str(e)}")

    @staticmethod
    def calculate_per_channel(
        references: np.ndarray, hypotheses: np.ndarray, thresholds: List[float], normalize: bool = False
    ) -> List[JensenShannonDriftResult]:
        """
        Calculates the Jensen-Shannon divergence for each channel between reference and hypothesis tensors.

        Args:
            references: The reference tensor (first dimension is samples, second is channels)
            hypotheses: The hypothesis tensor (must have same dimensions as references)
            thresholds: Per-channel thresholds to determine differences
            normalize: Whether to normalize via the per-channel tensor size

        Returns:
            A list of JensenShannonDriftResult, where the ith element contains the computed
            Jensen-Shannon statistic for the ith channel and whether the threshold is violated.
        """
        try:
            # Validate inputs using _validate_tensor_inputs
            references, hypotheses = JensenShannon._validate_tensor_inputs(references, hypotheses, thresholds)

            n_channels = references.shape[1]
            results = []

            for channel in range(n_channels):
                reference_slice = references[:, channel].flatten()
                hypothesis_slice = hypotheses[:, channel].flatten()
                js_stat = JensenShannon.jensen_shannon_divergence(reference_slice, hypothesis_slice)

                if normalize:
                    js_stat /= reference_slice.size

                reject = js_stat > thresholds[channel]

                results.append(JensenShannonDriftResult(js_stat, thresholds[channel], bool(reject)))

            return results

        except (ValueError, TypeError) as e:
            raise type(e)(f"Error in Jensen-Shannon per-channel calculation: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in Jensen-Shannon per-channel calculation: {str(e)}")
