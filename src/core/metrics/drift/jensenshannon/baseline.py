import numpy as np
from typing import List, Optional, Union
import random
from .jensenshannon import JensenShannon


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                   JENSEN-SHANNON BASELINE CLASS                                      ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
class JensenShannonBaseline:
    """
    Cross-validation functions to determine a sensible baseline threshold for Jensen-Shannon
    Adapted from @christinaexyou's original algorithm
    """

    def __init__(self, min_threshold: float, max_threshold: float, avg_threshold: float):
        """
        Initialize JensenShannonBaseline.

        Args:
            min_threshold: Minimum threshold value found during cv
            max_threshold: Maximum threshold value found during cv
            avg_threshold: Average threshold value found during cv
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.avg_threshold = avg_threshold

    def set_min_threshold(self, min_threshold: float) -> None:
        """Set the minimum threshold."""
        self.min_threshold = min_threshold

    def set_max_threshold(self, max_threshold: float) -> None:
        """Set the maximum threshold."""
        self.max_threshold = max_threshold

    def set_avg_threshold(self, avg_threshold: float) -> None:
        """Set the average threshold."""
        self.avg_threshold = avg_threshold

    def get_min_threshold(self) -> float:
        """Get the minimum threshold."""
        return self.min_threshold

    def get_max_threshold(self) -> float:
        """Get the maximum threshold."""
        return self.max_threshold

    def get_avg_threshold(self) -> float:
        """Get the average threshold."""
        return self.avg_threshold

    def __str__(self) -> str:
        return (
            "JensenShannonBaseline{"
            + "minThreshold="
            + str(self.min_threshold)
            + ", maxThreshold="
            + str(self.max_threshold)
            + ", avgThreshold="
            + str(self.avg_threshold)
            + "}"
        )

    @staticmethod
    def _validate_baseline_inputs(
        references: np.ndarray,
        num_cv: int,
        cv_size: int,
    ) -> np.ndarray:
        """
        Validate input parameters for baseline calculations.

        Args:
            references: The reference tensor
            num_cv: Number of cross-validation iterations
            cv_size: Size of each cross-validation sample
            random_seed: Optional random seed for reproducibility

        Returns:
            Validated reference tensor
        """
        # Input presence and type
        if references is None:
            raise ValueError("Reference tensor cannot be None")

        if not isinstance(num_cv, int) or num_cv <= 0:
            raise ValueError(f"Number of cross-validations must be a positive integer, got {num_cv}")

        if not isinstance(cv_size, int) or cv_size <= 0:
            raise ValueError(f"Cross-validation size must be a positive integer, got {cv_size}")

        references = np.asarray(references, dtype=np.float64)

        # contains at least 1 item
        if references.size == 0:
            raise ValueError("Reference tensor cannot be empty")

        # NaN or infinity values
        if np.isnan(references).any():
            raise ValueError("Reference tensor contains NaN values")

        if np.isinf(references).any():
            raise ValueError("Reference tensor contains infinity values")

        # at least 2 dimensions
        if len(references.shape) < 2:
            raise ValueError(f"Reference tensor must have at least 2 dimensions, got shape {references.shape}")

        # enough samples for cross-validation
        if cv_size * 2 > references.shape[0]:
            raise ValueError(
                f"cv_size*2 cannot be larger than the total number of references: "
                f"cv_size*2={cv_size * 2}, but there are only {references.shape[0]} references."
            )

        n_channels = references.shape[1]
        if n_channels <= 0:
            raise ValueError(f"[ve] - Number of channels must be positive, got {n_channels}")

        return references

    @staticmethod
    def calculate(
        references: np.ndarray,
        num_cv: int,
        cv_size: int,
        random_number_generator: Optional[Union[int, random.Random]] = None,
        normalize: bool = True,
    ) -> "JensenShannonBaseline":
        """
        Calculate baseline JS divergence thresholds through cross-validation.

        Args:
            references: The reference tensor (numpy array)
            num_cv: The number of cross-validations to run
            cv_size: The size of each individual cross-validation sample
            random_number_generator: Random seed or random generator object
            normalize: Whether to normalize values
        """
        try:
            # RANDOM NUMBER GENERATOR - for now accepts none, int or object with shuffle method
            if random_number_generator is None:
                rng = random.Random()
            elif isinstance(random_number_generator, int):
                rng = random.Random(random_number_generator)
            else:
                if not hasattr(random_number_generator, "shuffle"):
                    raise TypeError(
                        f"Random number generator wront type, got {type(random_number_generator)}, no shuffle method here!"
                    )
                rng = random_number_generator

            references = JensenShannonBaseline._validate_baseline_inputs(references, num_cv, cv_size)

            min_threshold = float("inf")
            max_threshold = float("-inf")
            avg_threshold = 0.0

            idxs = list(range(references.shape[0]))
            for i in range(num_cv):
                rng.shuffle(idxs)
                slice1 = references[idxs[:cv_size]]
                slice2 = references[idxs[cv_size : cv_size * 2]]
                try:
                    js_result = JensenShannon.calculate(slice1, slice2, 0.5, normalize)
                    js_stat = js_result.get_js_stat()

                    if js_stat < min_threshold:
                        min_threshold = js_stat
                    if js_stat > max_threshold:
                        max_threshold = js_stat

                    avg_threshold += js_stat

                except Exception as e:
                    raise ValueError(f"Error in cross-validation iteration {i + 1}: {str(e)}")

            if min_threshold > max_threshold:
                raise ValueError("[ve] - Calculation error: Minimum threshold is greater than maximum threshold")

            if not np.isfinite(min_threshold) or not np.isfinite(max_threshold):
                raise ValueError(
                    "[ve] - Calculation resulted in non-finite threshold values"
                )  # this could happen if nothing changes the values
            return JensenShannonBaseline(min_threshold, max_threshold, avg_threshold / num_cv)

        except (ValueError, TypeError) as e:
            raise type(e)(f"[ve] - Error in Jensen-Shannon baseline calculation: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"[ue] - Unexpected error in Jensen-Shannon baseline calculation: {str(e)}")

    @staticmethod
    def calculate_per_channel(
        references: np.ndarray,
        num_cv: int,
        cv_size: int,
        random_number_generator: Optional[Union[int, random.Random]] = None,
        normalize: bool = True,
    ) -> List["JensenShannonBaseline"]:
        """
        Calculate baseline per-channel JS divergence thresholds through cross-validation.

        Args:
            references: The reference tensor (numpy array)
            num_cv: The number of cross-validations to run
            cv_size: The size of each individual cross-validation sample
            random_number_generator: Random seed or random generator object
            normalize: Whether to normalize values

        Returns:
            A list of JensenShannonBaseline objects, one for each channel
        """
        try:
            # RANDOM NUMBER GENERATOR - for now accepts none, int or object with shuffle method
            if random_number_generator is None:
                rng = random.Random()
            elif isinstance(random_number_generator, int):
                rng = random.Random(random_number_generator)
                np.random.seed(random_number_generator)
            else:
                if not hasattr(random_number_generator, "shuffle"):
                    raise TypeError(
                        f"Random number generator wront type, got {type(random_number_generator)}, no shuffle method here!"
                    )
                rng = random_number_generator
                # np.random.seed() ??

            references = JensenShannonBaseline._validate_baseline_inputs(references, num_cv, cv_size)

            n_channels = references.shape[1]
            # same set up but for each item (channel)
            min_thresholds = np.full(n_channels, float("inf"))
            max_thresholds = np.full(n_channels, float("-inf"))
            avg_thresholds = np.zeros(n_channels)

            # threshold values - do not think they matter for baseline
            channel_thresholds = [0.5] * n_channels

            idxs = list(range(references.shape[0]))
            successful_iterations = 0

            for i in range(num_cv):
                try:
                    # explore difference in python vs JAVAs
                    rng.shuffle(idxs)
                    slice1_indices = idxs[:cv_size]
                    slice2_indices = idxs[cv_size : cv_size * 2]
                    slice1 = references[slice1_indices]
                    slice2 = references[slice2_indices]

                    js_results = JensenShannon.calculate_per_channel(slice1, slice2, channel_thresholds, normalize)

                    for channel in range(n_channels):
                        js_stat = js_results[channel].get_js_stat()

                        if js_stat < min_thresholds[channel]:
                            min_thresholds[channel] = js_stat
                        if js_stat > max_thresholds[channel]:
                            max_thresholds[channel] = js_stat

                        avg_thresholds[channel] += js_stat

                    successful_iterations += 1

                except Exception as e:
                    print(f"[e] - Error in cross-validation iteration {i + 1}: {str(e)}")
                    continue

            if successful_iterations == 0:
                raise ValueError("All cross-validation iterations failed")

            baselines = []
            for channel in range(n_channels):
                avg_thresholds[channel] /= successful_iterations

                # San check
                if min_thresholds[channel] > max_thresholds[channel]:
                    raise ValueError(
                        f"[ve] - Calculation error for channel {channel}: Minimum threshold is greater than maximum threshold"
                    )

                baselines.append(
                    JensenShannonBaseline(min_thresholds[channel], max_thresholds[channel], avg_thresholds[channel])
                )

            return baselines

        except (ValueError, TypeError) as e:
            raise type(e)(f"  : {str(e)}")
        except Exception as e:
            raise RuntimeError(f"[ue] - Unexpected error in per-channel Jensen-Shannon baseline calculation: {str(e)}")
