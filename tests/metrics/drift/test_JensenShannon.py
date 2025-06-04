import pytest
import numpy as np
import random
from typing import List
from src.core.metrics.drift.jensenshannon import JensenShannon, JensenShannonBaseline
# ▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄ TESTS ▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀


# ╭───────────────────────────────────────────────────────────────────────────╮
# │                             HELPER FUNCTIONS                              │
# ╰───────────────────────────────────────────────────────────────────────────╯
def generateImage(width: int, height: int) -> np.ndarray:
    """Creates a test image split into 4 colored quadrants and returns it as a numpy array"""
    img_array = np.zeros((1, 3, height, width), dtype=np.float64)
    img_array[0, 0, 0:height//2, 0:width//2] = 1.0                  # Red quadrant (top-left) - just R  
    img_array[0, 1, 0:height//2, width//2:width] = 1.0              # Green quadrant (top-right) - just G channel
    img_array[0, 2, height//2:height, 0:width//2] = 1.0             # Blue quadrant (bottom-left) - just B
    img_array[0, 0, height//2:height, width//2:width] = 1.0         # Yellow quadrant (bottom-left) - B 
    img_array[0, 1, height//2:height, width//2:width] = 1.0         # Yellow quadrant (bottom-left) - G
    return img_array

def get_3d_float_array(dimensions: List[int]) -> np.ndarray:
    """Generate a 3D array with random values"""
    return np.random.rand(*dimensions).astype(np.float64)

def get_4d_float_array(dimensions: List[int]) -> np.ndarray:
    """Generate a 4D array with random values"""
    return np.random.rand(*dimensions).astype(np.float64)

def get_3d_non_random_array(dimensions: List[int]) -> np.ndarray:
    """
    Generate a 3D array with sequential values, similar to the Java implementation.
    """
    a, b, c = dimensions
    total = a * b * c
    arr = np.arange(total, dtype=np.float64).reshape((a, b, c))
    return arr

def get_4d_non_random_array(dimensions: List[int]) -> np.ndarray:
    """
    Generate a 4D array with sequential values, similar to the Java implementation.
    """
    a, b, c, d = dimensions
    total = a * b * c * d
    arr = np.arange(total, dtype=np.float64).reshape((a, b, c, d))
    return arr

def get_4d_non_random_array_copyJava(dimensions: List[int]) -> np.ndarray:
        a, b, c, d = dimensions
        arr = np.zeros((a, b, c, d))
        idx = 0
        for i in range(a):
            for ii in range(b):
                for iii in range(c):
                    for iv in range(d):
                        arr[i, ii, iii, iv] = idx
                        idx += 1
        return arr


# ╭───────────────────────────────────────────────────────────────────────────╮
# │                      IDENTICAL COMPARISON TESTS                           │
# ╰───────────────────────────────────────────────────────────────────────────╯
def test_same_buffered_images() -> None:
    """
    Test that comparing identical images results in zero drift.
    .testSameBufferedImages
    """
    images_ref = generateImage(100, 100)
    result = JensenShannon.calculate(images_ref, images_ref, 0.5)
    
    # Test that JS statistic is 0 and reject is False for identical images
    assert result.get_js_stat() == 0.0
    assert result.is_reject() is False

def test_same_tensor_3d_images() -> None:
    """
    Test that comparing identical 3D array images results in zero drift.
    .testSameTensor3dImages
    """

    tensor_3d = get_3d_float_array([3, 256, 256])
    images_ref = np.expand_dims(tensor_3d, 0)   # Add the batch dimensions
    n_entries = np.prod(images_ref.shape)
    ref_normalized = images_ref / n_entries     # transform to probability distribution
    
    result = JensenShannon.calculate(ref_normalized, ref_normalized, 0.5)
    
    assert result.get_js_stat() == 0.0
    assert result.is_reject() is False

# # ╭───────────────────────────────────────────────────────────────────────────╮
# # │                       INPUT TYPE COMPATIBILITY                            │
# # ╰───────────────────────────────────────────────────────────────────────────╯
def test_diff_input_dtypes() -> None:
    """
    Test that comparing different input data types works correctly.
    """
    standard_list = [[[0.1, 0.2], [0.3, 0.4]], 
                   [[0.5, 0.6], [0.7, 0.8]], 
                   [[0.9, 1.0], [1.1, 1.2]]]
    np_array = np.array(standard_list)
    
    assert not isinstance(standard_list, np.ndarray)
    assert isinstance(np_array, np.ndarray)
    assert np.array_equal(np.array(standard_list), np_array)
    
    n_entries = np.prod(np_array.shape)
    standard_list_normalized = np.array([[[(val/n_entries) for val in row] for row in channel] for channel in standard_list])
    np_array_normalized = np_array / n_entries
    
    result = JensenShannon.calculate(standard_list_normalized, np_array_normalized, 0.5)
    
    assert result is not None
    # TODO: Check output in more detail

# # ╭───────────────────────────────────────────────────────────────────────────╮
# # │                       4D TENSOR CALCULATIONS                              │
# # ╰───────────────────────────────────────────────────────────────────────────╯
def test_4d_tensors() -> None:
    """
    Test with 4D arrays with the same shape but different values.
    """
    tensor_ref = get_4d_float_array([5, 6, 7, 8])
    tensor_hyp = get_4d_float_array([5, 6, 7, 8])
    n_entries = np.prod(tensor_ref.shape)
    ref_normalized = tensor_ref / n_entries
    hyp_normalized = tensor_hyp / n_entries
    
    same_result = JensenShannon.calculate(ref_normalized, ref_normalized, 0.5)
    
    assert same_result.get_js_stat() == 0.0
    assert same_result.is_reject() is False
    
    diff_result = JensenShannon.calculate(ref_normalized, hyp_normalized, 0.5)
    assert diff_result.get_js_stat() > 0.0

# # ╭───────────────────────────────────────────────────────────────────────────╮
# # │                       ERROR HANDLING TESTS                                │
# # ╰───────────────────────────────────────────────────────────────────────────╯
def test_shape_mismatch() -> None:
    """
    Test that comparing arrays with different shapes raises an exception.
    """
    tensor_ref = get_4d_float_array([5, 6, 7, 8])
    tensor_hyp = get_4d_float_array([5, 6, 7, 9])  # Different last dimension

    with pytest.raises(ValueError):
        JensenShannon.calculate(tensor_ref, tensor_hyp, 0.5)

def test_dimension_mismatch() -> None:
    """
    Test that comparing arrays with different dimensions raises an exception.
    """
    tensor_ref = get_3d_float_array([5, 6, 7])
    tensor_hyp = get_4d_float_array([5, 6, 7, 9])
    with pytest.raises(ValueError):
        JensenShannon.calculate(tensor_ref, tensor_hyp, 0.5)

# # ╭───────────────────────────────────────────────────────────────────────────╮
# # │                        BASELINE CALCULATION TESTS                         │
# # ╰───────────────────────────────────────────────────────────────────────────╯
def test_baseline() -> None:
    """
    Test baseline calculation for Jensen-Shannon drift detection.
    """
    tensor_ref = get_4d_non_random_array([64, 3, 32, 32])

    random.seed(50)
    local_rng = random.Random(0)

    n_entries = 64 * 3 * 32 * 32 
    tensor_ref = tensor_ref / n_entries 

    jsb = JensenShannonBaseline.calculate(tensor_ref, 500, 32, local_rng, False)

    assert jsb.get_avg_threshold() <= jsb.get_max_threshold()
    assert jsb.get_avg_threshold() >= jsb.get_min_threshold()
    assert 5050 < jsb.get_avg_threshold() < 5150

def test_baseline_normalized() -> None:
    """
    Test normalized baseline calculation for Jensen-Shannon drift detection.
    
    This is equivalent to testBaselineNormalized in Java
    """
    np.random.seed(0)
    tensor_ref = get_4d_non_random_array([64, 3, 32, 32])
    #tensor_ref = get_4d_float_array([64, 3, 32, 32])
    n_entries = np.prod(tensor_ref.shape)
    ref_normalized = tensor_ref / n_entries
    
    random.seed(0)
    local_rng = random.Random(0)
    jsb = JensenShannonBaseline.calculate(ref_normalized, 100, 32, local_rng, True)
    
    assert jsb.get_avg_threshold() <= jsb.get_max_threshold()
    assert jsb.get_avg_threshold() >= jsb.get_min_threshold()
    print(jsb.get_avg_threshold())
    assert 0.05 < jsb.get_avg_threshold() < 0.055

# def test_baseline_simple() -> None:
#     """
#     Test baseline calculation for Jensen-Shannon drift detection.
#     """
#     print("BASELINE TEST".center(60, '*'))
#     tensor_ref_raw = get_4d_non_random_array([64, 3, 32, 32])
#     print("[T] - Initial tensorRef created with dimensions [64, 3, 32, 32]\n")
#     random.seed(0)
#     local_rng = random.Random(0)

#     n_entries = tensor_ref_raw.size
#     print("[T] - Number of Entries: " + str(n_entries) + "\n")
#     tensor_ref = tensor_ref_raw / n_entries 

#     print("\n[T] - Normalised Tensor Ref: ")
#     print(tensor_ref)
#     print("\n")

#     jsb = JensenShannonBaseline.calculate(tensor_ref, 500, 32, local_rng, False)

#     print(f"\n FINAL JS RESULTS::\n\tMIN: {jsb.min_threshold}\n\tMAX: {jsb.max_threshold}\n\tAVG: {jsb.avg_threshold}\n")

#     assert jsb.get_avg_threshold() <= jsb.get_max_threshold()
#     assert jsb.get_avg_threshold() >= jsb.get_min_threshold()
#     assert 5000 < jsb.get_avg_threshold() < 5100
# ╭───────────────────────────────────────────────────────────────────────────╮
# │                      LARGE IMAGE CALCULATION TESTS                        │
# ╰───────────────────────────────────────────────────────────────────────────╯
def test_large_image() -> None:
    """
    Test Jensen-Shannon calculation with large images.
 
    """

    np.random.seed(0) 
    tensor_ref = get_4d_float_array([64, 3, 128, 128])
    tensor_hyp = get_4d_float_array([64, 3, 128, 128])
    n_entries = np.prod(tensor_ref.shape)
    ref_normalized = tensor_ref / n_entries
    hyp_normalized = tensor_hyp / n_entries

    random.seed(0)
    local_rng = random.Random(0)
    jsb = JensenShannonBaseline.calculate(ref_normalized, 10, 32, local_rng, True)
    jsdr = JensenShannon.calculate(ref_normalized, hyp_normalized, jsb.get_max_threshold() * 1.5, True)
    
    assert jsdr.is_reject() is False

def test_large_image_per_channel() -> None:
    """
    Test Jensen-Shannon calculation per channel with large images.
    
    This is equivalent to testLargeImagePerChannel in Java
    """
    np.random.seed(0)
    tensor_ref = get_4d_non_random_array([64, 3, 128, 128])
    tensor_hyp = get_4d_non_random_array([64, 3, 128, 128])
    n_entries = np.prod(tensor_ref.shape)
    ref_normalized = tensor_ref / n_entries
    hyp_normalized = tensor_hyp / n_entries

    random.seed(0)
    local_rng = random.Random(0)
    jensen_shannon_baselines = JensenShannonBaseline.calculate_per_channel(ref_normalized, 10, 32, local_rng, True)
    per_channel_thresholds = [j.get_max_threshold() for j in jensen_shannon_baselines]      # add safety margin ?
    jsdr = JensenShannon.calculate_per_channel(ref_normalized, hyp_normalized, per_channel_thresholds, True)
    
    for i in range(tensor_ref.shape[1]):
        assert jsdr[i].is_reject() is False

def test_js_divergence_hand_calculation() -> None:
    """
    Test the Jensen-Shannon divergence calculation using a hand-calculated example.
    
    Consider:
      p1 = [0.5, 0.5]
      p2 = [0.9, 0.1]
      
    Then:
      m = (p1 + p2) / 2 = [0.7, 0.3]
      KL(p1||m) ≈ 0.08718
      KL(p2||m) ≈ 0.11632
      JS divergence ≈ (0.08718 + 0.11632) / 2 ≈ 0.10175
    """
    p1 = np.array([0.5, 0.5], dtype=np.float64)
    p2 = np.array([0.9, 0.1], dtype=np.float64)
    expected_js = 0.10175
    calculated_js = JensenShannon.jensen_shannon_divergence(p1, p2)
    assert abs(calculated_js - expected_js) < 1e-5



    