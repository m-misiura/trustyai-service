from typing import Dict, Any, Tuple, List, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TensorParser:
    """Handles parsing and processing of KServe tensor formats"""
    
    @staticmethod
    def get_numpy_dtype(kserve_dtype: str) -> np.dtype:
        """Map KServe datatype to numpy dtype."""
        dtype_map = {
            "BOOL": np.bool_,
            "UINT8": np.uint8, "UINT16": np.uint16, "UINT32": np.uint32, "UINT64": np.uint64,
            "INT8": np.int8, "INT16": np.int16, "INT32": np.int32, "INT64": np.int64,
            "FP16": np.float16, "FP32": np.float32, "FP64": np.float64,
            "STRING": np.dtype('S64')
        }
        return dtype_map.get(kserve_dtype.upper(), np.float32)
    
    @classmethod
    async def parse_tensor(cls, tensor):
        """Parse a single KServe tensor into a numpy array and name."""
        # Handle both dictionary-style access and Pydantic model access
        if hasattr(tensor, "get"):
            # Dictionary-style access
            name = tensor.get("name", "unknown")
            shape = tensor.get("shape", [len(tensor.get("data", []))])
            datatype = tensor.get("datatype", "FP32")
            data = tensor.get("data", [])
        else:
            # Pydantic model access
            name = tensor.name
            shape = tensor.shape if tensor.shape else [len(tensor.data)]
            datatype = tensor.datatype
            data = tensor.data
        
        # Input validation
        if not data:
            raise ValueError(f"Tensor '{name}' contains no data")
            
        # Convert to numpy array
        try:
            np_dtype = cls.get_numpy_dtype(datatype)
            array = np.array(data, dtype=np_dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert tensor '{name}' to {datatype}: {str(e)}")
        
        # Reshape if needed
        if shape:
            try:
                array = array.reshape(shape)
            except ValueError as e:
                raise ValueError(f"Cannot reshape tensor '{name}' to {shape}: {str(e)}")
        
        return array, name
    
    @staticmethod
    async def combine_arrays(arrays: List[np.ndarray]) -> np.ndarray:
        """Combine arrays for storage, ensuring 2D format."""
        if not arrays:
            raise ValueError("No arrays to combine")
            
        # Normalize dimensions to ensure 2D arrays (samples × features)
        formatted = []
        for array in arrays:
            if len(array.shape) == 1:
                # Convert 1D to 2D
                formatted.append(array.reshape(-1, 1))
            elif len(array.shape) > 2:
                # Flatten dimensions after the first
                formatted.append(array.reshape(array.shape[0], -1))
            else:
                # Already 2D
                formatted.append(array)
        
        # Combine horizontally (concatenate features)
        if len(formatted) == 1:
            return formatted[0]
        
        try:
            return np.hstack(formatted)
        except ValueError as e:
            raise ValueError(f"Failed to combine arrays: {str(e)}")
    
    @classmethod
    def process_multiple_tensors(cls, tensors, contains_non_numeric_fn=None):
        """Process multiple tensors, similar to the multi-tensor case in consumer_endpoint.py."""
        data = []
        shapes = set()
        shape_tuples = []
        column_names = []
        
        for tensor in tensors:
            data.append(tensor.data)
            shapes.add(tuple(tensor.shape))
            column_names.append(tensor.name)
            shape_tuples.append((tensor.name, tensor.shape))
        
        if len(shapes) == 1:
            if contains_non_numeric_fn and contains_non_numeric_fn(data):
                return np.array(data, dtype="O").T, column_names, shape_tuples
            else:
                return np.array(data).T, column_names, shape_tuples
        
        # Return shape_tuples to allow caller to handle mismatched shapes
        return None, None, shape_tuples
    
    @classmethod
    def process_single_tensor(cls, tensor, contains_non_numeric_fn=None):
        """Process a single tensor, similar to the single tensor case in consumer_endpoint.py."""
        column_names = []
        
        if len(tensor.shape) > 1:
            column_names = [f"{tensor.name}-{i}" for i in range(tensor.shape[1])]
        else:
            column_names = [tensor.name]
        
        if contains_non_numeric_fn and contains_non_numeric_fn(tensor.data):
            return np.array(tensor.data, dtype="O"), column_names
        else:
            return np.array(tensor.data), column_names
    
    @classmethod
    def process_payload(cls, payload, get_data_fn: Callable, enforced_first_shape: int = None, contains_non_numeric_fn=None):
        """Process payload in a way that matches consumer_endpoint.py's process_payload function."""
        tensors = get_data_fn(payload)
        
        if len(tensors) > 1:  # multi tensor case
            array, column_names, shape_tuples = cls.process_multiple_tensors(tensors, contains_non_numeric_fn)
            
            if array is None:  # Shapes mismatch, caller should handle error
                return None, None, shape_tuples
            
            row_count = array.shape[0]
            if enforced_first_shape is not None and row_count != enforced_first_shape:
                # Return information for caller to handle row count mismatch
                return None, None, None, enforced_first_shape, row_count
            
            return array, column_names, None
        
        else:  # single tensor case
            tensor = tensors[0]
            if enforced_first_shape is not None and tensor.shape[0] != enforced_first_shape:
                # Return information for caller to handle row count mismatch
                return None, None, None, enforced_first_shape, tensor.shape[0]
            
            return cls.process_single_tensor(tensor, contains_non_numeric_fn)