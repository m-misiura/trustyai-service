from typing import Dict, Any, Tuple, List, Optional
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