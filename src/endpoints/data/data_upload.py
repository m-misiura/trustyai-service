from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Tuple, List, Optional
from src.service.data.storage import get_storage_interface
from src.service.constants import INPUT_SUFFIX, OUTPUT_SUFFIX, METADATA_SUFFIX

import logging
import numpy as np
import pickle
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)
storage_interface = get_storage_interface()

class ModelInferJointPayload(BaseModel):
    model_name: str
    data_tag: str = None
    is_ground_truth: bool = False
    request: Dict[str, Any]
    response: Dict[str, Any]

def get_numpy_dtype(kserve_dtype: str) -> np.dtype:
    """Map KServe datatype to numpy dtype."""
    dtype_map = {
        "BOOL": np.bool_,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "UINT32": np.uint32,
        "UINT64": np.uint64,
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "FP16": np.float16,
        "FP32": np.float32,
        "FP64": np.float64,
        "STRING": np.dtype('S64')
    }
    return dtype_map.get(kserve_dtype.upper(), np.float32)

async def parse_tensor(tensor: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    """Parse a single KServe tensor into a numpy array and name."""
    name = tensor.get("name", "unknown")
    shape = tensor.get("shape", [len(tensor.get("data", []))])
    datatype = tensor.get("datatype", "FP32")
    data = tensor.get("data", [])
    
    # Convert to numpy array
    np_dtype = get_numpy_dtype(datatype)
    array = np.array(data, dtype=np_dtype)
    
    # Reshape if needed
    if shape:
        try:
            array = array.reshape(shape)
        except ValueError as e:
            raise ValueError(f"Cannot reshape data to {shape}: {str(e)}")
    
    return array, name

async def combine_arrays(arrays: List[np.ndarray]) -> np.ndarray:
    """Combine arrays for storage, ensuring 2D format."""
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
    
async def create_metadata(sample_count: int, data_tag: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
    """Create metadata as a list of dictionaries."""
    metadata_list = []
    
    # Generate execution IDs
    for _ in range(sample_count):
        item = {"execution_id": str(uuid.uuid4())}
        if data_tag:
            item["data_tag"] = data_tag
        metadata_list.append(item)
    
    # Get all potential columns
    all_keys = set()
    for item in metadata_list:
        all_keys.update(item.keys())
    
    return metadata_list, list(all_keys)

async def read_metadata_safely(dataset_name: str) -> List[Dict]:
    """Read metadata with format detection and fallback."""
    try:
        # Read data from storage
        existing_data, existing_column_names = await storage_interface.read_data(dataset_name)
        
        # Add detailed logging to diagnose issues
        logger.info(f"Read metadata format: dtype={getattr(existing_data, 'dtype', 'unknown')}, "
                    f"type={type(existing_data)}, shape={getattr(existing_data, 'shape', 'unknown')}")
        
        # Check if it's our JSON-encoded metadata format
        if 'metadata' in existing_column_names:
            # It's in our JSON-encoded format
            existing_list = []
            for i in range(len(existing_data)):
                try:
                    # Try to decode the JSON string
                    if hasattr(existing_data[i]['metadata'], 'decode'):
                        json_str = existing_data[i]['metadata'].decode('utf-8').rstrip('\x00')
                        item = json.loads(json_str)
                        existing_list.append(item)
                except Exception as e:
                    logger.warning(f"Error decoding metadata item {i}: {e}")
            
            logger.info(f"Successfully read {len(existing_list)} metadata items in JSON format")
            return existing_list
            
        # If we got here, data is in an unexpected format
        logger.warning(f"Metadata is in an unexpected format, will reset and start fresh")
        return []
        
    except Exception as e:
        logger.warning(f"Error reading metadata: {e}, will reset and start fresh")
        return []
    
async def write_metadata_directly(dataset_name: str, metadata_list: List[Dict], column_names: List[str]) -> None:
    """Write metadata in a format compatible with the drift metrics code and HDF5 storage."""
    logger.info(f"Writing metadata with {len(metadata_list)} items")
    
    # Create a structured dtype with these columns - make it large enough for pickled data
    dtype_spec = [('metadata', 'S2000')]  # Increase buffer size for pickle protocol 0
    
    # Create a numpy structured array
    metadata_array = np.zeros(len(metadata_list), dtype=np.dtype(dtype_spec))
    
    # Store each dictionary using pickle protocol 0 (ASCII, no NULL bytes)
    for i, item in enumerate(metadata_list):
        # Clean any bytes to strings
        clean_item = {}
        for k, v in item.items():
            if isinstance(v, bytes):
                clean_item[k] = v.decode('utf-8')
            else:
                clean_item[k] = v
        
        # Use protocol 0 which is ASCII-only with no NULL bytes
        pickled_data = pickle.dumps(clean_item, protocol=0)
        metadata_array[i]['metadata'] = pickled_data
    
    # Write the array
    logger.info(f"Writing {len(metadata_list)} pickled metadata items (protocol 0)")
    await storage_interface.write_data(dataset_name, metadata_array, ["metadata"])

async def handle_numeric_dataset(dataset_name: str, new_data: np.ndarray, column_names: List[str]) -> None:
    """Handle appending to numeric datasets."""
    logger.info(f"Storing dataset {dataset_name} with shape {new_data.shape} and columns {column_names}")
    
    if not await storage_interface.dataset_exists(dataset_name):
        # Create new dataset
        logger.info(f"Creating new dataset: {dataset_name}")
        await storage_interface.write_data(dataset_name, new_data, column_names)
        # Verify data was written
        if await storage_interface.dataset_exists(dataset_name):
            logger.info(f"Successfully created dataset: {dataset_name}")
        else:
            logger.error(f"Failed to create dataset: {dataset_name}")
        return
    
    # Read existing data
    try:
        existing_data, existing_column_names = await storage_interface.read_data(dataset_name)
        logger.info(f"Read existing data from {dataset_name}: shape={existing_data.shape if hasattr(existing_data, 'shape') else 'unknown'}")
        
        # Verify column names match
        if set(existing_column_names) != set(column_names):
            logger.warning(f"Column mismatch in {dataset_name}: existing={existing_column_names}, new={column_names}")
            raise ValueError(f"Column mismatch: existing={existing_column_names}, new={column_names}")
        
        # Combine data vertically
        combined_data = np.vstack([existing_data, new_data])
        logger.info(f"Combined data shape: {combined_data.shape}")
        
        # Delete and rewrite
        await storage_interface.delete_dataset(dataset_name)
        await storage_interface.write_data(dataset_name, combined_data, column_names)
        
        # Verify rewrite was successful
        if await storage_interface.dataset_exists(dataset_name):
            logger.info(f"Successfully updated dataset: {dataset_name}")
        else:
            logger.error(f"Failed to update dataset: {dataset_name}")
    except Exception as e:
        logger.error(f"Error handling numeric dataset {dataset_name}: {str(e)}", exc_info=True)
        raise

@router.post("/upload")
async def upload_data(payload: ModelInferJointPayload):
    """Upload a batch of model data to TrustyAI."""
    try:
        logger.info(f"Received data upload for model: {payload.model_name}")
        
        # Validate input payload
        if not payload.request or "tensorPayloads" not in payload.request:
            raise HTTPException(status_code=400, detail="No input tensors provided")
        
        # Parse input tensors
        input_arrays = []
        input_names = []
        for tensor in payload.request.get("tensorPayloads", []):
            array, name = await parse_tensor(tensor)
            input_arrays.append(array)
            input_names.append(name)
        
        if not input_arrays:
            raise HTTPException(status_code=400, detail="No valid input tensors found")
        
        # Get sample count from first array
        sample_count = input_arrays[0].shape[0]
        
        # Verify all inputs have same sample count
        for i, arr in enumerate(input_arrays):
            if arr.shape[0] != sample_count:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Input tensor '{input_names[i]}' has {arr.shape[0]} samples, expected {sample_count}"
                )
        
        # Parse output tensors
        if not payload.response or "tensorPayloads" not in payload.response:
            raise HTTPException(status_code=400, detail="No output tensors provided")
        
        output_arrays = []
        output_names = []
        for tensor in payload.response.get("tensorPayloads", []):
            array, name = await parse_tensor(tensor)
            if array.shape[0] != sample_count:
                raise HTTPException(
                    status_code=400,
                    detail=f"Output tensor '{name}' has {array.shape[0]} samples, expected {sample_count}"
                )
            output_arrays.append(array)
            output_names.append(name)
        
        if not output_arrays:
            raise HTTPException(status_code=400, detail="No valid output tensors found")
        
        # Format data for storage
        combined_inputs = await combine_arrays(input_arrays)
        combined_outputs = await combine_arrays(output_arrays)
        
        # Create metadata
        metadata_list, _ = await create_metadata(sample_count, payload.data_tag)
        
        # Save data
        model_name = payload.model_name
        
        # Handle numeric data (inputs/outputs)
        await handle_numeric_dataset(model_name + INPUT_SUFFIX, combined_inputs, input_names)
        await handle_numeric_dataset(model_name + OUTPUT_SUFFIX, combined_outputs, output_names)
        
        # Handle metadata differently - always read, combine, and rewrite
        metadata_dataset = model_name + METADATA_SUFFIX
        if await storage_interface.dataset_exists(metadata_dataset):
            # Use our safer reader
            existing_list = await read_metadata_safely(metadata_dataset)
            
            # If we got any valid metadata, combine and write
            if existing_list:
                # Delete existing dataset
                await storage_interface.delete_dataset(metadata_dataset)
                
                # Combine metadata
                combined_metadata = existing_list + metadata_list
                
                # Get all keys
                all_keys = set()
                for item in combined_metadata:
                    all_keys.update(item.keys())
                
                # Write combined metadata
                logger.info(f"Writing combined metadata with {len(combined_metadata)} items")
                await write_metadata_directly(metadata_dataset, combined_metadata, list(all_keys))
            else:
                # Just write new metadata
                all_keys = set()
                for item in metadata_list:
                    all_keys.update(item.keys())
                
                logger.info(f"Writing fresh metadata with {len(metadata_list)} items")
                await write_metadata_directly(metadata_dataset, metadata_list, list(all_keys))
        else:
            # Just write new metadata
            all_keys = set()
            for item in metadata_list:
                all_keys.update(item.keys())
            
            logger.info(f"Writing initial metadata with {len(metadata_list)} items")
            await write_metadata_directly(metadata_dataset, metadata_list, list(all_keys))
        
        return f"{sample_count} datapoints successfully added to {model_name} data."
        
    except ValueError as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")


@router.get("/check_datasets/{model_name}")
async def check_datasets(model_name: str):
    """Check if all required datasets for a model exist and return their details."""
    result = {}
    
    # Check input dataset
    input_dataset = model_name + INPUT_SUFFIX
    if await storage_interface.dataset_exists(input_dataset):
        try:
            data, cols = await storage_interface.read_data(input_dataset)
            result["inputs"] = {
                "exists": True,
                "shape": data.shape if hasattr(data, 'shape') else "unknown",
                "columns": cols,
                "sample": str(data[0]) if len(data) > 0 else "empty"
            }
        except Exception as e:
            result["inputs"] = {"exists": True, "error": str(e)}
    else:
        result["inputs"] = {"exists": False}
    
    # Check output dataset
    output_dataset = model_name + OUTPUT_SUFFIX
    if await storage_interface.dataset_exists(output_dataset):
        try:
            data, cols = await storage_interface.read_data(output_dataset)
            result["outputs"] = {
                "exists": True,
                "shape": data.shape if hasattr(data, 'shape') else "unknown",
                "columns": cols,
                "sample": str(data[0]) if len(data) > 0 else "empty"
            }
        except Exception as e:
            result["outputs"] = {"exists": True, "error": str(e)}
    else:
        result["outputs"] = {"exists": False}
    
    # Check metadata dataset
    metadata_dataset = model_name + METADATA_SUFFIX
    if await storage_interface.dataset_exists(metadata_dataset):
        try:
            data, cols = await storage_interface.read_data(metadata_dataset)
            result["metadata"] = {
                "exists": True,
                "shape": data.shape if hasattr(data, 'shape') else "unknown",
                "columns": cols,
                "sample": str(data[0]) if len(data) > 0 else "empty"
            }
        except Exception as e:
            result["metadata"] = {"exists": True, "error": str(e)}
    else:
        result["metadata"] = {"exists": False}
    
    return result