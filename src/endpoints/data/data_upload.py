from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from typing import Dict, Any, Tuple, List, Optional, Union
import logging
import numpy as np
import uuid
import pickle
import json

from src.service.data.storage import get_storage_interface
from src.service.constants import INPUT_SUFFIX, OUTPUT_SUFFIX, METADATA_SUFFIX

router = APIRouter()
logger = logging.getLogger(__name__)
storage_interface = get_storage_interface()

class ModelInferJointPayload(BaseModel):
    model_name: str
    data_tag: Optional[str] = None
    is_ground_truth: bool = False
    request: Dict[str, Any]
    response: Dict[str, Any]
    
    @validator('model_name')
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        if "/" in v or ".." in v or "\\" in v:
            raise ValueError("Model name contains invalid characters")
        return v

# --- Tensor Processing ---

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
    async def parse_tensor(cls, tensor: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """Parse a single KServe tensor into a numpy array and name."""
        name = tensor.get("name", "unknown")
        shape = tensor.get("shape", [len(tensor.get("data", []))])
        datatype = tensor.get("datatype", "FP32")
        data = tensor.get("data", [])
        
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

# --- Metadata Management ---

class MetadataManager:
    """Handles creation and management of metadata"""
    
    @staticmethod
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
    
    @staticmethod
    async def read_metadata_safely(dataset_name: str) -> List[Dict]:
        """Read metadata with format detection and fallback."""
        try:
            # Read data from storage
            existing_data, existing_column_names = await storage_interface.read_data(dataset_name)
            
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
    
    @staticmethod
    async def write_metadata(dataset_name: str, metadata_list: List[Dict], column_names: List[str]) -> None:
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

# --- Data Storage Handling ---

class DataStorageHandler:
    """Handles storage operations for model data"""
    
    @staticmethod
    async def handle_dataset(dataset_name: str, new_data: np.ndarray, column_names: List[str]) -> None:
        """Handle writing to datasets with proper error handling."""
        logger.info(f"Storing dataset {dataset_name} with shape {new_data.shape} and columns {column_names}")
        
        try:
            if not await storage_interface.dataset_exists(dataset_name):
                # Create new dataset
                logger.info(f"Creating new dataset: {dataset_name}")
                await storage_interface.write_data(dataset_name, new_data, column_names)
                logger.info(f"Successfully created dataset: {dataset_name}")
                return
            
            # Read existing data
            existing_data, existing_column_names = await storage_interface.read_data(dataset_name)
            logger.info(f"Read existing data from {dataset_name}: shape={existing_data.shape if hasattr(existing_data, 'shape') else 'unknown'}")
            
            # Verify column names match
            if set(existing_column_names) != set(column_names):
                logger.warning(f"Column mismatch in {dataset_name}: existing={existing_column_names}, new={column_names}")
                raise ValueError(f"Column mismatch: existing={existing_column_names}, new={column_names}")
            
            # Combine data vertically
            combined_data = np.vstack([existing_data, new_data])
            logger.info(f"Combined data shape: {combined_data.shape}")
            
            # Safer approach: write to new dataset first, then delete old
            temp_dataset = dataset_name + "_temp"
            
            # Write combined data to temp
            await storage_interface.write_data(temp_dataset, combined_data, column_names)
            
            # Delete old dataset
            await storage_interface.delete_dataset(dataset_name)
            
            # Write to final destination 
            await storage_interface.write_data(dataset_name, combined_data, column_names)
            
            # Clean up temp
            await storage_interface.delete_dataset(temp_dataset)
            
            logger.info(f"Successfully updated dataset: {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error handling dataset {dataset_name}: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    async def handle_metadata(model_name: str, metadata_list: List[Dict]) -> None:
        """Handle storage of metadata with all edge cases."""
        metadata_dataset = model_name + METADATA_SUFFIX
        
        try:
            if await storage_interface.dataset_exists(metadata_dataset):
                # Read existing metadata safely
                existing_list = await MetadataManager.read_metadata_safely(metadata_dataset)
                
                # If we got valid metadata, combine and write
                if existing_list:
                    # Combine metadata
                    combined_metadata = existing_list + metadata_list
                    
                    # Get all keys
                    all_keys = set()
                    for item in combined_metadata:
                        all_keys.update(item.keys())
                    
                    # Write to temp first
                    temp_dataset = metadata_dataset + "_temp"
                    await MetadataManager.write_metadata(temp_dataset, combined_metadata, list(all_keys))
                    
                    # Delete original
                    await storage_interface.delete_dataset(metadata_dataset)
                    
                    # Write final
                    await MetadataManager.write_metadata(metadata_dataset, combined_metadata, list(all_keys))
                    
                    # Clean up temp
                    await storage_interface.delete_dataset(temp_dataset)
                else:
                    # Just write new metadata
                    all_keys = set()
                    for item in metadata_list:
                        all_keys.update(item.keys())
                    
                    await MetadataManager.write_metadata(metadata_dataset, metadata_list, list(all_keys))
            else:
                # Just write new metadata
                all_keys = set()
                for item in metadata_list:
                    all_keys.update(item.keys())
                
                await MetadataManager.write_metadata(metadata_dataset, metadata_list, list(all_keys))
                
        except Exception as e:
            logger.error(f"Error handling metadata for {model_name}: {str(e)}", exc_info=True)
            raise

# --- Main Service Class ---

class DataUploadService:
    """Main service for handling data uploads"""
    
    @staticmethod
    async def process_upload(payload: ModelInferJointPayload) -> str:
        """Process a data upload request and return a status message."""
        logger.info(f"Processing data upload for model: {payload.model_name}")
        
        # Validate input payload
        if not payload.request or "tensorPayloads" not in payload.request:
            raise ValueError("No input tensors provided")
        
        # Parse input tensors
        input_arrays = []
        input_names = []
        for tensor in payload.request.get("tensorPayloads", []):
            array, name = await TensorParser.parse_tensor(tensor)
            input_arrays.append(array)
            input_names.append(name)
        
        if not input_arrays:
            raise ValueError("No valid input tensors found")
        
        # Get sample count from first array
        sample_count = input_arrays[0].shape[0]
        
        # Verify all inputs have same sample count
        for i, arr in enumerate(input_arrays):
            if arr.shape[0] != sample_count:
                raise ValueError(f"Input tensor '{input_names[i]}' has {arr.shape[0]} samples, expected {sample_count}")
        
        # Parse output tensors
        if not payload.response or "tensorPayloads" not in payload.response:
            raise ValueError("No output tensors provided")
        
        output_arrays = []
        output_names = []
        for tensor in payload.response.get("tensorPayloads", []):
            array, name = await TensorParser.parse_tensor(tensor)
            if array.shape[0] != sample_count:
                raise ValueError(f"Output tensor '{name}' has {array.shape[0]} samples, expected {sample_count}")
            output_arrays.append(array)
            output_names.append(name)
        
        if not output_arrays:
            raise ValueError("No valid output tensors found")
        
        # Format data for storage
        combined_inputs = await TensorParser.combine_arrays(input_arrays)
        combined_outputs = await TensorParser.combine_arrays(output_arrays)
        
        # Create metadata
        metadata_list, _ = await MetadataManager.create_metadata(sample_count, payload.data_tag)
        
        # Save data
        model_name = payload.model_name
        
        # Handle numeric data (inputs/outputs)
        await DataStorageHandler.handle_dataset(model_name + INPUT_SUFFIX, combined_inputs, input_names)
        await DataStorageHandler.handle_dataset(model_name + OUTPUT_SUFFIX, combined_outputs, output_names)
        
        # Handle metadata
        await DataStorageHandler.handle_metadata(model_name, metadata_list)
        
        return f"{sample_count} datapoints successfully added to {model_name} data."

# --- API Endpoints ---

@router.post("/upload")
async def upload_data(payload: ModelInferJointPayload):
    """Upload a batch of model data to TrustyAI."""
    try:
        result = await DataUploadService.process_upload(payload)
        return result
    except ValueError as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")