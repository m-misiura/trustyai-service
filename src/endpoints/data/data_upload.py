from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator, Field
from typing import Dict, Any, Tuple, List, Optional, Union, Literal, Callable
import logging
import numpy as np
import uuid
import pickle
import json
import re
import os
import h5py
import time
from datetime import datetime

from src.service.data.storage import get_storage_interface
from src.service.constants import INPUT_SUFFIX, OUTPUT_SUFFIX, METADATA_SUFFIX
from src.service.data.parsers import TensorParser
from src.service.utils import list_utils

# Import shared utilities to avoid duplication
from src.endpoints.consumer.consumer_endpoint import (
    KServeData, KServeInferenceRequest, KServeInferenceResponse,
    reconcile_mismatching_shape_error, reconcile_mismatching_row_count_error,
    process_payload
)

router = APIRouter()
logger = logging.getLogger(__name__)
storage_interface = get_storage_interface()

class TensorPayload(BaseModel):
    name: str
    data: List[Any]
    datatype: str = "FP32"
    shape: Optional[List[int]] = None
    executionIDs: Optional[List[str]] = None

class RequestResponse(BaseModel):
    tensorPayloads: List[TensorPayload]

class ModelInferJointPayload(BaseModel):
    model_name: str
    data_tag: Optional[str] = None
    is_ground_truth: bool = False
    request: RequestResponse
    response: RequestResponse
    
    @validator('model_name')
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        if "/" in v or ".." in v or "\\" in v:
            raise ValueError("Model name contains invalid characters")
        return v
        
    @validator('data_tag')
    def validate_data_tag(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError("Data tag cannot be empty if provided")
            # Add any specific validation rules for data tags
        return v

class MetadataManager:
    """Handles creation and management of metadata"""
    
    @staticmethod
    async def create_metadata(sample_count: int, data_tag: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
        """Create metadata as a list of dictionaries with timestamp information."""
        metadata_list = []
        
        # Get current timestamp information (from consumer endpoint)
        iso_time = datetime.isoformat(datetime.utcnow())
        unix_timestamp = time.time()
        
        # Generate execution IDs
        for _ in range(sample_count):
            item = {
                "execution_id": str(uuid.uuid4()),
                "iso_time": iso_time,
                "unix_timestamp": unix_timestamp
            }
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
        """Read metadata directly from HDF5 file with specialized handling for the TrustyAI format."""
        try:
            storage_interface_instance = get_storage_interface()
            data_directory = getattr(storage_interface_instance, "data_directory", "/tmp/trustyai-data")
            
            # Get the direct path to the HDF5 file
            file_path = os.path.join(data_directory, f"{dataset_name}_trustyai_data")
            logger.info(f"Attempting to read metadata directly from: {file_path}")
            
            # Check if the file exists
            if not os.path.exists(file_path):
                logger.warning(f"Metadata file does not exist: {file_path}")
                return []
            
            metadata_list = []
            
            # Open the file directly with h5py
            try:
                with h5py.File(file_path, 'r') as f:
                    # Check if the dataset exists
                    if dataset_name in f:
                        # Get the dataset
                        dataset = f[dataset_name]
                        
                        # Check if there's metadata
                        if 'metadata' in dataset.dtype.names:
                            logger.info(f"Found metadata column in dataset {dataset_name}")
                            
                            # Process each row
                            for i in range(len(dataset)):
                                try:
                                    # Get the metadata bytes
                                    metadata_bytes = dataset[i]['metadata']
                                    
                                    # First try pickle deserialization
                                    try:
                                        # Clean the data and try pickle
                                        clean_bytes = metadata_bytes.rstrip(b'\x00')
                                        metadata_dict = pickle.loads(clean_bytes, encoding='latin1')
                                        metadata_list.append(metadata_dict)
                                        continue
                                    except Exception as pickle_error:
                                        # If pickle fails, try manual extraction with regex
                                        logger.debug(f"Pickle deserialization failed for item {i}: {pickle_error}")
                                    
                                    # Convert bytes to string for regex processing
                                    metadata_str = metadata_bytes.decode('latin1', errors='replace')
                                    
                                    # Extract with regex patterns
                                    metadata_dict = {}
                                    
                                    # Extract execution_id
                                    execution_id_match = re.search(r'execution_id.*?\'([^\']+)\'', metadata_str)
                                    if execution_id_match:
                                        metadata_dict['execution_id'] = execution_id_match.group(1)
                                    
                                    # Extract data_tag
                                    data_tag_match = re.search(r'data_tag.*?\'([^\']+)\'', metadata_str)
                                    if data_tag_match:
                                        metadata_dict['data_tag'] = data_tag_match.group(1)
                                    
                                    if metadata_dict:
                                        metadata_list.append(metadata_dict)
                                except Exception as row_error:
                                    logger.warning(f"Error decoding metadata row {i}: {row_error}")
            except Exception as file_error:
                logger.error(f"Error opening HDF5 file {file_path}: {file_error}")
                
            logger.info(f"Successfully read {len(metadata_list)} metadata items")
            return metadata_list
        except Exception as e:
            logger.error(f"Error reading metadata: {e}", exc_info=True)
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

class GroundTruthHandler:
    """Handles validation and processing of ground truth data"""
    
    @staticmethod
    async def compare_features(original_features, uploaded_features):
        """Compare original features with uploaded features and generate detailed mismatch report"""
        if len(original_features) != len(uploaded_features):
            return False, f"Feature count mismatch: original={len(original_features)}, uploaded={len(uploaded_features)}"
        
        mismatches = []
        for i, (orig, uploaded) in enumerate(zip(original_features, uploaded_features)):
            if not np.array_equal(orig, uploaded):
                mismatches.append(f"Feature {i}: original={orig}, uploaded={uploaded}")
        
        if mismatches:
            mismatch_report = "\n".join(mismatches)
            return False, f"Feature value mismatches:\n{mismatch_report}"
        
        return True, ""
    
    @staticmethod
    async def compare_outputs(original_outputs, uploaded_outputs):
        """Compare original outputs with uploaded outputs and generate detailed mismatch report"""
        if len(original_outputs) != len(uploaded_outputs):
            return False, f"Output count mismatch: original={len(original_outputs)}, uploaded={len(uploaded_outputs)}"
        
        mismatches = []
        for i, (orig, uploaded) in enumerate(zip(original_outputs, uploaded_outputs)):
            if not np.array_equal(orig, uploaded):
                mismatches.append(f"Output {i}: original={orig}, uploaded={uploaded}")
        
        if mismatches:
            mismatch_report = "\n".join(mismatches)
            return False, f"Output value mismatches:\n{mismatch_report}"
        
        return True, ""
    
    @staticmethod
    async def handle_ground_truths(payload, input_arrays, input_names, output_arrays, output_names, execution_ids):
        """Process ground truth data with validation against existing data"""
        model_name = payload.model_name
        
        if not execution_ids:
            raise ValueError("No execution IDs provided. When uploading ground truths, all inputs must have a corresponding execution ID.")
        
        # Check if model data exists
        input_dataset = model_name + INPUT_SUFFIX
        output_dataset = model_name + OUTPUT_SUFFIX
        metadata_dataset = model_name + METADATA_SUFFIX
        
        if not await storage_interface.dataset_exists(input_dataset):
            raise ValueError(f"No TrustyAI data named {model_name} found. Ground truths can only be uploaded for existing data.")
        
        # Read existing data
        existing_inputs, existing_input_names = await storage_interface.read_data(input_dataset)
        existing_outputs, existing_output_names = await storage_interface.read_data(output_dataset)
        existing_metadata = await MetadataManager.read_metadata_safely(metadata_dataset)
        
        # Create mapping of execution_id to row index
        id_to_idx = {}
        for i, metadata in enumerate(existing_metadata):
            if "execution_id" in metadata:
                id_to_idx[metadata["execution_id"]] = i
        
        # Check each provided ground truth against existing data
        ground_truth_metadata = []
        ground_truth_outputs = []
        row_mismatch_errors = []
        
        for i, exec_id in enumerate(execution_ids):
            if exec_id not in id_to_idx:
                row_mismatch_errors.append(f"Execution ID {exec_id} not found in existing data")
                continue
                
            idx = id_to_idx[exec_id]
            
            # Extract the row from existing inputs
            existing_row = existing_inputs[idx]
            
            # Extract the input row we're validating
            current_row = input_arrays[0][i] if len(input_arrays) == 1 else np.hstack([arr[i:i+1] for arr in input_arrays])
            
            # Compare input features
            inputs_match, input_error = await GroundTruthHandler.compare_features(existing_row, current_row)
            if not inputs_match:
                row_mismatch_errors.append(f"ID={exec_id}: {input_error}")
                continue
            
            # Outputs match, so we can add this ground truth
            ground_truth_metadata.append({
                "execution_id": exec_id,
                "is_ground_truth": True,
                "data_tag": payload.data_tag if payload.data_tag else "ground_truth"
            })
            
            # Extract the output row we're saving
            output_row = output_arrays[0][i] if len(output_arrays) == 1 else np.hstack([arr[i:i+1] for arr in output_arrays])
            ground_truth_outputs.append(output_row)
        
        # If any mismatches, raise error
        if row_mismatch_errors:
            mismatch_report = "\n".join(row_mismatch_errors)
            raise ValueError(f"Found mismatches between uploaded data and recorded data:\n{mismatch_report}")
        
        if not ground_truth_metadata:
            raise ValueError("No valid ground truths found in uploaded data")
        
        # Convert to numpy array for storage
        ground_truth_outputs_array = np.vstack(ground_truth_outputs)
        
        # Define a ground truth dataset name
        ground_truth_dataset = model_name + "_ground_truths"
        ground_truth_metadata_dataset = ground_truth_dataset + METADATA_SUFFIX
        
        # Save ground truth data
        if await storage_interface.dataset_exists(ground_truth_dataset):
            # Read existing ground truths
            existing_ground_truths, existing_ground_truth_names = await storage_interface.read_data(ground_truth_dataset)
            
            # Validate column names match
            if set(existing_ground_truth_names) != set(output_names):
                raise ValueError(f"Ground truth output names mismatch: existing={existing_ground_truth_names}, new={output_names}")
            
            # Combine with existing ground truths
            combined_ground_truths = np.vstack([existing_ground_truths, ground_truth_outputs_array])
            
            # Save combined ground truths
            await DataStorageHandler.handle_dataset(ground_truth_dataset, combined_ground_truths, output_names)
            
            # Handle metadata
            await DataStorageHandler.handle_metadata(ground_truth_dataset, ground_truth_metadata)
        else:
            # Create new ground truth dataset
            await DataStorageHandler.handle_dataset(ground_truth_dataset, ground_truth_outputs_array, output_names)
            
            # Handle metadata
            await DataStorageHandler.handle_metadata(ground_truth_dataset, ground_truth_metadata)
        
        return f"{len(ground_truth_metadata)} ground truths successfully added to {ground_truth_dataset}."


class DataUploadService:
    """Main service for handling data uploads"""
    
    @staticmethod
    async def process_upload(payload: ModelInferJointPayload) -> str:
        """Process a data upload request and return a status message."""
        logger.info(f"Processing data upload for model: {payload.model_name}")
        
        # Validate input payload
        if not payload.request or not payload.request.tensorPayloads:
            raise ValueError("No input tensors provided")
        
        # Parse input tensors
        input_arrays = []
        input_names = []
        execution_ids = None
        
        for tensor in payload.request.tensorPayloads:
            array, name = await TensorParser.parse_tensor(tensor)
            input_arrays.append(array)
            input_names.append(name)
            
            # Extract execution IDs from the first tensor if available
            if tensor.executionIDs and execution_ids is None:
                execution_ids = tensor.executionIDs
                logger.info(f"Found {len(execution_ids)} execution IDs in tensor '{name}'")
        
        if not input_arrays:
            raise ValueError("No valid input tensors found")
        
        # Get sample count from first array
        sample_count = input_arrays[0].shape[0]
        
        # Verify all inputs have same sample count
        for i, arr in enumerate(input_arrays):
            if arr.shape[0] != sample_count:
                raise ValueError(f"Input tensor '{input_names[i]}' has {arr.shape[0]} samples, expected {sample_count}")
        
        # Check execution IDs match sample count
        if execution_ids and len(execution_ids) != sample_count:
            raise ValueError(f"Mismatch between number of samples ({sample_count}) and execution IDs ({len(execution_ids)})")
        
        # Parse output tensors
        if not payload.response or not payload.response.tensorPayloads:
            raise ValueError("No output tensors provided")
        
        output_arrays = []
        output_names = []
        for tensor in payload.response.tensorPayloads:
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
        
        # Handle ground truth vs. regular data upload
        if payload.is_ground_truth:
            if not execution_ids:
                raise ValueError("Ground truth uploads require execution IDs to match with existing data")
                
            # Process and validate ground truths
            return await GroundTruthHandler.handle_ground_truths(
                payload,
                input_arrays, 
                input_names, 
                output_arrays, 
                output_names, 
                execution_ids
            )
        else:
            # Regular data upload process
            if execution_ids:
                logger.info("Execution IDs were provided but this is not a ground truth upload - IDs will be used for metadata")
            
            # Create metadata with proper handling of execution IDs
            if execution_ids:
                metadata_list = []
                for i, exec_id in enumerate(execution_ids):
                    item = {"execution_id": exec_id}
                    if payload.data_tag:
                        item["data_tag"] = payload.data_tag
                    metadata_list.append(item)
            else:
                # Generate new execution IDs
                metadata_list, _ = await MetadataManager.create_metadata(sample_count, payload.data_tag)
            
            # Save data
            model_name = payload.model_name
            
            # Handle data tagging validation
            if payload.data_tag:
                # Add validation for data tag if needed
                logger.info(f"Data will be tagged with '{payload.data_tag}'")
            
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
        # Validate request structure
        if not payload.request or not payload.request.tensorPayloads or len(payload.request.tensorPayloads) < 1:
            raise ValueError("Directly uploaded datapoints must specify at least one input tensor.")
            
        # Validate model name
        if not payload.model_name or not payload.model_name.strip():
            raise ValueError("Model name cannot be empty")
            
        # Validate ground truth requirements
        if payload.is_ground_truth:
            # Check for execution IDs
            has_execution_ids = False
            for tensor in payload.request.tensorPayloads:
                if tensor.executionIDs and len(tensor.executionIDs) > 0:
                    has_execution_ids = True
                    break
                    
            if not has_execution_ids:
                raise ValueError("No execution IDs were provided. When uploading ground truths, all inputs must have a corresponding "
                                "TrustyAI Execution ID to correlate them with existing inferences.")
                
            # Check if the model exists
            if not await storage_interface.dataset_exists(payload.model_name + INPUT_SUFFIX):
                raise ValueError(f"No TrustyAI dataframe named {payload.model_name}. Ground truths can only be uploaded for extant dataframes.")
                
        # Process the upload
        result = await DataUploadService.process_upload(payload)
        return {"message": result}
        
    except ValueError as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")