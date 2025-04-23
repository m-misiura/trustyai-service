from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from typing import List, Any, Optional, Union, Dict, Tuple, Callable
from enum import Enum
import logging
import numpy as np
import io
import csv
from datetime import datetime
import re

from src.service.data.storage import get_storage_interface
from src.service.constants import INPUT_SUFFIX, OUTPUT_SUFFIX, METADATA_SUFFIX

router = APIRouter()
logger = logging.getLogger(__name__)
storage_interface = get_storage_interface()

# Constants
TRUSTY_PREFIX = "trustyai."

class MatchOperation(str, Enum):
    EQUALS = "EQUALS"
    BETWEEN = "BETWEEN"

class DataType(str, Enum):
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    CATEGORICAL = "CATEGORICAL"
    DATE = "DATE"
    UNKNOWN = "UNKNOWN"

class RowMatcher(BaseModel):
    columnName: str
    operation: str
    values: List[Any]
    
    @validator('operation')
    def validate_operation(cls, v):
        try:
            return MatchOperation(v)
        except ValueError:
            raise ValueError(f"Invalid operation: {v}. Must be one of {[op.value for op in MatchOperation]}")
        
    @validator('values')
    def validate_values(cls, v, values):
        operation = values.get('operation')
        if operation == MatchOperation.EQUALS and len(v) != 1:
            raise ValueError(f"EQUALS operation requires exactly 1 value, got {len(v)}")
        if operation == MatchOperation.BETWEEN and len(v) != 2:
            raise ValueError(f"BETWEEN operation requires exactly 2 values, got {len(v)}")
        return v

class DataRequestPayload(BaseModel):
    modelId: str
    matchAny: List[RowMatcher] = []
    matchAll: List[RowMatcher] = []
    matchNone: List[RowMatcher] = []

class DataResponsePayload(BaseModel):
    dataCSV: str

# Utility functions for data types and conversions
class DataTypeUtils:
    """Utilities for data type detection and conversion"""
    
    @staticmethod
    def get_data_type(value: Any) -> DataType:
        """Determine the data type for a value"""
        if isinstance(value, (int, float, np.number)):
            return DataType.NUMBER
        elif isinstance(value, (bool, np.bool_)):
            return DataType.BOOLEAN
        elif isinstance(value, str):
            # Check if it looks like a date (simple heuristic)
            date_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})')
            if date_pattern.match(value):
                return DataType.DATE
            return DataType.STRING
        return DataType.UNKNOWN
    
    @staticmethod
    def parse_date(date_str: str) -> float:
        """Parse date string to timestamp for comparison"""
        try:
            # Try different date formats
            for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y-%m-%d %H:%M:%S'):
                try:
                    return datetime.strptime(date_str, fmt).timestamp()
                except ValueError:
                    continue
            # If all formats fail, raise ValueError
            raise ValueError(f"Could not parse date: {date_str}")
        except Exception as e:
            logger.warning(f"Date parsing error: {e}, treating as string")
            return 0.0

# Utility functions for download operations
class DownloadUtils:
    """Utility functions for data download operations"""
    
    @staticmethod
    def get_data_type(metadata: Dict, row_matcher: RowMatcher) -> DataType:
        """Determine the data type for a column based on metadata and row matcher"""
        # Try to infer from the first value
        return DataTypeUtils.get_data_type(row_matcher.values[0])
    
    @staticmethod
    def equals_matcher_numpy(data: np.ndarray, column_index: int, value: Any,
                           data_type: DataType, negate: bool = False) -> np.ndarray:
        """Filter numpy array rows where column equals the specified value"""
        if column_index < 0 or column_index >= data.shape[1]:
            raise ValueError(f"Column index {column_index} out of bounds")
        
        # Convert value based on data type
        if data_type == DataType.NUMBER:
            try:
                compare_value = float(value)
                column_data = data[:, column_index].astype(float)
                mask = np.isclose(column_data, compare_value)
            except (ValueError, TypeError):
                # Fall back to string comparison if conversion fails
                column_data = np.array([str(x) for x in data[:, column_index]])
                mask = column_data == str(value)
        elif data_type == DataType.BOOLEAN:
            try:
                compare_value = bool(value)
                mask = data[:, column_index].astype(bool) == compare_value
            except (ValueError, TypeError):
                column_data = np.array([str(x) for x in data[:, column_index]])
                mask = column_data == str(value)
        elif data_type == DataType.DATE:
            # Convert both to timestamps for comparison
            try:
                compare_timestamp = DataTypeUtils.parse_date(str(value))
                column_timestamps = np.array([
                    DataTypeUtils.parse_date(str(x)) for x in data[:, column_index]
                ])
                mask = np.isclose(column_timestamps, compare_timestamp)
            except Exception as e:
                logger.warning(f"Date comparison error: {e}, falling back to string comparison")
                column_data = np.array([str(x) for x in data[:, column_index]])
                mask = column_data == str(value)
        else:  # STRING, CATEGORICAL, UNKNOWN
            column_data = np.array([str(x) for x in data[:, column_index]])
            mask = column_data == str(value)
        
        if negate:
            return data[~mask]
        else:
            return data[mask]
    
    @staticmethod
    def between_matcher_numpy(data: np.ndarray, column_index: int, lower_value: Any,
                            upper_value: Any, data_type: DataType, negate: bool = False) -> np.ndarray:
        """Filter numpy array rows where column is between two values"""
        if column_index < 0 or column_index >= data.shape[1]:
            raise ValueError(f"Column index {column_index} out of bounds")
        
        # Convert values based on data type
        if data_type == DataType.NUMBER:
            try:
                lower = float(lower_value)
                upper = float(upper_value)
                column_data = data[:, column_index].astype(float)
                mask = (column_data >= lower) & (column_data <= upper)
            except (ValueError, TypeError):
                # Fall back to string comparison
                lower = str(lower_value)
                upper = str(upper_value)
                column_data = np.array([str(x) for x in data[:, column_index]])
                mask = (column_data >= lower) & (column_data <= upper)
        elif data_type == DataType.DATE:
            try:
                lower_timestamp = DataTypeUtils.parse_date(str(lower_value))
                upper_timestamp = DataTypeUtils.parse_date(str(upper_value))
                column_timestamps = np.array([
                    DataTypeUtils.parse_date(str(x)) for x in data[:, column_index]
                ])
                mask = (column_timestamps >= lower_timestamp) & (column_timestamps <= upper_timestamp)
            except Exception as e:
                logger.warning(f"Date comparison error: {e}, falling back to string comparison")
                lower = str(lower_value)
                upper = str(upper_value)
                column_data = np.array([str(x) for x in data[:, column_index]])
                mask = (column_data >= lower) & (column_data <= upper)
        else:  # STRING, BOOLEAN, CATEGORICAL, UNKNOWN
            lower = str(lower_value)
            upper = str(upper_value)
            column_data = np.array([str(x) for x in data[:, column_index]])
            mask = (column_data >= lower) & (column_data <= upper)
        
        if negate:
            return data[~mask]
        else:
            return data[mask]
    
    @staticmethod
    def apply_matchers(data: np.ndarray, column_names: List[str],
                      matchers: List[RowMatcher],
                      metadata_dict: Dict = None,
                      negate: bool = False) -> np.ndarray:
        """Apply a list of matchers to filter a numpy array"""
        if len(matchers) == 0:
            return data
            
        filtered_data = data.copy()
        
        for matcher in matchers:
            if matcher.columnName.startswith(TRUSTY_PREFIX):
                # Handle internal columns (like execution_id, data_tag)
                internal_column = matcher.columnName.replace(TRUSTY_PREFIX, "").upper()
                
                # Find the index of this internal column in metadata
                internal_idx = -1
                metadata_column_name = internal_column.lower()  # Typically stored lowercase
                
                if metadata_column_name in column_names:
                    internal_idx = column_names.index(metadata_column_name)
                
                if internal_idx == -1:
                    logger.warning(f"Internal column {internal_column} not found, skipping matcher")
                    continue
                
                # Apply the matcher
                if matcher.operation == MatchOperation.BETWEEN:
                    filtered_data = DownloadUtils.between_matcher_numpy(
                        filtered_data, internal_idx, matcher.values[0], matcher.values[1], 
                        DataType.STRING, negate)
                elif matcher.operation == MatchOperation.EQUALS:
                    filtered_data = DownloadUtils.equals_matcher_numpy(
                        filtered_data, internal_idx, matcher.values[0], 
                        DataType.STRING, negate)
            else:
                # Regular column matching
                if matcher.columnName not in column_names:
                    logger.warning(f"Column {matcher.columnName} not found, skipping matcher")
                    continue
                    
                column_index = column_names.index(matcher.columnName)
                data_type = DownloadUtils.get_data_type(metadata_dict or {}, matcher)
                
                if matcher.operation == MatchOperation.BETWEEN:
                    filtered_data = DownloadUtils.between_matcher_numpy(
                        filtered_data, column_index, matcher.values[0], matcher.values[1], 
                        data_type, negate)
                elif matcher.operation == MatchOperation.EQUALS:
                    filtered_data = DownloadUtils.equals_matcher_numpy(
                        filtered_data, column_index, matcher.values[0], 
                        data_type, negate)
        
        return filtered_data

class CSVConverter:
    """Utility for converting data to CSV format without pandas"""
    
    @staticmethod
    def numpy_to_csv(data: np.ndarray, column_names: List[str], include_header: bool = True) -> str:
        """Convert a numpy array to a CSV string without using pandas"""
        # Ensure number of columns matches number of column names
        if data.shape[1] != len(column_names):
            logger.warning(f"Column count mismatch: data has {data.shape[1]} columns but {len(column_names)} names provided")
            # Adjust column names if needed
            if data.shape[1] > len(column_names):
                # Add generic column names for extra columns
                extra_columns = [f"column_{i}" for i in range(len(column_names), data.shape[1])]
                column_names = column_names + extra_columns
            else:
                # Truncate column names if there are too many
                column_names = column_names[:data.shape[1]]
        
        # Create CSV string using io and csv modules
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header if requested
        if include_header:
            writer.writerow([''] + column_names)  # Add empty cell for index
        
        # Write data rows with row index
        for i, row in enumerate(data):
            # Convert all values to strings and add row index
            row_values = [str(i)]  # Row index
            for val in row:
                if isinstance(val, (np.ndarray, bytes)) and hasattr(val, 'decode'):
                    try:
                        # Handle binary data
                        row_values.append(val.decode('utf-8', errors='replace'))
                    except:
                        row_values.append(str(val))
                else:
                    row_values.append(str(val))
            writer.writerow(row_values)
        
        return output.getvalue()

@router.post("/download")
async def download_data(payload: DataRequestPayload):
    """Download model data with filtering options."""
    try:
        model_id = payload.modelId
        logger.info(f"Processing data download request for model: {model_id}")
        
        # Check if model exists
        input_dataset = model_id + INPUT_SUFFIX
        output_dataset = model_id + OUTPUT_SUFFIX
        metadata_dataset = model_id + METADATA_SUFFIX
        
        if not await storage_interface.dataset_exists(input_dataset):
            raise ValueError(f"No data found for model {model_id}")
        
        # Read data as numpy arrays
        inputs, input_names = await storage_interface.read_data(input_dataset)
        outputs, output_names = await storage_interface.read_data(output_dataset)
        
        # Get metadata if available
        metadata_list = []
        metadata_dict = {}
        try:
            from src.endpoints.data.data_upload import MetadataManager
            logger.info(f"Attempting to read metadata from {metadata_dataset}")
            metadata_list = await MetadataManager.read_metadata_safely(metadata_dataset)
            logger.info(f"Read {len(metadata_list)} metadata items")
            
            # Create a numpy array from metadata
            if metadata_list:
                # Get all unique keys from metadata
                all_keys = set()
                for item in metadata_list:
                    all_keys.update(item.keys())
                metadata_names = list(all_keys)
                logger.info(f"Metadata column names: {metadata_names}")
                
                # Create metadata array
                metadata_array = np.zeros((len(metadata_list), len(metadata_names)), dtype=object)
                for i, item in enumerate(metadata_list):
                    for j, key in enumerate(metadata_names):
                        metadata_array[i, j] = item.get(key, "")
            else:
                logger.warning(f"No metadata items found for {model_id}")
                metadata_array = np.array([])
                metadata_names = []
        except Exception as e:
            logger.error(f"Error processing metadata for {model_id}: {str(e)}", exc_info=True)
            metadata_array = np.array([])
            metadata_names = []
        
        # Combine input and output data
        combined_data = None
        combined_names = []
        
        # Only combine if we have data
        if inputs.size > 0:
            combined_data = inputs
            combined_names = list(input_names)  # Convert to list to ensure we can use extend
            
            if outputs.size > 0:
                # Ensure both arrays have the same number of rows
                if inputs.shape[0] != outputs.shape[0]:
                    raise ValueError(f"Input shape {inputs.shape} doesn't match output shape {outputs.shape}")
                
                # Stack horizontally
                combined_data = np.hstack((combined_data, outputs))
                combined_names.extend(output_names)
            
            # Add metadata if available and compatible
            if metadata_array.size > 0 and metadata_array.shape[0] == inputs.shape[0]:
                combined_data = np.hstack((combined_data, metadata_array))
                combined_names.extend(metadata_names)
        else:
            # No data to process
            raise ValueError(f"No input data found for model {model_id}")
        
        # Filter data based on the matchers
        # First apply matchAll filters
        filtered_data = combined_data
        if payload.matchAll:
            filtered_data = DownloadUtils.apply_matchers(
                filtered_data, combined_names, payload.matchAll, metadata_dict, False)
        
        # Then apply matchNone filters
        if payload.matchNone:
            filtered_data = DownloadUtils.apply_matchers(
                filtered_data, combined_names, payload.matchNone, metadata_dict, True)
        
        # Special handling for matchAny - needs to be OR logic
        if payload.matchAny:
            # Create empty result array
            result_rows = []
            
            # For each matcher, get matching rows and add to result
            for matcher in payload.matchAny:
                # Create a temporary array with just this matcher
                temp_result = DownloadUtils.apply_matchers(
                    filtered_data, combined_names, [matcher], metadata_dict, False)
                
                if temp_result.size > 0:
                    # Add rows to result if not already present
                    for row in temp_result:
                        row_tuple = tuple(row)
                        if row_tuple not in [tuple(r) for r in result_rows]:
                            result_rows.append(row)
            
            # If we found any matches, use them, otherwise keep filtered_data
            if result_rows:
                filtered_data = np.array(result_rows)
        
        # Convert to CSV 
        csv_data = CSVConverter.numpy_to_csv(filtered_data, combined_names, True)
        
        # Return response
        return DataResponsePayload(dataCSV=csv_data)
        
    except ValueError as e:
        logger.error(f"Error in data download request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in data download: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error downloading data: {str(e)}")
