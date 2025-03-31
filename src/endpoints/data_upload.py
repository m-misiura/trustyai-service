from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from src.service.data.model_data import ModelData


router = APIRouter()
logger = logging.getLogger(__name__)

# Create an in-memory data store for testing
_test_data_store = {}
_modeldata_patched = False

class ModelInferJointPayload(BaseModel):
    model_name: str
    data_tag: str = None
    is_ground_truth: bool = False
    request: Dict[str, Any]
    response: Dict[str, Any]


class SimpleDataPayload(BaseModel):
    """Data payload for uploading test and reference data."""
    featureNames: List[str]
    referenceData: List[List[float]]
    testData: List[List[float]]
class TestDataPayload(BaseModel):
    modelId: str
    featureNames: List[str]
    referenceData: List[List[float]]
    testData: List[List[float]]
    
# --- ModelData patching ---
def patch_model_data():
    """Patch ModelData class to use our in-memory test data."""
    global _modeldata_patched
    if _modeldata_patched:
        return  # Already patched
        
    # Store original methods
    if not hasattr(ModelData, '_original_data'):
        ModelData._original_data = ModelData.data
    if not hasattr(ModelData, '_original_column_names'):
        ModelData._original_column_names = ModelData.column_names
        
    # Override data method
    async def new_data(self, start_row=None, n_rows=None, get_input=True, get_output=True, get_metadata=True):
        """Use in-memory data instead of storage backend."""
        logger.debug(f"ModelData.data called for model: {self.model_name}")
        
        # Check for model in data store
        if self.model_name in _test_data_store:
            logger.debug(f"Using in-memory data for {self.model_name}")
            data = _test_data_store[self.model_name]
            return (
                data['inputs'] if get_input else None,
                data['outputs'] if get_output else None,
                data['metadata'] if get_metadata else None
            )
        
        # Check with normalized ID
        normalized_id = self.model_name.replace("_inputs", "")
        if normalized_id in _test_data_store:
            logger.debug(f"Using in-memory data for normalized ID: {normalized_id}")
            data = _test_data_store[normalized_id]
            return (
                data['inputs'] if get_input else None,
                data['outputs'] if get_output else None,
                data['metadata'] if get_metadata else None
            )
            
        # Fallback to original method
        return await ModelData._original_data(self, start_row, n_rows, get_input, get_output, get_metadata)
    
    # Override column_names method
    async def new_column_names(self):
        """Use in-memory column names instead of storage backend."""
        logger.debug(f"ModelData.column_names called for {self.model_name}")
        
        # Check for model in data store
        if self.model_name in _test_data_store:
            data = _test_data_store[self.model_name]
            return (
                data['input_names'],
                data['output_names'],
                data['metadata_names']
            )
        
        # Check with normalized ID
        normalized_id = self.model_name.replace("_inputs", "")
        if normalized_id in _test_data_store:
            data = _test_data_store[normalized_id]
            return (
                data['input_names'],
                data['output_names'],
                data['metadata_names']
            )
            
        # Fallback to original method
        return await ModelData._original_column_names(self)
    
    # Apply patches
    ModelData.data = new_data
    ModelData.column_names = new_column_names
    _modeldata_patched = True
    logger.info("ModelData patched to use in-memory test data")


@router.post("/data/upload")
async def upload_data(payload: ModelInferJointPayload):
    """Upload a batch of model data to TrustyAI."""
    try:
        logger.info(f"Received data upload for model: {payload.model_name}")
        # TODO: Implement
        return {"status": "success", "message": "Data uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")
    
class SimpleDataPayload(BaseModel):
    """Data payload for uploading test and reference data."""
    featureNames: List[str]
    referenceData: List[List[float]]
    testData: List[List[float]]

@router.post("/test/simple-data/{model_id}")
async def simple_data(model_id: str, payload: SimpleDataPayload):
    """Load test data for drift metrics testing."""
    try:
        # Convert lists to numpy arrays
        reference_data = np.array(payload.referenceData, dtype=np.float64)
        test_data = np.array(payload.testData, dtype=np.float64)
        
        # Combine data
        inputs = np.vstack([reference_data, test_data])
        
        # Create metadata with tags
        ref_count = len(payload.referenceData)
        test_count = len(payload.testData)
        
        metadata = np.empty((inputs.shape[0], 1), dtype='S10')
        metadata[:ref_count] = b'reference'
        metadata[ref_count:] = b'test'
        
        # Store data
        _test_data_store[model_id] = {
            'inputs': inputs, 
            'outputs': np.array([]),
            'metadata': metadata,
            'input_names': payload.featureNames,
            'output_names': [],
            'metadata_names': ['data_tag']
        }
        
        # Also store with _inputs suffix
        _test_data_store[f"{model_id}_inputs"] = _test_data_store[model_id]
        
        # Patch ModelData
        patch_model_data()
        
        return {
            "status": "success",
            "message": f"Test data loaded for {model_id}",
            "data_shape": inputs.shape,
            "reference_count": ref_count,
            "test_count": test_count
        }
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return {"status": "error", "message": str(e)}