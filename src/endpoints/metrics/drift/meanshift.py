from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from src.service.metrics.drift.meanshift import Meanshift
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)


class ScheduleId(BaseModel):
    requestId: str


# Meanshift
class StatisticalSummaryValues(BaseModel):
    mean: float
    variance: float
    n: int
    max: float
    min: float
    sum: float
    standardDeviation: float


class MeanshiftMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    thresholdDelta: Optional[float] = None
    referenceTag: Optional[str] = None
    fitColumns: List[str] = []
    fitting: Optional[Dict[str, StatisticalSummaryValues]] = None


@router.post("/metrics/drift/meanshift")
async def compute_meanshift(request: MeanshiftMetricRequest):
    try:
        model_id = request.modelId
        logger.info(f"Computing Meanshift for model: {model_id}")
        
        # Direct access to input data - skip ModelData intermediary
        input_dataset = model_id + INPUT_SUFFIX
        metadata_dataset = model_id + METADATA_SUFFIX
        
        # Read input data
        if not await storage_interface.dataset_exists(input_dataset):
            return {"error": f"Input dataset {input_dataset} does not exist"}
        
        input_data, input_names = await storage_interface.read_data(input_dataset)
        logger.info(f"Read input data: shape={input_data.shape}, columns={input_names}")
        
        # Convert NumPy array of column names to a list if needed
        if isinstance(input_names, np.ndarray):
            input_names = input_names.tolist()
        
        # Read metadata
        if not await storage_interface.dataset_exists(metadata_dataset):
            return {"error": f"Metadata dataset {metadata_dataset} does not exist"}
        
        
        metadata_data, metadata_names = await storage_interface.read_data(metadata_dataset)
        logger.info(f"Read metadata: shape={metadata_data.shape}, columns={metadata_names}")
        
        # Extract reference and test data indices
        ref_indices = []
        test_indices = []
        
        # Process metadata
        logger.info("Processing metadata")
        
        # Print sample metadata row for debugging
        if len(metadata_data) > 0:
            sample_row = metadata_data[0]
            logger.info(f"Sample metadata row type: {type(sample_row)}, value: {sample_row}")
        
        
        # Process each metadata record
        for i in range(len(metadata_data)):
            try:
                row = metadata_data[i]
                
                # Case 1: Direct dictionary format
                if isinstance(row, dict) and 'data_tag' in row:
                    if i < 3:
                        logger.info(f"Metadata {i} (direct dict): {row}")
                    
                    # Add to proper index list based on tag
                    if row['data_tag'] == request.referenceTag:
                        ref_indices.append(i)
                    else:
                        test_indices.append(i)
                
                # Case 2: Tuple of bytes format
                elif isinstance(row, tuple) and len(row) > 0 and isinstance(row[0], bytes):
                    pickled_data = row[0]
                    metadata_dict = pickle.loads(pickled_data)
                    
                    if i < 3:
                        logger.info(f"Metadata {i} (pickled): {metadata_dict}")
                    
                    if metadata_dict.get('data_tag') == request.referenceTag:
                        ref_indices.append(i)
                    else:
                        test_indices.append(i)
                
                # Case 3: Structured array with 'metadata' field
                elif hasattr(row, 'dtype') and 'metadata' in row.dtype.names:
                    pickled_data = row['metadata']
                    metadata_dict = pickle.loads(pickled_data)
                    
                    if i < 3:
                        logger.info(f"Metadata {i} (structured): {metadata_dict}")
                    
                    if metadata_dict.get('data_tag') == request.referenceTag:
                        ref_indices.append(i)
                    else:
                        test_indices.append(i)
                else:
                    if i < 3:
                        logger.warning(f"Unknown metadata format at index {i}: {type(row)}")
                    
            except Exception as e:
                logger.warning(f"Error processing metadata row {i}: {str(e)}")
        
        logger.info(f"Found {len(ref_indices)} reference indices and {len(test_indices)} test indices")
        
        # Check if we have enough data
        if not ref_indices:
            return {"error": "No reference data found"}
        if not test_indices:
            return {"error": "No test data found"}
        
        # Extract reference and test data
        ref_data = input_data[ref_indices]
        test_data = input_data[test_indices]
        
        logger.info(f"Reference data shape: {ref_data.shape}, Test data shape: {test_data.shape}")
        
        # Calculate meanshift
        meanshift = Meanshift.precompute(ref_data, input_names)
        
        alpha = request.thresholdDelta or 0.05
        logger.info(f"Calling calculate with input_names type: {type(input_names)}")
        if isinstance(input_names, (list, tuple)):
            logger.info(f"Column names: {input_names}")
        else:
            logger.info(f"Column names: {input_names.tolist() if hasattr(input_names, 'tolist') else str(input_names)}")
        
        mean_shift_results = meanshift.calculate(
            test_data=test_data, 
            column_names=input_names,  # Using the converted list of column names
            alpha=alpha
        )
        
        
        # Format results
        namedValues = {}
        for column_name, result in mean_shift_results.items():
            namedValues[column_name] = float(result.p_value)
            
        return namedValues
            
    except Exception as e:
        logger.error(f"Error computing Meanshift: {str(e)}", exc_info=True)
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/meanshift/definition")
async def get_meanshift_definition():
    """Provide a general definition of Meanshift metric."""
    return {
        "name": "Meanshift Drift Detection",
        "description": "MeanShift gives the per-column probability that the data values seen in a test dataset come from the same distribution of a training dataset, under the assumption that the values are normally distributed.",
        "interpretation": "Low p-values (below the threshold) indicate significant drift in the distribution of values.",
        "thresholdMeaning": "P-value threshold for significance testing. Lower values require stronger evidence of drift."
    }


@router.post("/metrics/drift/meanshift/request")
async def schedule_meanshift(
    request: MeanshiftMetricRequest, background_tasks: BackgroundTasks
):
    """Schedule a recurring computation of Meanshift metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling Meanshift computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/drift/meanshift/request")
async def delete_meanshift_schedule(schedule: ScheduleId):
    """Delete a recurring computation of Meanshift metric."""
    logger.info(f"Deleting Meanshift schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/drift/meanshift/requests")
async def list_meanshift_requests():
    """List the currently scheduled computations of Meanshift metric."""
    # TODO: Implement
    return {"requests": []}
