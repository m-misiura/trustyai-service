from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from src.service.metrics.drift.meanshift import Meanshift
from src.service.constants import INPUT_SUFFIX, METADATA_SUFFIX
from src.service.data.storage import get_storage_interface
import logging
import numpy as np
import uuid
import pickle
import os

router = APIRouter()
logger = logging.getLogger(__name__)
storage_interface = get_storage_interface()

class ScheduleId(BaseModel):
    requestId: str

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
        reference_tag = request.referenceTag
        logger.info(f"Computing Meanshift for model: {model_id}, reference tag: {reference_tag}")
        
        # Access datasets
        input_dataset = model_id + INPUT_SUFFIX
        metadata_dataset = model_id + METADATA_SUFFIX
        
        # Check datasets exist
        if not await storage_interface.dataset_exists(input_dataset):
            return {"error": f"Input dataset {input_dataset} does not exist"}
        
        if not await storage_interface.dataset_exists(metadata_dataset):
            return {"error": f"Metadata dataset {metadata_dataset} does not exist"}
        
        # Read input data
        input_data, input_names = await storage_interface.read_data(input_dataset)
        logger.info(f"Read input data: shape={input_data.shape}, columns={input_names}")
        
        # Ensure input_names is a list
        if isinstance(input_names, np.ndarray):
            input_names = input_names.tolist()
        
        # Read metadata
        metadata_data, metadata_names = await storage_interface.read_data(metadata_dataset)
        logger.info(f"Read metadata: length={len(metadata_data)}, columns={metadata_names}")
        
        # Extract reference and test data indices
        ref_indices = []
        test_indices = []
        available_tags = set()
        
        # Process metadata - check first row to see format
        if len(metadata_data) == 0:
            return {"error": "No metadata found"}
            
        sample_row = metadata_data[0]
        logger.info(f"Sample metadata type: {type(sample_row)}")
        
        # Process all metadata rows
        for i in range(len(metadata_data)):
            try:
                row = metadata_data[i]
                metadata_dict = None
                if isinstance(row, dict) and 'data_tag' in row:
                    # Handle direct dictionary format
                    metadata_dict = row
                else:
                    # raise an error if the expected format is not found
                    raise ValueError(f"Unexpected metadata format at row {i}")
        
                if metadata_dict and 'data_tag' in metadata_dict:
                    tag = metadata_dict['data_tag']
                    available_tags.add(tag)
                    
                    if i < 3:
                        logger.info(f"Row {i} has tag: {tag}")
                    
                    if tag == reference_tag:
                        ref_indices.append(i)
                    else:
                        test_indices.append(i)
                        
            except Exception as e:
                logger.warning(f"Error processing metadata row {i}: {str(e)}")
        
        logger.info(f"Available tags: {available_tags}")
        logger.info(f"Found {len(ref_indices)} reference rows and {len(test_indices)} test rows")
        
        # Check if we have enough data
        if not ref_indices:
            if reference_tag:
                return {"error": f"No reference data found with tag '{reference_tag}'. Available tags: {list(available_tags)}"}
            else:
                return {"error": "No reference tag specified and no default reference data found"}
                
        if not test_indices:
            return {"error": "No test data found"}
        
        # Extract reference and test data
        ref_data = input_data[ref_indices]
        test_data = input_data[test_indices]
        
        logger.info(f"Reference data shape: {ref_data.shape}, Test data shape: {test_data.shape}")
        
        # Check if test data has enough observations
        if len(test_data) < 2:
            logger.warning("Test data has less than two observations; Meanshift results will not be numerically reliable.")
        
        # Handle fitting calculation
        meanshift = None
        if request.fitting is None:
            logger.debug(f"Fitting a meanshift drift request for model={model_id}")
            # Precompute statistics from reference data
            meanshift = Meanshift.precompute(ref_data, input_names)
        else:
            logger.debug(f"Using previously found meanshift fitting in request for model={model_id}")
            # Create Meanshift with provided fitting
            fitting_dict = {
                col_name: StatisticalSummaryValues.from_dict(stats.dict()) 
                for col_name, stats in request.fitting.items()
            }
            meanshift = Meanshift.from_fitting(fitting_dict)
        
        alpha = request.thresholdDelta or 0.05
        logger.info(f"Using alpha threshold: {alpha}")
        
        # Calculate meanshift
        mean_shift_results = meanshift.calculate(
            test_data=test_data, 
            column_names=input_names,
            alpha=alpha
        )
        
        # Format results for response
        named_values = {}
        for column_name, result in mean_shift_results.items():
            named_values[column_name] = float(result.p_value)
            
        return {"namedValues": named_values}
            
    except Exception as e:
        logger.error(f"Error computing Meanshift: {str(e)}", exc_info=True)
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


def get_specific_definition(named_values: Dict[str, float], threshold: float) -> str:
    """Generate a specific definition for Meanshift results similar to Java implementation."""
    general_def = "MeanShift gives the per-column probability that the data values seen in a test dataset come from the same distribution of a training dataset, under the assumption that the values are normally distributed."
    
    out = [general_def, ""]
    
    # Find max column name length for formatting
    max_col_len = max([len(col) for col in named_values.keys()]) if named_values else 0
    fmt = f"%{max_col_len}s"
    
    # Add details for each column
    for column_name, p_value in named_values.items():
        formatted_column = fmt % column_name
        reject = p_value <= threshold
        
        line = f"  - Column {formatted_column} has p={p_value:.6f} probability of coming from the training distribution."
        
        if reject:
            line += f" p <= {threshold:.6f} -> [SIGNIFICANT DRIFT]"
        else:
            line += f" p >  {threshold:.6f}"
            
        out.append(line)
        
    return os.linesep.join(out)


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