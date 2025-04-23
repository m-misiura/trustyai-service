from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from src.service.metrics.drift.fourier_mmd.fourier_mmd import FourierMMD
from src.service.metrics.drift.fourier_mmd.fourier_mmd_fitting import FourierMMDFitting
from src.service.constants import INPUT_SUFFIX, METADATA_SUFFIX
from src.service.data.storage import get_storage_interface
import logging
import numpy as np
import uuid
import os

router = APIRouter()
logger = logging.getLogger(__name__)
storage_interface = get_storage_interface()


class ScheduleId(BaseModel):
    requestId: str


class FourierMMDParameters(BaseModel):
    """Parameters for FourierMMD calculation."""
    deltaStat: bool = Field(False, description="If True, compute MMD score for Dx = x[t+1]-x[t]")
    nTest: int = Field(100, description="Number of MMD scores to compute")
    nWindow: int = Field(100, description="Number of samples to compute a MMD score")
    sig: float = Field(1.0, description="Sigma, a scale parameter of the kernel")
    nMode: int = Field(100, description="Number of Fourier modes to approximate the kernel")
    epsilon: float = Field(1e-6, description="Minimum value for standard deviation")


# FourierMMD
class FourierMMDMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    thresholdDelta: float = Field(0.05, description="Threshold for p-value above which drift is detected")
    gamma: float = Field(1.0, description="Parameter controlling sensitivity")
    referenceTag: Optional[str] = None
    parameters: FourierMMDParameters = Field(default_factory=FourierMMDParameters)
    fitting: Optional[Dict[str, Any]] = None


class MetricValueCarrier(BaseModel):
    value: float
    namedValues: Optional[Dict[str, float]] = None
    description: Optional[str] = None


class MetricThreshold(BaseModel):
    min: float = 0.0
    max: float
    value: float


@router.post("/metrics/drift/fouriermmd")
async def compute_fouriermmd(request: FourierMMDMetricRequest) -> Union[Dict[str, float], Dict[str, str]]:
    """Compute the FourierMMD metric."""
    try:
        model_id = request.modelId
        reference_tag = request.referenceTag
        logger.info(f"Computing FourierMMD for model: {model_id}, reference tag: {reference_tag}")
        
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
        
        # Process metadata
        if len(metadata_data) == 0:
            return {"error": "No metadata found"}
            
        # Process all metadata rows to extract data tags
        for i in range(len(metadata_data)):
            try:
                row = metadata_data[i]
                metadata_dict = None
                
                # Handle different potential metadata formats
                if isinstance(row, dict) and 'data_tag' in row:
                    metadata_dict = row
                elif isinstance(row, np.ndarray) and 'data_tag' in metadata_names:
                    # If metadata is stored as a numpy array with named columns
                    tag_idx = list(metadata_names).index('data_tag')
                    metadata_dict = {'data_tag': row[tag_idx]}
                elif isinstance(row, (bytes, np.void)) and hasattr(row, 'dtype') and 'metadata' in row.dtype.names:
                    # Special handling for pickled metadata
                    import pickle
                    try:
                        metadata_bytes = row['metadata']
                        clean_bytes = metadata_bytes.rstrip(b'\x00') if isinstance(metadata_bytes, bytes) else metadata_bytes
                        metadata_dict = pickle.loads(clean_bytes, encoding='latin1')
                    except Exception as e:
                        logger.warning(f"Failed to unpickle metadata: {e}")
                else:
                    # Try alternate formats or read from the metadata column
                    for col_idx, col_name in enumerate(metadata_names):
                        if col_name == 'data_tag' and col_idx < len(row):
                            metadata_dict = {'data_tag': row[col_idx]}
                            break
        
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
            return {"error": "No test data found (all data has the reference tag)"}
        
        # Extract reference and test data
        ref_data = input_data[ref_indices]
        test_data = input_data[test_indices]
        
        logger.info(f"Reference data shape: {ref_data.shape}, Test data shape: {test_data.shape}")
        
        # Check for sufficient test data
        if len(test_indices) == 0:
            logger.warning("Test data has no observations; FourierMMD results will not be numerically reliable.")
            # Return a default result with 0.0 p-value similar to Java implementation
            return {"value": 0.0}
        
        # Get parameters from request
        params = request.parameters
        
        # Initialize FourierMMD
        fourier_mmd_fitting = None
        if request.fitting:
            # Use provided fitting data
            logger.info("Using previously found FourierMMD fitting in request")
            fourier_mmd_fitting = FourierMMDFitting.from_dict(request.fitting)
        else:
            # Precompute fitting from reference data
            logger.info("Fitting a FourierMMD drift request")
            # Fixed parameter names to match the method signature
            fourier_mmd_fitting = FourierMMD.precompute(
                data=ref_data,  # Changed from train_data to data
                column_names=input_names,
                delta_stat=params.deltaStat,
                n_test=params.nTest,
                n_window=params.nWindow,
                sig=params.sig,
                random_seed=0,  # Match Java's use of 0 as seed
                n_mode=params.nMode,
                epsilon=params.epsilon
            )
            
            # Store fitting in request for potential caching
            request.fitting = fourier_mmd_fitting.to_dict()
        
        # Create FourierMMD with fitting
        fourier_mmd = FourierMMD(fourier_mmd_fitting=fourier_mmd_fitting)
        
        # Calculate FourierMMD result
        threshold = request.thresholdDelta
        gamma = request.gamma
        
        logger.info(f"Computing FourierMMD with threshold={threshold}, gamma={gamma}")
        
        result = fourier_mmd.calculate(test_data, threshold, gamma)
        
        # Get p-value from result
        p_value = result.get_p_value()
        statistic = result.get_stat_val()
        is_drifted = result.is_reject()
        
        logger.info(f"FourierMMD result: p_value={p_value}, statistic={statistic}, drift_detected={is_drifted}")
        
        # Create a specific definition string similar to Java implementation
        description = get_specific_definition(p_value, threshold)
        
        # Return p-value as in the Java implementation
        return {"value": float(p_value)}
        
    except Exception as e:
        logger.error(f"Error computing FourierMMD: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


def get_specific_definition(p_value: float, threshold: float) -> str:
    """Generate a specific definition for FourierMMD results similar to Java implementation."""
    general_def = "FourierMMD gives probability that the data values seen in a test dataset have drifted from the training dataset distribution, under the assumption that the computed MMD values are normally distributed."
    
    is_drifted = p_value > threshold
    result = [
        general_def,
        "",
        f"  - Test data has p={p_value:.6f} probability of having drifted from the training distribution."
    ]
    
    if is_drifted:
        result[-1] += f" p > {threshold:.6f} -> [SIGNIFICANT DRIFT]"
    else:
        result[-1] += f" p <= {threshold:.6f}"
    
    return os.linesep.join(result)


@router.get("/metrics/drift/fouriermmd/definition")
async def get_fouriermmd_definition():
    """Provide a general definition of FourierMMD metric."""
    return "FourierMMD gives probability that the data values seen in a test dataset have drifted from the training dataset distribution, under the assumption that the computed MMD values are normally distributed."


@router.post("/metrics/drift/fouriermmd/request")
async def schedule_fouriermmd(
    request: FourierMMDMetricRequest, background_tasks: BackgroundTasks
):
    """Schedule a recurring computation of FourierMMD metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling FourierMMD computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/drift/fouriermmd/request")
async def delete_fouriermmd_schedule(schedule: ScheduleId):
    """Delete a recurring computation of FourierMMD metric."""
    logger.info(f"Deleting FourierMMD schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/drift/fouriermmd/requests")
async def list_fouriermmd_requests():
    """List the currently scheduled computations of FourierMMD metric."""
    # TODO: Implement
    return {"requests": []}