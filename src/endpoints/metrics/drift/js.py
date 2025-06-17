from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union, cast
from src.core.metrics.drift.jensenshannon import JensenShannon, JensenShannonBaseline
from src.service.constants import INPUT_SUFFIX, METADATA_SUFFIX
from src.service.data.storage import get_storage_interface
import logging
import numpy as np
from numpy.typing import NDArray
import uuid
import os

router = APIRouter()
logger = logging.getLogger(__name__)



class ScheduleId(BaseModel):
    requestId: str


class JensenShannonFitting(BaseModel):
    thresholdDelta: Optional[float] = None


class JensenShannonMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    referenceTag: Optional[str] = None
    fitColumns: List[str] = []
    fitting: Optional[JensenShannonFitting] = None
    normalizeValues: bool = False
    usePerChannel: bool = False
    numCV: int = 10
    cvSize: int = 16


@router.post("/metrics/drift/jensenshannon")
async def compute_jensenshannon(request: JensenShannonMetricRequest) -> Dict[str, Any]:
    """Compute the current value of Jensen-Shannon drift metric."""
    storage_interface = get_storage_interface()
    try:
        logger.info(f"Computing Jensen-Shannon drift for model: {request.modelId}")

        # Validate initial params
        if not request.referenceTag:
            raise HTTPException(
                status_code=400,
                detail="Must provide a reference tag in request defining the original data distribution",
            )
        if request.batchSize is not None and request.batchSize <= 0:
            raise HTTPException(status_code=400, detail="Request batch size must be bigger than 0")

        # Data retrieval setup
        model_id = request.modelId
        reference_tag = request.referenceTag
        input_dataset = model_id + INPUT_SUFFIX
        metadata_dataset = model_id + METADATA_SUFFIX

        if not await storage_interface.dataset_exists(input_dataset):
            return {"error": f"Input dataset {input_dataset} does not exist"}

        if not await storage_interface.dataset_exists(metadata_dataset):
            return {"error": f"Metadata dataset {metadata_dataset} does not exist"}

        # retrieve input and metadata - they are parallel data structs
        inputs, input_names = await storage_interface.read_data(input_dataset)
        metadata, metadata_names = await storage_interface.read_data(metadata_dataset)

        logger.info(f"Read input data: shape={inputs.shape}, columns={input_names}")
        logger.info(f"Read metadata: length={len(metadata)}, columns={metadata_names}")

        if inputs is None or len(inputs) == 0:
            raise HTTPException(status_code=400, detail="No input data found")
        if len(metadata) == 0:
            return {"error": "No metadata found"}

        if isinstance(input_names, np.ndarray):
            input_names = input_names.tolist()

        try:
            # [{"data_tag": "reference"}, {"data_tag": "test"}]
            metadata_dicts = [dict(zip(metadata_names, row)) for row in metadata]
            data_tags = np.array([row["tag"] for row in metadata_dicts])
            available_tags = set(data_tags)

            # Create boolean masks for reference and test data
            ref_mask = data_tags == reference_tag
            test_mask = ~ref_mask  # I am assuming that we do not have a tag to select test data and we can just inverse

            ref_indices = np.where(ref_mask)[0]
            test_indices = np.where(test_mask)[0]

            logger.info(f"Available tags: {available_tags}")
            logger.info(f"Found {len(ref_indices)} reference rows and {len(test_indices)} test rows")

            if not ref_indices.size:
                raise HTTPException(
                    status_code=400,
                    detail=f"No reference data found with tag '{reference_tag}'. Available tags: {list(available_tags)}",
                )

            if not test_indices.size:
                raise HTTPException(status_code=400, detail="No test data found")

            ref_data = inputs[ref_indices]
            test_data = inputs[test_indices]

        except Exception as e:
            logger.error(f"Error processing metadata: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing metadata: {str(e)}")

        # Filter
        if request.fitColumns and len(request.fitColumns) > 0:
            print(f"Filtering columns based on fitColumns: {request.fitColumns}")
            column_indices = [
                i for i, name in enumerate(input_names) if name in request.fitColumns
            ]
            if not column_indices:
                raise HTTPException(
                    status_code=400,
                    detail="None of the specified fitColumns match input feature names"
                )
            filtered_ref_inputs = ref_data[:, column_indices]
            filtered_hyp_inputs = test_data[:, column_indices]
            filtered_names = [input_names[i] for i in column_indices]
        else:
            filtered_ref_inputs = ref_data
            filtered_hyp_inputs = test_data
            filtered_names = input_names

        # Calculate thresholds
        try:
            # Initialize threshold variables
            channel_thresholds: List[float] = []
            single_threshold: float = 0.0

            if request.fitting is None:
                num_cv = request.numCV
                cv_size = request.cvSize
                
                if request.usePerChannel:
                    baselines = JensenShannonBaseline.calculate_per_channel(
                        filtered_ref_inputs, num_cv, cv_size, request.normalizeValues
                    )
                    channel_thresholds = [b.max_threshold for b in baselines] 
                else:
                    baseline = JensenShannonBaseline.calculate(
                        filtered_ref_inputs, num_cv, cv_size, request.normalizeValues
                    )
                    single_threshold = baseline.max_threshold
            else:
                if request.usePerChannel:
                    threshold_delta = request.fitting.thresholdDelta if request.fitting.thresholdDelta is not None else 0.0
                    channel_thresholds = [threshold_delta] * filtered_ref_inputs.shape[1]
                else:
                    single_threshold = request.fitting.thresholdDelta if request.fitting.thresholdDelta is not None else 0.0
                    
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
                
        # Calculate drift
        try:
            
            if request.usePerChannel:
                js_channel_results = JensenShannon.calculate_per_channel(
                    filtered_ref_inputs,
                    filtered_hyp_inputs,
                    channel_thresholds, 
                    request.normalizeValues
                )

                channel_results = []
                for i, result in enumerate(js_channel_results):
                    col_name = filtered_names[i] if i < len(filtered_names) else f"feature_{i}"
                    channel_results.append({
                        "columnName": col_name,
                        "js_stat": float(result.js_stat),  # Ensure JSON serializable
                        "threshold": float(result.threshold),
                        "driftDetected": bool(result.reject),
                        "text_interpretation": f"Jensen-Shannon divergence is {result.js_stat:.6f}, threshold is {result.threshold:.6f}: {'SIGNIFICANT DRIFT' if result.reject else 'No significant drift.'}"
                    })
                
                # Calculate overall drift statistics
                overall_stats = {
                    "totalFeatures": len(channel_results),
                    "featuresWithDrift": sum(1 for r in channel_results if r["driftDetected"]),
                    "averageJS": sum(r["js_stat"] for r in channel_results) / len(channel_results),
                    "maxJS": max(r["js_stat"] for r in channel_results),
                    "minJS": min(r["js_stat"] for r in channel_results)
                }

                formatted_results = {
                    "status": "success",
                    "Result": channel_results,
                    "overall": overall_stats
                }
            else:

                result = JensenShannon.calculate(
                    filtered_ref_inputs,
                    filtered_hyp_inputs,
                    single_threshold,
                    request.normalizeValues
                )
                
                formatted_results = {
                    "status": "success",
                    "Result": {
                        "js_stat": float(result.js_stat),
                        "threshold": float(single_threshold),
                        "driftDetected": bool(result.reject),
                        "text_interpretation": f"Jensen-Shannon divergence is {result.js_stat:.6f}, threshold is {result.threshold:.6f}: {'SIGNIFICANT DRIFT' if result.reject else 'No significant drift.'}",
                    }
                }
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error calculating drift: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error calculating drift: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error computing Jensen-Shannon drift: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/jensenshannon/definition")
async def get_jensenshannon_definition() -> Dict[str, Any]:
    """Provide a general definition of the Jensen-Shannon drift metric."""
    return {
        "name": "Jensen Shannon Divergence",
        "description": """
        The Jensen Shannon divergence is a method of measuring the similarity between two probability distributions.
        It is based on the Kullback-Leibler divergence but addresses its limitations by being symmetric and bounded.
        
        Key properties:
        - Range: [0, 1] where 0 means identical distributions and 1 means maximally different
        - Symmetric: JS(p,q) = JS(q,p)
        - Bounded: Always finite, unlike KL divergence
        - Interpretable: Values can be directly compared across different distributions
        
        In the context of data drift detection:
        - Values close to 0 indicate the test data closely matches the reference distribution
        - Values approaching 1 indicate significant drift from the reference distribution
        - The metric can be calculated globally or per-feature to identify specific drift sources
        
        The algorithm supports both automatic threshold determination through cross-validation
        and manual threshold specification for drift detection.
        """,
        "parameters": {
            "normalizeValues": "Boolean, default=false. Whether to normalize JS values by tensor size.",
            "usePerChannel": "Boolean, default=false. Whether to calculate drift per feature.",
            "fitColumns": "List of strings, optional. Specific columns to analyze.",
            "numCV": "Integer, default=10. Number of cross-validation iterations for threshold calculation.",
            "cvSize": "Integer, default=16. Size of each cross-validation sample.",
            "fitting": "Optional object containing manual threshold specification."
        }
    }


@router.post("/metrics/drift/jensenshannon/request")
async def schedule_jensenshannon(
    request: JensenShannonMetricRequest, background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Schedule a recurring computation of Jensen-Shannon metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling Jensen-Shannon computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/drift/jensenshannon/request")
async def delete_jensenshannon_schedule(schedule: ScheduleId) -> Dict[str, str]:
    """Delete a recurring computation of Jensen-Shannon metric."""
    logger.info(f"Deleting Jensen-Shannon schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/drift/jensenshannon/requests")
async def list_jensenshannon_requests() -> Dict[str, List[Any]]:
    """List the currently scheduled computations of Jensen-Shannon metric."""
    # TODO: Implement
    return {"requests": []}
