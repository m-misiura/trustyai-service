from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Any
from src.service.data.model_data import ModelData
from src.service.metrics.drift.meanshift import Meanshift
import logging
import math
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)


class ScheduleId(BaseModel):
    requestId: str


# ApproxKSTest
class GKSketch(BaseModel):
    epsilon: float
    summary: List[Dict[str, Any]] = []
    xmin: float
    xmax: float
    numx: int


class ApproxKSTestMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    thresholdDelta: Optional[float] = None
    referenceTag: Optional[str] = None
    fitColumns: List[str] = []
    epsilon: Optional[float] = None
    sketchFitting: Optional[Dict[str, GKSketch]] = None


@router.post("/metrics/drift/approxkstest")
async def compute_approxkstest(request: ApproxKSTestMetricRequest):
    """Compute the current value of ApproxKSTest metric."""
    try:
        logger.info(f"Computing ApproxKSTest for model: {request.modelId}")
        # TODO: Implement
        return {"status": "success", "value": 0.5}
    except Exception as e:
        logger.error(f"Error computing ApproxKSTest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/approxkstest/definition")
async def get_approxkstest_definition():
    """Provide a general definition of ApproxKSTest metric."""
    return {
        "name": "Approximate Kolmogorov-Smirnov Test",
        "description": "Description.",
    }


@router.post("/metrics/drift/approxkstest/request")
async def schedule_approxkstest(
    request: ApproxKSTestMetricRequest, background_tasks: BackgroundTasks
):
    """Schedule a recurring computation of ApproxKSTest metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling ApproxKSTest computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/drift/approxkstest/request")
async def delete_approxkstest_schedule(schedule: ScheduleId):
    """Delete a recurring computation of ApproxKSTest metric."""
    logger.info(f"Deleting ApproxKSTest schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/drift/approxkstest/requests")
async def list_approxkstest_requests():
    """List the currently scheduled computations of ApproxKSTest metric."""
    # TODO: Implement
    return {"requests": []}


# FourierMMD
class FourierMMDParameters(BaseModel):
    nWindow: Optional[int] = None
    nTest: Optional[int] = None
    nMode: Optional[int] = None
    randomSeed: Optional[int] = None
    sig: Optional[float] = None
    deltaStat: Optional[bool] = None
    epsilon: Optional[float] = None


class FourierMMDFitting(BaseModel):
    randomSeed: Optional[int] = None
    deltaStat: Optional[bool] = None
    nMode: Optional[int] = None
    scale: Optional[List[float]] = None
    aRef: Optional[List[float]] = None
    meanMMD: Optional[float] = None
    stdMMD: Optional[float] = None


class FourierMMDMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    thresholdDelta: Optional[float] = None
    referenceTag: Optional[str] = None
    fitColumns: List[str] = []
    parameters: Optional[FourierMMDParameters] = None
    gamma: Optional[float] = None
    fitting: Optional[FourierMMDFitting] = None


@router.post("/metrics/drift/fouriermmd")
async def compute_fouriermmd(request: FourierMMDMetricRequest):
    """Compute the current value of FourierMMD metric."""
    try:
        logger.info(f"Computing FourierMMD for model: {request.modelId}")
        # TODO: Implement
        return {"status": "success", "value": 0.5}
    except Exception as e:
        logger.error(f"Error computing FourierMMD: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/fouriermmd/definition")
async def get_fouriermmd_definition():
    """Provide a general definition of FourierMMD metric."""
    return {
        "name": "FourierMMD Drift",
        "description": "Description",
    }


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


# KSTest
class KSTestMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    thresholdDelta: Optional[float] = None
    referenceTag: Optional[str] = None
    fitColumns: List[str] = []


@router.post("/metrics/drift/kstest")
async def compute_kstest(request: KSTestMetricRequest):
    """Compute the current value of KSTest metric."""
    try:
        logger.info(f"Computing KSTest for model: {request.modelId}")
        # TODO: Implement
        return {"status": "success", "value": 0.5}
    except Exception as e:
        logger.error(f"Error computing KSTest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/kstest/definition")
async def get_kstest_definition():
    """Provide a general definition of KSTest metric."""
    return {
        "name": "Kolmogorov-Smirnov Test",
        "description": "Description.",
    }


@router.post("/metrics/drift/kstest/request")
async def schedule_kstest(
    request: KSTestMetricRequest, background_tasks: BackgroundTasks
):
    """Schedule a recurring computation of KSTest metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling KSTest computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/drift/kstest/request")
async def delete_kstest_schedule(schedule: ScheduleId):
    """Delete a recurring computation of KSTest metric."""
    logger.info(f"Deleting KSTest schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/drift/kstest/requests")
async def list_kstest_requests():
    """List the currently scheduled computations of KSTest metric."""
    # TODO: Implement
    return {"requests": []}


# Meanshift
class StatisticalSummaryValues(BaseModel):
    mean: float
    variance: float
    n: int
    max: float
    min: float
    sum: float
    standardDeviation: Optional[float] = None

    @field_validator('standardDeviation', mode='before')
    def calculate_std_if_missing(cls, v, info):
        values = info.data
        if v is None and 'variance' in values:
            return math.sqrt(values['variance']) if values['variance'] > 0 else 0
        return v


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
    """Compute the current value of Meanshift metric."""
    try:
        logger.info(f"Computing Meanshift for model: {request.modelId}")

        # Enhanced validation
        if not request.referenceTag:
            return {
                "status": "error",
                "message": "Must provide a reference tag in request defining the original data distribution",
            }

        if request.batchSize is not None and request.batchSize <= 0:
            return {
                "status": "error",
                "message": "Request batch size must be bigger than 0",
            }

        model_data = ModelData(request.modelId)

        # Create Meanshift instance
        if request.fitting:
            # Use provided fitting parameters
            meanshift = Meanshift.from_fitting(request.fitting)
        else:
            # Fit from reference data
            try:
                meanshift = await Meanshift.from_model_data(
                    model_data, request.referenceTag
                )
            except ValueError as e:
                return {"status": "error", "message": str(e)}

        # Get test data for drift detection
        batch_size = request.batchSize or 100
        inputs, _, metadata = await model_data.data(n_rows=batch_size)

        if inputs is None or len(inputs) == 0:
            return {"status": "error", "message": "No data found for drift calculation"}

        # Get column names
        input_names, _, _ = await model_data.column_names()

        # Filter columns if specified
        if request.fitColumns and len(request.fitColumns) > 0:
            column_indices = [
                i for i, name in enumerate(input_names) if name in request.fitColumns
            ]
            if not column_indices:
                return {
                    "status": "error",
                    "message": "None of the specified columns found in the model data",
                }

            filtered_inputs = inputs[:, column_indices]
            filtered_names = [input_names[i] for i in column_indices]
        else:
            filtered_inputs = inputs
            filtered_names = input_names

        # Calculate meanshift
        alpha = request.thresholdDelta or 0.05
        results = meanshift.calculate(filtered_inputs, filtered_names, alpha)

        # Format results to match Java implementation more closely
        namedValues = {}
        for column_name, result in results.items():
            namedValues[column_name] = result.p_value

        formatted_results = {
            "status": "success",
            "namedValues": namedValues,  # Adding named values like in Java
            "results": {
                column_name: result.to_dict() for column_name, result in results.items()
            },
        }

        # Calculate overall drift
        if results:
            drift_detected_count = sum(
                1 for result in results.values() if result.reject_null
            )
            overall_drift_score = drift_detected_count / len(results) if results else 0
            drift_detected = overall_drift_score > 0.5  # Majority of columns show drift

            formatted_results["overall"] = {
                "driftScore": overall_drift_score,
                "driftDetected": drift_detected,
                "driftedColumns": drift_detected_count,
                "totalColumns": len(results),
            }

            # Add detailed column results for better interpretation like in Java
            col_details = []
            max_col_len = max(len(col) for col in results.keys()) if results else 0

            for col_name, result in results.items():
                reject = result.reject_null
                detail = {
                    "column": col_name,
                    "pValue": result.p_value,
                    "tStat": result.t_stat,
                    "threshold": alpha,
                    "driftDetected": reject,
                    "interpretation": f"Column has p={result.p_value:.6f} probability of coming from the training distribution",
                }
                if reject:
                    detail["interpretation"] += f" p <= {alpha} -> [SIGNIFICANT DRIFT]"
                else:
                    detail["interpretation"] += f" p > {alpha}"

                col_details.append(detail)

            formatted_results["columnDetails"] = col_details

        return formatted_results

    except Exception as e:
        logger.error(f"Error computing Meanshift: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/meanshift/definition")
async def get_meanshift_definition():
    """Provide a general definition of Meanshift metric."""
    return {
        "name": "Meanshift",
        "description": "MeanShift gives the per-column probability that the data values seen in a test dataset come from the same distribution of a training dataset, under the assumption that the values are normally distributed.",
        "interpretation": "Low p-values (below the threshold) indicate significant drift in the distribution of values.",
        "thresholdMeaning": "P-value threshold for significance testing. Lower values require stronger evidence of drift.",
    }


@router.post("/metrics/drift/meanshift/request")
async def schedule_meanshift(
    request: MeanshiftMetricRequest, background_tasks: BackgroundTasks
):
    """Schedule a recurring computation of Meanshift metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling Meanshift computation with ID: {request_id}")
    try:
        # Validate model exists
        model_data = ModelData(request.modelId)
        input_rows, _, _ = await model_data.row_counts()

        if input_rows == 0:
            raise ValueError(f"No data found for model {request.modelId}")

        # TODO: Store the schedule in the persistent storage
        # This would involve creating a scheduled task entry with:
        # - request_id
        # - model_id
        # - metric type (meanshift)
        # - parameters (thresholdDelta, referenceTag, fitColumns, etc.)
        # - schedule information

        # For now, we just return the ID
        return {"requestId": request_id}

    except Exception as e:
        logger.error(f"Error scheduling Meanshift: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error scheduling metric: {str(e)}"
        )


@router.delete("/metrics/drift/meanshift/request")
async def delete_meanshift_schedule(schedule: ScheduleId):
    """Delete a recurring computation of Meanshift metric."""
    logger.info(f"Deleting Meanshift schedule: {schedule.requestId}")

    try:
        # TODO: Remove the schedule from the persistent storage
        # This would involve deleting the scheduled task with the given ID

        return {
            "status": "success",
            "message": f"Schedule {schedule.requestId} deleted",
        }

    except Exception as e:
        logger.error(f"Error deleting Meanshift schedule: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting schedule: {str(e)}"
        )


@router.get("/metrics/drift/meanshift/requests")
async def list_meanshift_requests():
    """List the currently scheduled computations of Meanshift metric."""
    try:
        # TODO: Retrieve scheduled tasks from the persistent storage
        # This would involve querying all scheduled tasks of type meanshift

        # For now, return empty list
        return {"requests": []}

    except Exception as e:
        logger.error(f"Error listing Meanshift requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing requests: {str(e)}")

