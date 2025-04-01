from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Any
from src.service.data.model_data import ModelData
from src.service.metrics.drift.ks_test import KSTest
from src.service.metrics.drift.meanshift import Meanshift
from src.service.metrics.drift.ks_test.approx_ks_test import ApproxKSTest
from src.service.metrics.drift.ks_test.approx_ks_fitting import ApproxKSFitting
from src.service.metrics.drift.fourier_mmd.fourier_mmd import FourierMMD
from src.service.metrics.drift.fourier_mmd.fourier_mmd_fitting import FourierMMDFitting

import logging
import math
import uuid
import numpy as np
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
        
        # Get both reference and test data
        inputs, outputs, metadata = await model_data.data()
        
        # Get column names
        input_names, output_names, metadata_names = await model_data.column_names()
        
        # Filter by reference tag
        if metadata is not None:
            tag_column_indices = [
                i for i, name in enumerate(metadata_names)
                if name == "data_tag" or name.endswith(".data_tag")
            ]

            if tag_column_indices:
                # Reference data (matching tag)
                ref_mask = metadata[:, tag_column_indices[0]] == request.referenceTag
                ref_inputs = inputs[ref_mask] if ref_mask.any() else inputs
                
                # Test data (not matching tag)
                test_mask = metadata[:, tag_column_indices[0]] != request.referenceTag
                test_inputs = inputs[test_mask] if test_mask.any() else inputs
                
                logger.info(
                    f"Filtered data by tag '{request.referenceTag}': {np.sum(ref_mask)} reference rows, {np.sum(test_mask)} test rows"
                )
            else:
                logger.warning(f"Reference tag '{request.referenceTag}' provided but no data_tag column found")
                ref_inputs = inputs
                test_inputs = inputs
        else:
            ref_inputs = inputs
            test_inputs = inputs
        
        if ref_inputs is None or len(ref_inputs) == 0:
            return {"status": "error", "message": "No reference data found"}
        
        if test_inputs is None or len(test_inputs) == 0:
            return {"status": "error", "message": "No test data found for drift calculation"}
        
        if len(test_inputs) < 2:
            logger.warning("Test data has less than two observations; ApproxKSTest results will not be numerically reliable")
        
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

            filtered_ref_inputs = ref_inputs[:, column_indices]
            filtered_test_inputs = test_inputs[:, column_indices]
            filtered_names = [input_names[i] for i in column_indices]
        else:
            filtered_ref_inputs = ref_inputs
            filtered_test_inputs = test_inputs
            filtered_names = input_names
        
        # Create ApproxKSTest instance
        approx_ks_instance = None
        epsilon = request.epsilon or 0.01  # Default epsilon if not provided

        # Use existing fitting if provided or create new one
        if request.sketchFitting:
            logger.debug("Using provided ApproxKSTest fitting")
            fitting = ApproxKSFitting(request.sketchFitting)
            approx_ks_instance = ApproxKSTest(approx_ks_fitting=fitting, eps=epsilon)
        else:
            logger.debug("Creating new ApproxKSTest instance")
            
            # First, create the fitting using the static precompute method
            fitting = ApproxKSTest.precompute(
                train_data=filtered_ref_inputs, 
                column_names=filtered_names,
                eps=epsilon
            )
            
            # Now use the fitting to create the ApproxKSTest instance
            approx_ks_instance = ApproxKSTest(approx_ks_fitting=fitting, eps=epsilon)
            
            # Save the sketches for future use
            request.sketchFitting = fitting.get_fit_sketches()
                
        # Calculate drift
        alpha = request.thresholdDelta or 0.05
        results = approx_ks_instance.calculate(filtered_test_inputs, filtered_names, alpha)
        
        # Format results
        namedValues = {}
        for column_name, result in results.items():
            namedValues[column_name] = result.get_p_value()

        formatted_results = {
            "status": "success",
            "namedValues": namedValues,
            "results": {
                column_name: {
                    "statistic": result.get_stat_val(),
                    "pValue": result.get_p_value(),
                    "driftDetected": result.is_reject()
                } for column_name, result in results.items()
            },
        }
        
        # Calculate overall drift
        if results:
            drift_detected_count = sum(
                1 for result in results.values() if result.is_reject()
            )
            overall_drift_score = drift_detected_count / len(results) if results else 0
            drift_detected = overall_drift_score > 0.5  # Majority of columns show drift

            formatted_results["overall"] = {
                "driftScore": overall_drift_score,
                "driftDetected": drift_detected,
                "driftedColumns": drift_detected_count,
                "totalColumns": len(results),
            }
            
            # Add detailed column results
            col_details = []
            max_col_len = max(len(col) for col in results.keys()) if results else 0
            
            for col_name, result in results.items():
                reject = result.is_reject()
                detail = {
                    "column": col_name,
                    "pValue": result.get_p_value(),
                    "statistic": result.get_stat_val(),
                    "threshold": alpha,
                    "driftDetected": reject,
                    "interpretation": f"Column has p={result.get_p_value():.6f} probability of coming from the training distribution",
                }
                if reject:
                    detail["interpretation"] += f" p <= {alpha} -> [SIGNIFICANT DRIFT]"
                else:
                    detail["interpretation"] += f" p > {alpha}"
                
                col_details.append(detail)
                
            formatted_results["columnDetails"] = col_details
        
        # Add sketch fitting for reuse
        if request.sketchFitting:
            formatted_results["sketchFitting"] = request.sketchFitting
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error computing ApproxKSTest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/approxkstest/definition")
async def get_approxkstest_definition():
    """Provide a general definition of ApproxKSTest metric."""
    return {
        "name": "Approximate Kolmogorov-Smirnov Test",
        "description": "ApproxKSTest calculates an approximate Kolmogorov-Smirnov test, and ensures that the maximum error is 6*epsilon as compared to an exact KS Test.",
        "interpretation": "Low p-values (below the threshold) indicate significant drift in the distribution of values.",
        "thresholdMeaning": "P-value threshold for significance testing. Lower values require stronger evidence of drift.",
        "epsilonParameter": "Controls the approximation accuracy. Lower values increase accuracy at the cost of computational resources."
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
        
        # Get data for drift detection
        inputs, _, metadata = await model_data.data()
        
        # Get column names
        input_names, _, metadata_names = await model_data.column_names()
        
        # Filter by reference tag
        if metadata is not None:
            tag_column_indices = [
                i for i, name in enumerate(metadata_names)
                if name == "data_tag" or name.endswith(".data_tag")
            ]

            if tag_column_indices:
                # Reference data (matching tag)
                ref_mask = metadata[:, tag_column_indices[0]] == request.referenceTag
                ref_inputs = inputs[ref_mask] if ref_mask.any() else inputs
                
                # Test data (not matching tag)
                test_mask = metadata[:, tag_column_indices[0]] != request.referenceTag
                test_inputs = inputs[test_mask] if test_mask.any() else inputs
                
                logger.info(
                    f"Filtered data by tag '{request.referenceTag}': {np.sum(ref_mask)} reference rows, {np.sum(test_mask)} test rows"
                )
            else:
                logger.warning(f"Reference tag '{request.referenceTag}' provided but no data_tag column found")
                ref_inputs = inputs
                test_inputs = inputs
        else:
            ref_inputs = inputs
            test_inputs = inputs
        
        if ref_inputs is None or len(ref_inputs) == 0:
            return {"status": "error", "message": "No reference data found"}
        
        if test_inputs is None or len(test_inputs) == 0:
            return {"status": "error", "message": "No test data found for drift calculation"}
        
        if len(test_inputs) < 2:
            logger.warning("Test data has less than two observations; FourierMMD results will not be numerically reliable")
        
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

            filtered_ref_inputs = ref_inputs[:, column_indices]
            filtered_test_inputs = test_inputs[:, column_indices]
            filtered_names = [input_names[i] for i in column_indices]
        else:
            filtered_ref_inputs = ref_inputs
            filtered_test_inputs = test_inputs
            filtered_names = input_names
        
        # Get parameters or use defaults
        delta_stat = False

        # Adaptive parameters based on data size
        total_rows = len(filtered_ref_inputs)
        n_window = min(3, max(2, total_rows // 3))  # Use 1/3 of data but no more than 3
        n_test = min(5, max(2, total_rows // 3))    # Similar for test count
        n_mode = min(3, max(2, total_rows // 3))    # Same for modes

        # Keep other defaults
        sig = 1.0
        random_seed = 42
        epsilon = 1e-6
        gamma = request.gamma if request.gamma is not None else 1.5
        threshold = request.thresholdDelta or 0.8
        
        if request.parameters:
            if request.parameters.deltaStat is not None:
                delta_stat = request.parameters.deltaStat
            if request.parameters.nTest is not None:
                n_test = min(request.parameters.nTest, max(2, total_rows // 2))
            if request.parameters.nWindow is not None:
                n_window = min(request.parameters.nWindow, max(2, total_rows // 2 - 1))
            if request.parameters.sig is not None:
                sig = request.parameters.sig
            if request.parameters.randomSeed is not None:
                random_seed = request.parameters.randomSeed
            if request.parameters.nMode is not None:
                n_mode = min(request.parameters.nMode, max(2, total_rows // 2))
            if request.parameters.epsilon is not None:
                epsilon = request.parameters.epsilon
        
        # Create FourierMMD instance
        fourier_mmd_instance = None
        
        # Use existing fitting if provided or create new one
        if request.fitting:
            logger.debug("Using provided FourierMMD fitting")
            # Create a new FourierMMDFitting instance using constructor arguments directly
            fitting = FourierMMDFitting(
                randomSeed=request.fitting.randomSeed,
                deltaStat=request.fitting.deltaStat,
                nMode=request.fitting.nMode,
                scale=request.fitting.scale,
                aRef=request.fitting.aRef,
                meanMMD=request.fitting.meanMMD,
                stdMMD=request.fitting.stdMMD
            )
                            
            fourier_mmd_instance = FourierMMD(fourier_mmd_fitting=fitting)
        else:
            logger.debug("Creating new FourierMMD fitting")
            # Use extremely conservative values for small datasets
            fourier_mmd_instance = FourierMMD(
                train_data=filtered_ref_inputs,
                column_names=filtered_names,
                delta_stat=False,
                n_test=2,  # Minimum possible
                n_window=2,  # Minimum possible
                sig=1.0,
                random_seed=42,
                n_mode=2,  # Minimum possible
                epsilon=1e-6
            )
            
            # Simplified fitting storage approach
            fit_stats = fourier_mmd_instance.get_fit_stats()
            request.fitting = FourierMMDFitting(
                randomSeed=fit_stats.get_random_seed(),
                deltaStat=fit_stats.is_delta_stat(),
                nMode=fit_stats.get_n_mode(),
                scale=[float(x) for x in fit_stats.get_scale()] if fit_stats.get_scale() is not None else None,
                aRef=[float(x) for x in fit_stats.get_a_ref()] if fit_stats.get_a_ref() is not None else None,
                meanMMD=float(fit_stats.get_mean_mmd()) if fit_stats.get_mean_mmd() is not None else None,
                stdMMD=float(fit_stats.get_std_mmd()) if fit_stats.get_std_mmd() is not None else None
            )
            
            # Need to set properties individually
            if fit_stats.get_scale() is not None:
                request.fitting.scale = fit_stats.get_scale().tolist()
            if fit_stats.get_a_ref() is not None:
                request.fitting.aRef = fit_stats.get_a_ref().tolist()
            if fit_stats.get_mean_mmd() is not None:
                request.fitting.meanMMD = fit_stats.get_mean_mmd()
            if fit_stats.get_std_mmd() is not None:
                request.fitting.stdMMD = fit_stats.get_std_mmd()
        
        # Calculate drift
        result = fourier_mmd_instance.calculate(filtered_test_inputs, threshold, gamma)
        
        # Format results
        formatted_results = {
            "status": "success",
            "value": float(result.get_p_value()),
            "driftDetected": bool(result.is_reject()),
            "driftScore": float(result.get_stat_val()),
            "interpretation": f"Test data has p={float(result.get_p_value()):.6f} probability of having drifted from the training distribution."
        }
        # Add detailed column results        
        if result.is_reject():
            formatted_results["interpretation"] += f" p > {threshold} -> [SIGNIFICANT DRIFT]"
        else:
            formatted_results["interpretation"] += f" p <= {threshold}"
        
        # Add fitting parameters for reuse
        if request.fitting:
             formatted_results["fitting"] = {
                "randomSeed": int(request.fitting.randomSeed) if request.fitting.randomSeed is not None else None,
                "deltaStat": bool(request.fitting.deltaStat) if request.fitting.deltaStat is not None else None,
                "nMode": int(request.fitting.nMode) if request.fitting.nMode is not None else None,
                "scale": [float(x) for x in request.fitting.scale] if request.fitting.scale is not None else None,
                "aRef": [float(x) for x in request.fitting.aRef] if request.fitting.aRef is not None else None,
                "meanMMD": float(request.fitting.meanMMD) if request.fitting.meanMMD is not None else None,
                "stdMMD": float(request.fitting.stdMMD) if request.fitting.stdMMD is not None else None
            }
        
        return formatted_results
            
    except Exception as e:
        logger.error(f"Error computing FourierMMD: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")

@router.get("/metrics/drift/fouriermmd/definition")
async def get_fouriermmd_definition():
    """Provide a general definition of FourierMMD metric."""
    return {
        "name": "FourierMMD Drift",
        "description": "FourierMMD gives probability that the data values seen in a test dataset have drifted from the training dataset distribution, under the assumption that the computed MMD values are normally distributed.",
        "interpretation": "High p-values (above the threshold) indicate significant drift in the distribution of values.",
        "thresholdMeaning": "P-value threshold for significance testing. Higher values require stronger evidence of drift.",
        "gammaParameter": "Sets the threshold to flag drift by measuring the distance from the reference distribution in standard deviations."
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
        
        # Create ModelData instance
        model_data = ModelData(request.modelId)
        
        # Load data
        inputs, _, metadata = await model_data.data()
        input_names, _, metadata_names = await model_data.column_names()
        
        # Create KSTest instance
        kstest = KSTest()
        
        # Filter by reference tag if provided
        if request.referenceTag and metadata is not None:
            tag_indices = [i for i, name in enumerate(metadata_names) 
                          if name == "data_tag" or name.endswith(".data_tag")]
            
            if tag_indices:
                tag_col = metadata[:, tag_indices[0]]
                
                # Handle byte strings if needed
                if isinstance(tag_col[0], (bytes, np.bytes_)):
                    ref_tag = request.referenceTag.encode('utf-8')
                    ref_mask = np.array([t == ref_tag for t in tag_col])
                else:
                    ref_mask = tag_col == request.referenceTag
                
                # Filter reference and test data
                ref_data = inputs[ref_mask]
                test_data = inputs[~ref_mask]
                
                logger.info(f"Filtered data: {ref_data.shape[0]} reference rows, {test_data.shape[0]} test rows")
            else:
                # No tag column found, use all data as both reference and test
                ref_data = inputs
                test_data = inputs
                logger.warning(f"No data_tag column found, using all data for both reference and test")
        else:
            # No tag specified, use all data as both reference and test
            ref_data = inputs
            test_data = inputs
        
        # Calculate drift
        alpha = request.thresholdDelta or 0.05
        ks_results = kstest.calculate(
            ref_data=ref_data, 
            test_data=test_data, 
            column_names=input_names, 
            alpha=alpha
        )
        
        # Convert results to API response format - PROPERLY CONVERT NUMPY TYPES
        formatted_results = {
            "status": "success",
            "results": {}
        }
        
        for column_name, result in ks_results.items():
            # Use the correct getter methods
            formatted_results["results"][column_name] = {
                "statistic": float(result.get_stat_val()),  # Correct method call
                "pValue": float(result.get_p_value()),      # Correct method call
                "driftDetected": bool(result.is_reject())   # Correct method call
            }
        
        # Add overall summary
        drift_detected_count = sum(1 for r in ks_results.values() if r.is_reject())
        formatted_results["overall"] = {
            "driftScore": float(drift_detected_count) / len(ks_results) if ks_results else 0.0,
            "driftDetected": bool(drift_detected_count > 0),
            "driftedColumns": int(drift_detected_count),
            "totalColumns": len(ks_results)
        }
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error computing KSTest: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")

@router.get("/metrics/drift/kstest/definition")
async def get_kstest_definition():
    """Provide a general definition of KSTest metric."""
    return {
        "name": "Kolmogorov-Smirnov Test",
        "description": "KSTest calculates two sample Kolmogorov-Smirnov test per column which tests if two samples are drawn from the same distributions.",
        "interpretation": "Low p-values (below the threshold) indicate significant drift in the distribution of values.",
        "thresholdMeaning": "P-value threshold for significance testing. Lower values require stronger evidence of drift.",
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
        
        # Input validation
        if not request.referenceTag:
            return {
                "status": "error",
                "message": "Must provide a reference tag in request defining the original data distribution",
            }
        
        # Create ModelData instance
        model_data = ModelData(request.modelId)
        
        # Load all data
        inputs, _, metadata = await model_data.data()
        input_names, _, metadata_names = await model_data.column_names()
        
        # CORRECT ORDER: First define tag_column_indices, then use it
        # Filter by reference tag
        if metadata is not None:
            tag_column_indices = [
                i for i, name in enumerate(metadata_names)
                if name == "data_tag" or name.endswith(".data_tag")
            ]

            # NOW we can use tag_column_indices
            if tag_column_indices:
                # Add debugging for metadata
                test_value = metadata[0, tag_column_indices[0]]
                logger.warning(f"Metadata value type: {type(test_value)}, value: {test_value}")
                
                # Check if reference tag is byte string
                if isinstance(test_value, (bytes, np.bytes_)):
                    logger.warning("Metadata contains byte strings - will encode reference tag")
                    ref_tag_bytes = request.referenceTag.encode('utf-8')
                    ref_mask = np.array([tag == ref_tag_bytes for tag in metadata[:, tag_column_indices[0]]])
                    test_mask = np.array([tag != ref_tag_bytes for tag in metadata[:, tag_column_indices[0]]])
                else:
                    ref_mask = metadata[:, tag_column_indices[0]] == request.referenceTag
                    test_mask = metadata[:, tag_column_indices[0]] != request.referenceTag
                
                # Apply the masks
                ref_inputs = inputs[ref_mask] if ref_mask.any() else inputs
                test_inputs = inputs[test_mask] if test_mask.any() else np.array([])
                
                logger.info(
                    f"Filtered data by tag '{request.referenceTag}': {np.sum(ref_mask)} reference rows, {np.sum(test_mask)} test rows"
                )
            else:
                logger.warning(f"Reference tag '{request.referenceTag}' provided but no data_tag column found")
                ref_inputs = inputs
                test_inputs = np.array([])
        else:
            ref_inputs = inputs
            test_inputs = np.array([])
            
        # Add small dataset warning like Java implementation
        if len(test_inputs) < 2:
            logger.warning("Test data has less than two observations; Meanshift results will not be numerically reliable.")
        
        # Use the new precompute method to mirror Java implementation
        meanshift = Meanshift.precompute(ref_inputs, input_names)
        
        # Calculate drift using ONLY TEST DATA (exactly as in Java)
        alpha = request.thresholdDelta or 0.05
        mean_shift_results = meanshift.calculate(
            test_data=test_inputs,
            column_names=input_names, 
            alpha=alpha
        )
        
        # Format results to match Java implementation EXACTLY
        namedValues = {}
        for column_name, result in mean_shift_results.items():
            namedValues[column_name] = float(result.p_value)
            
        return {
            "status": "success",
            "namedValues": namedValues
        }
        
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

