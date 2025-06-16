"""
Data upload utilities for TrustyAI service.

This module provides optimized data processing and validation for model data uploads,
including both regular data ingestion and ground truth handling.
"""

import logging
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException

from src.service.constants import (
    INPUT_SUFFIX,
    METADATA_SUFFIX,
    OUTPUT_SUFFIX,
    TRUSTYAI_TAG_PREFIX,
)
from src.service.data.modelmesh_parser import ModelMeshPayloadParser
from src.service.data.storage import get_storage_interface

logger = logging.getLogger(__name__)


async def process_upload_request(payload: Any) -> Dict[str, str]:
    """
    Process complete upload request with validation and data handling.
    """
    model_name = ModelMeshPayloadParser.standardize_model_id(payload.model_name)
    if payload.data_tag:
        error = validate_data_tag(payload.data_tag)
        if error:
            raise HTTPException(400, error)
    inputs = payload.request.get("inputs", [])
    outputs = payload.response.get("outputs", [])
    if not inputs or not outputs:
        raise HTTPException(400, "Missing input or output tensors")
    input_arrays, input_names, _, execution_ids = process_tensors(inputs)
    output_arrays, output_names, _, _ = process_tensors(outputs)
    error = validate_input_shapes(input_arrays, input_names)
    if error:
        raise HTTPException(400, f"One or more errors in input tensors: {error}")
    if payload.is_ground_truth:
        return await _process_ground_truth_data(
            model_name, input_arrays, input_names, output_arrays, output_names, execution_ids
        )
    else:
        return await _process_regular_data(
            model_name, input_arrays, input_names, output_arrays, output_names, execution_ids, payload.data_tag
        )


async def _process_ground_truth_data(
    model_name: str,
    input_arrays: List[np.ndarray],
    input_names: List[str],
    output_arrays: List[np.ndarray],
    output_names: List[str],
    execution_ids: Optional[List[str]],
) -> Dict[str, str]:
    """Process ground truth data upload."""
    if not execution_ids:
        raise HTTPException(400, "Ground truth requires execution IDs")
    result = await handle_ground_truths(
        model_name,
        input_arrays,
        input_names,
        output_arrays,
        output_names,
        [sanitize_id(id) for id in execution_ids],
    )
    if not result.success:
        raise HTTPException(400, result.message)
    result_data = result.data
    if result_data is None:
        raise HTTPException(500, "Ground truth processing failed")
    gt_name = f"{model_name}_ground_truth"
    storage_interface = get_storage_interface()
    await storage_interface.write_data(gt_name + OUTPUT_SUFFIX, result_data["outputs"], result_data["output_names"])
    await storage_interface.write_data(
        gt_name + METADATA_SUFFIX,
        result_data["metadata"],
        result_data["metadata_names"],
    )
    logger.info(f"Ground truth data saved for model: {model_name}")
    return {"message": result.message}


async def _process_regular_data(
    model_name: str,
    input_arrays: List[np.ndarray],
    input_names: List[str],
    output_arrays: List[np.ndarray],
    output_names: List[str],
    execution_ids: Optional[List[str]],
    data_tag: Optional[str],
) -> Dict[str, str]:
    """Process regular model data upload."""
    n_rows = input_arrays[0].shape[0]
    exec_ids = execution_ids or [str(uuid.uuid4()) for _ in range(n_rows)]
    input_data = _flatten_tensor_data(input_arrays, n_rows)
    output_data = _flatten_tensor_data(output_arrays, n_rows)
    metadata, metadata_cols = _create_metadata(exec_ids, model_name, data_tag)
    await save_model_data(
        model_name,
        np.array(input_data),
        input_names,
        np.array(output_data),
        output_names,
        metadata,
        metadata_cols,
    )
    logger.info(f"Regular data saved for model: {model_name}, rows: {n_rows}")
    return {"message": f"{n_rows} datapoints added to {model_name}"}


def _flatten_tensor_data(arrays: List[np.ndarray], n_rows: int) -> List[List[Any]]:
    """
    Flatten tensor arrays into row-based format for storage.
    """

    def flatten_row(arrays: List[np.ndarray], row: int) -> List[Any]:
        """Flatten arrays for a single row."""
        return [x for arr in arrays for x in (arr[row].flatten() if arr.ndim > 1 else [arr[row]])]

    return [flatten_row(arrays, i) for i in range(n_rows)]


def _create_metadata(
    execution_ids: List[str], model_name: str, data_tag: Optional[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Create metadata array for model data storage.
    """
    current_timestamp = datetime.now().isoformat()

    # FIX: Use UPPERCASE column names to match download.py expectations
    metadata_cols = ["ID", "MODEL_ID", "TIMESTAMP", "TAG"]  # ← Changed to uppercase

    metadata_rows = [
        [
            str(eid),
            str(model_name),
            str(current_timestamp),
            str(data_tag or ""),
        ]
        for eid in execution_ids
    ]

    metadata = np.array(metadata_rows, dtype="<U100")
    return metadata, metadata_cols


class ValidationError(Exception):
    """Validation errors."""

    pass


class ProcessingError(Exception):
    """Processing errors."""

    pass


@dataclass
class GroundTruthValidationResult:
    """Result of ground truth validation."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


DTYPE_MAP = {
    "INT64": np.int64,
    "INT32": np.int32,
    "FP32": np.float32,
    "FP64": np.float64,
    "BOOL": np.bool_,
}
TYPE_MAP = {
    np.int64: "Long",
    np.int32: "Integer",
    np.float32: "Float",
    np.float64: "Double",
    np.bool_: "Boolean",
    int: "Long",
    float: "Double",
    bool: "Boolean",
    str: "String",
}


def get_numpy_dtype(datatype: str) -> Optional[np.dtype[Any]]:
    """Get numpy dtype from string."""
    dtype_class = DTYPE_MAP.get(datatype)
    return np.dtype(dtype_class) if dtype_class else None


def get_type_name(val: Any) -> str:
    """Get Java-style type name for a value."""
    if hasattr(val, "dtype"):
        return TYPE_MAP.get(val.dtype.type, "String")
    return TYPE_MAP.get(type(val), "String")


def sanitize_id(execution_id: str) -> str:
    """Sanitize execution ID."""
    return str(execution_id).strip()


def extract_row_data(arrays: List[np.ndarray], row_index: int) -> List[Any]:
    """Extract data from arrays for a specific row."""
    row_data = []
    for arr in arrays:
        if arr.ndim > 1:
            row_data.extend(arr[row_index].flatten())
        else:
            row_data.append(arr[row_index])
    return row_data


def process_tensors(
    tensors: List[Dict[str, Any]],
) -> Tuple[List[np.ndarray], List[str], List[str], Optional[List[str]]]:
    """Process tensor data from payload."""
    if not tensors:
        return [], [], [], None
    arrays, names, datatypes = [], [], []
    execution_ids = None
    for i, tensor in enumerate(tensors):
        data = tensor.get("data", [])
        shape = tensor.get("shape", [])
        name = tensor.get("name", f"tensor_{i}")
        datatype = tensor.get("datatype", "INT64")
        if not data:
            raise ProcessingError(f"Tensor '{name}' has no data")
        dtype = get_numpy_dtype(datatype)
        try:
            if shape and dtype:
                arr = np.array(data, dtype=dtype).reshape(shape)
            elif dtype:
                arr = np.array(data, dtype=dtype)
            else:
                arr = np.array(data)
        except ValueError:
            arr = np.array(data, dtype=dtype) if dtype else np.array(data)
        arrays.append(arr)
        names.append(name)
        datatypes.append(datatype)
        if execution_ids is None:
            execution_ids = tensor.get("execution_ids")
    return arrays, names, datatypes, execution_ids


def validate_data_tag(tag: str) -> Optional[str]:
    """Validate data tag."""
    if not tag:
        return None
    if tag.startswith(TRUSTYAI_TAG_PREFIX):
        return (
            f"The tag prefix '{TRUSTYAI_TAG_PREFIX}' is reserved for internal TrustyAI use only. "
            f"Provided tag '{tag}' violates this restriction."
        )
    return None


def validate_input_shapes(input_arrays: List[np.ndarray], input_names: List[str]) -> Optional[str]:
    """Validate input array shapes and names."""
    if not input_arrays:
        return None
    errors = []
    if len(set(input_names)) != len(input_names):
        errors.append("Input tensors must have unique names")
    first_dim = input_arrays[0].shape[0]
    for i, arr in enumerate(input_arrays[1:], 1):
        if arr.shape[0] != first_dim:
            errors.append(
                f"Input tensor '{input_names[i]}' has first dimension {arr.shape[0]}, "
                f"which doesn't match the first dimension {first_dim} of '{input_names[0]}'"
            )
    if errors:
        return ". ".join(errors) + "."
    return None


class GroundTruthValidator:
    """Ground truth validator."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.id_to_row: Dict[str, int] = {}
        self.inputs: Optional[np.ndarray] = None
        self.outputs: Optional[np.ndarray] = None
        self.metadata: Optional[np.ndarray] = None

    async def initialize(self) -> None:
        """Load existing data."""
        storage_interface = get_storage_interface()
        self.inputs, _ = await storage_interface.read_data(self.model_name + INPUT_SUFFIX)
        self.outputs, _ = await storage_interface.read_data(self.model_name + OUTPUT_SUFFIX)
        self.metadata, _ = await storage_interface.read_data(self.model_name + METADATA_SUFFIX)
        metadata_cols = await storage_interface.get_original_column_names(self.model_name + METADATA_SUFFIX)
        id_col = next((i for i, name in enumerate(metadata_cols) if name.upper() == "ID"), 0)
        if self.metadata is not None:
            for j, row in enumerate(self.metadata):
                id_val = row[id_col]
                self.id_to_row[str(id_val)] = j

    def find_row(self, exec_id: str) -> Optional[int]:
        """Find row index for execution ID."""
        return self.id_to_row.get(str(exec_id))

    async def validate_data(
        self,
        exec_id: str,
        uploaded_inputs: List[Any],
        uploaded_outputs: List[Any],
        row_idx: int,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Validate inputs and outputs."""
        if self.inputs is None or self.outputs is None:
            return f"ID={exec_id} no existing data found"
        existing_inputs = self.inputs[row_idx]
        existing_outputs = self.outputs[row_idx]
        if len(existing_inputs) != len(uploaded_inputs):
            return f"ID={exec_id} input shapes do not match. Observed inputs have length={len(existing_inputs)} while uploaded inputs have length={len(uploaded_inputs)}"
        for i, (existing, uploaded) in enumerate(zip(existing_inputs, uploaded_inputs)):
            existing_type = get_type_name(existing)
            uploaded_type = get_type_name(uploaded)
            if existing_type != uploaded_type:
                return f"ID={exec_id} input type mismatch at position {i + 1}: Class={existing_type} != Class={uploaded_type}"
            if existing != uploaded:
                return f"ID={exec_id} inputs are not identical: value mismatch at position {i + 1}"
        if len(existing_outputs) != len(uploaded_outputs):
            return f"ID={exec_id} output shapes do not match. Observed outputs have length={len(existing_outputs)} while uploaded ground-truths have length={len(uploaded_outputs)}"
        for i, (existing, uploaded) in enumerate(zip(existing_outputs, uploaded_outputs)):
            existing_type = get_type_name(existing)
            uploaded_type = get_type_name(uploaded)
            if existing_type != uploaded_type:
                return f"ID={exec_id} output type mismatch at position {i + 1}: Class={existing_type} != Class={uploaded_type}"
        if output_names:
            try:
                storage_interface = get_storage_interface()
                stored_output_names = await storage_interface.get_original_column_names(self.model_name + OUTPUT_SUFFIX)
                if len(stored_output_names) != len(output_names):
                    return (
                        f"ID={exec_id} output name count mismatch. "
                        f"Expected {len(stored_output_names)} names, "
                        f"got {len(output_names)} names."
                    )
                for i, (stored_name, uploaded_name) in enumerate(zip(stored_output_names, output_names)):
                    if stored_name != uploaded_name:
                        return (
                            f"ID={exec_id} output names do not match: "
                            f"position {i + 1}: {stored_name} != {uploaded_name}"
                        )
            except Exception as e:
                logger.warning(f"Could not validate output names for {exec_id}: {e}")
        if input_names:
            try:
                storage_interface = get_storage_interface()
                stored_input_names = await storage_interface.get_original_column_names(self.model_name + INPUT_SUFFIX)
                if len(stored_input_names) != len(input_names):
                    return (
                        f"ID={exec_id} input name count mismatch. "
                        f"Expected {len(stored_input_names)} names, "
                        f"got {len(input_names)} names."
                    )
                for i, (stored_name, uploaded_name) in enumerate(zip(stored_input_names, input_names)):
                    if stored_name != uploaded_name:
                        return (
                            f"ID={exec_id} input names do not match: position {i + 1}: {stored_name} != {uploaded_name}"
                        )
            except Exception as e:
                logger.warning(f"Could not validate input names for {exec_id}: {e}")
        return None


async def handle_ground_truths(
    model_name: str,
    input_arrays: List[np.ndarray],
    input_names: List[str],
    output_arrays: List[np.ndarray],
    output_names: List[str],
    execution_ids: List[str],
    config: Optional[Any] = None,
) -> GroundTruthValidationResult:
    """Handle ground truth validation."""
    if not execution_ids:
        return GroundTruthValidationResult(success=False, message="No execution IDs provided.")
    storage_interface = get_storage_interface()
    if not await storage_interface.dataset_exists(model_name + INPUT_SUFFIX):
        return GroundTruthValidationResult(success=False, message=f"Model {model_name} not found.")
    validator = GroundTruthValidator(model_name)
    await validator.initialize()
    errors = []
    valid_outputs = []
    valid_metadata = []
    n_rows = input_arrays[0].shape[0] if input_arrays else 0
    for i, exec_id in enumerate(execution_ids):
        if i >= n_rows:
            errors.append(f"ID={exec_id} index out of bounds")
            continue
        row_idx = validator.find_row(exec_id)
        if row_idx is None:
            errors.append(f"ID={exec_id} not found")
            continue
        uploaded_inputs = extract_row_data(input_arrays, i)
        uploaded_outputs = extract_row_data(output_arrays, i)
        error = await validator.validate_data(exec_id, uploaded_inputs, uploaded_outputs, row_idx)
        if error:
            errors.append(error)
            continue
        valid_outputs.append(uploaded_outputs)
        valid_metadata.append([exec_id])
    if errors:
        return GroundTruthValidationResult(
            success=False,
            message="Found fatal mismatches between uploaded data and recorded inference data:\n"
            + "\n".join(errors[:5]),
            errors=errors,
        )
    if not valid_outputs:
        return GroundTruthValidationResult(success=False, message="No valid ground truths found.")
    return GroundTruthValidationResult(
        success=True,
        message=f"{len(valid_outputs)} ground truths added.",
        data={
            "outputs": np.array(valid_outputs),
            "output_names": output_names,
            "metadata": np.array(valid_metadata),
            "metadata_names": ["ID"],  # Note: lowercase here for ground truth metadata
        },
    )


async def save_model_data(
    model_name: str,
    input_data: np.ndarray,
    input_names: List[str],
    output_data: np.ndarray,
    output_names: List[str],
    metadata_data: np.ndarray,
    metadata_names: List[str],
) -> Dict[str, Any]:
    """Save model data to storage."""
    storage_interface = get_storage_interface()
    await storage_interface.write_data(model_name + INPUT_SUFFIX, input_data, input_names)
    await storage_interface.write_data(model_name + OUTPUT_SUFFIX, output_data, output_names)
    await storage_interface.write_data(model_name + METADATA_SUFFIX, metadata_data, metadata_names)
    logger.info(f"Saved model data for {model_name}: {len(input_data)} rows")
    return {
        "model_name": model_name,
        "rows": len(input_data),
    }
