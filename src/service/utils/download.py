import logging
import numbers
import pickle
from datetime import datetime
from typing import Any, List

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.service.data.storage import get_storage_interface

logger = logging.getLogger(__name__)


class RowMatcher(BaseModel):
    columnName: str
    operation: str
    values: List[Any]


class DataRequestPayload(BaseModel):
    modelId: str
    matchAny: List[RowMatcher] = Field(default_factory=list)
    matchAll: List[RowMatcher] = Field(default_factory=list)
    matchNone: List[RowMatcher] = Field(default_factory=list)


class DataResponsePayload(BaseModel):
    dataCSV: str


def get_storage() -> Any:
    """Get storage instance"""
    return get_storage_interface()


def apply_matcher(df: pd.DataFrame, matcher: RowMatcher, negate: bool = False) -> pd.DataFrame:
    """Apply a single matcher to the dataframe."""
    if matcher.operation not in ["EQUALS", "BETWEEN"]:
        raise HTTPException(
            status_code=400,
            detail="RowMatch operation must be one of [BETWEEN, EQUALS]",
        )
    if matcher.operation == "EQUALS":
        return apply_equals_matcher(df, matcher, negate)
    elif matcher.operation == "BETWEEN":
        return apply_between_matcher(df, matcher, negate)


def apply_equals_matcher(df: pd.DataFrame, matcher: RowMatcher, negate: bool = False) -> pd.DataFrame:
    """Apply EQUALS matcher to dataframe."""
    column_name = matcher.columnName
    values = matcher.values
    if column_name not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"No feature or output found with name={column_name}",
        )
    mask = df[column_name].isin(values)
    if negate:
        mask = ~mask
    return df[mask]


def apply_between_matcher(df: pd.DataFrame, matcher: RowMatcher, negate: bool = False) -> pd.DataFrame:
    """Apply BETWEEN matcher to dataframe."""
    column_name = matcher.columnName
    values = matcher.values

    if column_name not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"No feature or output found with name={column_name}",
        )
    errors = []
    if len(values) != 2:
        errors.append(
            f"BETWEEN operation must contain exactly two values, describing the lower and upper bounds of the desired range. Received {len(values)} values"
        )
    if column_name == "trustyai.TIMESTAMP":
        if errors:
            combined_error = ", ".join(errors)
            raise HTTPException(status_code=400, detail=combined_error)
        try:
            start_time = pd.to_datetime(str(values[0]))
            end_time = pd.to_datetime(str(values[1]))
            df_times = pd.to_datetime(df[column_name])
            mask = (df_times >= start_time) & (df_times < end_time)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Timestamp value is unparseable as an ISO_LOCAL_DATE_TIME: {str(e)}",
            )
    elif column_name == "trustyai.INDEX":
        if errors:
            combined_error = ", ".join(errors)
            raise HTTPException(status_code=400, detail=combined_error)
        min_val, max_val = sorted([int(v) for v in values])
        mask = (df[column_name] >= min_val) & (df[column_name] < max_val)
    else:
        if not all(isinstance(v, numbers.Number) for v in values):
            errors.append(
                "BETWEEN operation must only contain numbers, describing the lower and upper bounds of the desired range. Received non-numeric values"
            )
        if errors:
            combined_error = ", ".join(errors)
            raise HTTPException(status_code=400, detail=combined_error)
        min_val, max_val = sorted(values)
        try:
            numeric_column = pd.to_numeric(df[column_name], errors="raise")
            mask = (numeric_column >= min_val) & (numeric_column < max_val)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column_name}' contains non-numeric values that cannot be compared with BETWEEN operation. "
                       f"BETWEEN operation requires numeric data."
            )
    
    if negate:
        mask = ~mask
    return df[mask]


async def load_model_dataframe(model_id: str) -> pd.DataFrame:
    """Load model dataframe from storage."""
    storage = get_storage()
    try:
        input_data, input_cols = await storage.read_data(f"{model_id}_inputs")
        output_data, output_cols = await storage.read_data(f"{model_id}_outputs")
        metadata_data, metadata_cols = await storage.read_data(f"{model_id}_metadata")
        if input_data is None or output_data is None or metadata_data is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        df = pd.DataFrame()
        if len(input_data) > 0:
            if input_data.ndim == 2 and len(input_cols) == 1 and input_data.shape[1] > 1:
                col_name = input_cols[0]
                for j in range(input_data.shape[1]):
                    df[f"{col_name}_{j}"] = input_data[:, j]
            else:
                input_df = pd.DataFrame(input_data, columns=input_cols)
                for col in input_cols:
                    df[col] = input_df[col]
        if len(output_data) > 0:
            if output_data.ndim == 2 and len(output_cols) == 1 and output_data.shape[1] > 1:
                col_name = output_cols[0]
                for j in range(output_data.shape[1]):
                    df[f"{col_name}_{j}"] = output_data[:, j]
            else:
                if output_data.ndim == 2:
                    output_data = output_data.flatten()
                output_df = pd.DataFrame({output_cols[0]: output_data})
                for col in output_cols:
                    df[col] = output_df[col]
        if len(metadata_data) > 0 and isinstance(metadata_data[0], bytes):
            deserialized_metadata = []
            for row in metadata_data:
                deserialized_row = pickle.loads(row)
                deserialized_metadata.append(deserialized_row)
            metadata_df = pd.DataFrame(deserialized_metadata, columns=metadata_cols)
        else:
            metadata_df = pd.DataFrame(metadata_data, columns=metadata_cols)
        trusty_mapping = {
            "id": "trustyai.ID",
            "model_id": "trustyai.MODEL_ID",
            "timestamp": "trustyai.TIMESTAMP",
            "tag": "trustyai.TAG",
        }
        for orig_col in metadata_cols:
            trusty_col = trusty_mapping.get(orig_col.lower(), orig_col)
            df[trusty_col] = metadata_df[orig_col]
        df["trustyai.INDEX"] = range(len(df))
        return df
    except Exception as e:
        if "not found" in str(e).lower() or "MissingH5PYDataException" in str(type(e).__name__):
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        raise HTTPException(status_code=500, detail=f"Error loading model data: {str(e)}")