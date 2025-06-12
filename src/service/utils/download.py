import logging
import numbers
import pickle
from datetime import datetime
from typing import Any, List, Optional

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
    matchAny: Optional[List[RowMatcher]] = Field(default_factory=list)
    matchAll: Optional[List[RowMatcher]] = Field(default_factory=list)
    matchNone: Optional[List[RowMatcher]] = Field(default_factory=list)


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
        if not all(isinstance(v, numbers.Number) and not isinstance(v, bool) for v in values):
            errors.append(
                "BETWEEN operation must only contain numbers, describing the lower and upper bounds of the desired range. Received non-numeric values"
            )
        if errors:
            combined_error = ", ".join(errors)
            raise HTTPException(status_code=400, detail=combined_error)
        min_val, max_val = sorted(values)
        # try to convert column to numeric if it is not already
        numeric_column = pd.to_numeric(df[column_name], errors="coerce")
        # check if conversion was successful
        if numeric_column.isna().any():
            # Some values couldn't be converted to numeric
            unique_non_numeric = df[column_name][numeric_column.isna()].unique()
            total_count = len(unique_non_numeric)
            examples = list(unique_non_numeric[:5])

            raise HTTPException(
                status_code=400,
                detail=f"BETWEEN operation requires numeric column data, but column '{column_name}' contains {total_count} different non-numeric values. Examples: {examples}",
            )

        # All values are numeric, proceed with comparison
        mask = (numeric_column >= min_val) & (numeric_column < max_val)

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
        logger.info(f"Input data shape: {input_data.shape if hasattr(input_data, 'shape') else 'No shape'}")
        logger.info(f"Input columns: {input_cols}")
        logger.info(f"Output data shape: {output_data.shape if hasattr(output_data, 'shape') else 'No shape'}")
        logger.info(f"Output columns: {output_cols}")
        logger.info(f"Metadata data shape: {metadata_data.shape if hasattr(metadata_data, 'shape') else 'No shape'}")
        logger.info(f"Metadata columns: {metadata_cols}")
        if input_data is None or output_data is None or metadata_data is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        # Handle the case where input_data has multiple columns but single column name
        if hasattr(input_data, "shape") and len(input_data.shape) > 1:
            data_cols = input_data.shape[1]
            if len(input_cols) == 1 and data_cols > 1:
                # Expand single column name to match data columns
                base_name = input_cols[0]
                input_cols = [f"{base_name}_{i}" for i in range(data_cols)]
                logger.info(f"Expanded input columns to: {input_cols}")
        # Create DataFrames
        input_df = pd.DataFrame(input_data, columns=input_cols)
        output_df = pd.DataFrame(output_data, columns=output_cols)
        metadata_df = pd.DataFrame(metadata_data, columns=metadata_cols)
        # Map metadata columns to trustyai.* format (like Java does)
        metadata_column_mapping = {
            "id": "trustyai.ID",
            "model_id": "trustyai.MODEL_ID",
            "timestamp": "trustyai.TIMESTAMP",
            "tag": "trustyai.TAG",
        }
        metadata_df = metadata_df.rename(columns=metadata_column_mapping)
        combined_df = pd.concat([metadata_df, input_df, output_df], axis=1)
        combined_df["trustyai.INDEX"] = combined_df.index
        return combined_df

    except Exception as e:
        logger.error(f"Error loading model data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model data: {str(e)}")
