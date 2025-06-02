import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.service.utils.download import (
    DataRequestPayload,
    DataResponsePayload,
    apply_matcher,
    load_model_dataframe,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/data/download")
async def download_data(payload: DataRequestPayload) -> DataResponsePayload:
    """Download model data with filtering."""
    try:
        logger.info(f"Received data download request for model: {payload.modelId}")

        # Load the dataframe
        df = await load_model_dataframe(payload.modelId)

        if df.empty:
            return DataResponsePayload(dataCSV="")
        # Apply matchAll filters (AND logic)
        if payload.matchAll:
            for matcher in payload.matchAll:
                df = apply_matcher(df, matcher, negate=False)
        # Apply matchNone filters (NOT logic)
        if payload.matchNone:
            for matcher in payload.matchNone:
                df = apply_matcher(df, matcher, negate=True)
        base_df = df.copy()
        # Apply matchAny filters (OR logic)
        if payload.matchAny:
            matching_dfs = []
            for matcher in payload.matchAny:
                matched_df = apply_matcher(base_df, matcher, negate=False)
                if not matched_df.empty:
                    matching_dfs.append(matched_df)
            # Union all results
            if matching_dfs:
                df = pd.concat(matching_dfs, ignore_index=True).drop_duplicates()
            else:
                # No matches found, return empty dataframe with same columns
                df = pd.DataFrame(columns=df.columns)
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        return DataResponsePayload(dataCSV=csv_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading data: {str(e)}")
