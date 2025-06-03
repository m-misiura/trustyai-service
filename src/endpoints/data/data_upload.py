import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.service.constants import METADATA_SUFFIX, OUTPUT_SUFFIX
from src.service.data.modelmesh_parser import ModelMeshPayloadParser
from src.service.data.storage import get_storage_interface
from src.service.utils.upload import (
    handle_ground_truths,
    process_tensors,
    sanitize_id,
    save_model_data,
    validate_data_tag,
    validate_input_shapes,
)

router = APIRouter()
storage = get_storage_interface()


class UploadPayload(BaseModel):
    model_name: str
    data_tag: Optional[str] = None
    is_ground_truth: bool = False
    request: Dict[str, Any]
    response: Dict[str, Any]


@router.post("/data/upload")
async def upload(payload: UploadPayload) -> Dict[str, str]:
    """Upload model data - regular or ground truth."""
    storage = get_storage_interface()
    print(f"Storage directory: {storage.data_directory}")
    print(f"Storage class: {type(storage).__name__}")
    
    model_name = ModelMeshPayloadParser.standardize_model_id(payload.model_name)
    print(f"Model name: {model_name}")
    print(f"Is ground truth: {payload.is_ground_truth}")
    
    if payload.data_tag and (error := validate_data_tag(payload.data_tag)):
        raise HTTPException(400, error)
    inputs = payload.request.get("inputs", [])
    outputs = payload.response.get("outputs", [])
    if not inputs or not outputs:
        raise HTTPException(400, "Missing input or output tensors")
    print(f"Processing {len(inputs)} inputs, {len(outputs)} outputs")
   
    input_arrays, input_names, _, execution_ids = process_tensors(inputs)
    output_arrays, output_names, _, _ = process_tensors(outputs)
    print(f"Processed arrays - inputs shape: {[arr.shape for arr in input_arrays]}")
    print(f"Processed arrays - outputs shape: {[arr.shape for arr in output_arrays]}")
    
    if error := validate_input_shapes(input_arrays, input_names):
        raise HTTPException(400, f"One or more errors in input tensors: {error}")
    if payload.is_ground_truth:
        print("=== GROUND TRUTH PATH ===")
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
        await storage.write_data(gt_name + OUTPUT_SUFFIX, result_data["outputs"], result_data["output_names"])
        await storage.write_data(
            gt_name + METADATA_SUFFIX,
            result_data["metadata"],
            result_data["metadata_names"],
        )
        return {"message": result.message}
    else:
        print("=== REGULAR DATA PATH ===")
        n_rows = input_arrays[0].shape[0]
        exec_ids = execution_ids or [str(uuid.uuid4()) for _ in range(n_rows)]
        print(f"Number of rows: {n_rows}")
        def flatten(arrays: List[np.ndarray], row: int) -> List[Any]:
            return [x for arr in arrays for x in (arr[row].flatten() if arr.ndim > 1 else [arr[row]])]

        input_data = [flatten(input_arrays, i) for i in range(n_rows)]
        output_data = [flatten(output_arrays, i) for i in range(n_rows)]
        cols = ["id", "model_id", "timestamp", "tag"]
        current_timestamp = datetime.now().isoformat()
        metadata_rows = [
            [
                str(eid),
                str(model_name),
                str(current_timestamp),
                str(payload.data_tag or ""),
            ]
            for eid in exec_ids
        ]
        metadata = np.array(metadata_rows, dtype="<U100")
        print(f"About to call save_model_data with {len(input_data)} input rows, {len(output_data)} output rows")
        print(f"Storage directory check: {os.path.exists(storage.data_directory)}")
        
        
        try:
            await save_model_data(
                model_name,
                np.array(input_data),
                input_names,
                np.array(output_data),
                output_names,
                metadata,
                cols,
            )
            print("=== save_model_data completed without exception ===")
            input_file = storage._get_filename(model_name + "_inputs")
            output_file = storage._get_filename(model_name + "_outputs")
            print(f"Input file created: {os.path.exists(input_file)}")
            print(f"Output file created: {os.path.exists(output_file)}")
            if os.path.exists(input_file):
                print(f"Input file size: {os.path.getsize(input_file)} bytes")
            if os.path.exists(output_file):
                print(f"Output file size: {os.path.getsize(output_file)} bytes")
                
        except Exception as e:
            print(f"=== save_model_data FAILED: {e} ===")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(500, f"Failed to save data: {str(e)}")
        
        return {"message": f"{n_rows} datapoints added to {model_name}"}
