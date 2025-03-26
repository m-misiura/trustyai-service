import os
from src.service.data.storage.pvc import PVCStorage

def get_storage_interface():
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT")
    if storage_format == "PVC":
        data_dir = os.environ.get("STORAGE_DATA_FOLDER", "/tmp/trustyai-data")
        data_file = os.environ.get("STORAGE_DATA_FILENAME", "trustyai_inference_data.hdf5")
        
        # Create the directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        return PVCStorage(data_directory=data_dir, data_file=data_file)
    else:
        raise ValueError(f"Storage format={storage_format} not yet supported by the Python implementation of the service.")
