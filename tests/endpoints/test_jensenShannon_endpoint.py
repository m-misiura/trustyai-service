import os
import shutil
from dotenv import load_dotenv
from typing import Generator
load_dotenv()
import pytest
from fastapi.testclient import TestClient

from src.main import app


TEST_DATA_DIR = "test_data"
os.makedirs(TEST_DATA_DIR, exist_ok=True)

def cleanup_test_data() -> None:
    """clean up all test data."""
    print(f"\nCleaning up test data in {TEST_DATA_DIR}")
    
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
        print(f"Removed directory: {TEST_DATA_DIR}")
    
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    print(f"Recreated directory: {TEST_DATA_DIR}")

@pytest.fixture(autouse=True)
def cleanup() -> Generator[None, None, None]:
    """Clean up before and after each test."""
    cleanup_test_data()
    yield
    cleanup_test_data()

@pytest.fixture
def e2e_client() -> Generator[TestClient, None, None]:
    """Create a test client for end-to-end testing."""
    yield TestClient(app)

def test_end_to_end_upload_and_jensenShannon(e2e_client: TestClient) -> None:
    """End-to-end test: Upload data and compute Jensen-Shannon drift."""
    model_id = "demo-model-end2end"

    # Upload reference data
    reference_payload = {
        "model_name": model_id,
        "data_tag": "reference",
        "is_ground_truth": False,
        "request": {
            "inputs": [
                {
                    "name": "credit_inputs",
                    "shape": [5, 3],
                    "datatype": "FP32",
                    "data": [
                        [0.1, 0.2, 0.3],
                        [0.2, 0.3, 0.4],
                        [0.3, 0.4, 0.5],
                        [0.4, 0.5, 0.6],
                        [0.5, 0.6, 0.7]
                    ]
                }
            ]
        },
        "response": {
            "model_name": "demo-model-end2end",
            "model_version": "1",
            "outputs": [
                {
                    "name": "output",
                    "datatype": "FP32",
                    "shape": [5, 1],
                    "data": [1, 1, 1, 1, 1]
                }
            ]
        }
    }

    response_ref = e2e_client.post("/data/upload", json=reference_payload)
    assert response_ref.status_code == 200, f"Upload failed: {response_ref.json()}"

    # Upload test data
    test_payload = {
        "model_name": model_id,
        "data_tag": "test",
        "is_ground_truth": False,
        "request": {
            "inputs": [
                {
                    "name": "credit_inputs",
                    "shape": [5, 3],
                    "datatype": "FP32",
                    "data": [
                        [0.3, 0.4, 0.5],
                        [0.4, 0.5, 0.6],
                        [0.5, 0.6, 0.7],
                        [0.6, 0.7, 0.8],
                        [0.7, 0.8, 0.9]
                    ]
                }
            ]
        },
        "response": {
            "model_name": "demo-model-end2end",
            "model_version": "1",
            "outputs": [
                {
                    "name": "output",
                    "datatype": "FP32",
                    "shape": [5, 1],
                    "data": [1, 1, 1, 1, 1]
                }
            ]
        }
    }

    response_test = e2e_client.post("/data/upload", json=test_payload)
    assert response_test.status_code == 200, f"Test upload failed: {response_test.json()}"

    drift_payload = {
        "modelId": model_id,
        "referenceTag": "reference",
        "normalizeValues": True,
        "usePerChannel": True,
        "numCV": 1,
        "cvSize": 2  
    }

    response_drift = e2e_client.post("/metrics/drift/jensenshannon", json=drift_payload)
    assert response_drift.status_code == 200, f"Drift request failed: {response_drift.json()}"
    result = response_drift.json()
    print("Drift response:", result)
    assert isinstance(result["Result"], list)
    assert len(result["Result"]) == 3  # One result per feature
    for channel_result in result["Result"]:
        assert "columnName" in channel_result
        assert "js_stat" in channel_result
        assert "threshold" in channel_result
        assert "driftDetected" in channel_result 