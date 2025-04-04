import pytest
import numpy as np
import csv
import os
import io
import random
from pathlib import Path
from typing import List, Tuple, Optional

from src.service.metrics.drift.fourier_mmd.fourier_mmd import FourierMMD

class TestFourierMMD:
    """Test the FourierMMD drift detection implementation."""
    
    train_dataset_filename = "train_ts_x.csv"
    valid_dataset_filename = "valid_ts_x.csv"
    test_dataset_filename = "test_ts_x.csv"
    
    train_df = None
    valid_df = None
    test_df = None
    
    @pytest.fixture(autouse=True)
    def setup(self, request):
        """Setup method called before each test."""
        try:
            self.train_df = self.read_csv(self.train_dataset_filename)
            self.valid_df = self.read_csv(self.valid_dataset_filename)
            self.test_df = self.read_csv(self.test_dataset_filename)
        except FileNotFoundError:
            # Create a marker to skip tests that require data files if they're not found
            marker = pytest.mark.skip(reason="Test data files not found")
            request.node.add_marker(marker)
    
    @staticmethod
    def generate_random_dataframe(observations: int, feature_diversity: int) -> np.ndarray:
        """
        Generate random test data.
        
        Args:
            observations: Number of observations
            feature_diversity: Range of diversity for age feature
        
        Returns:
            Numpy array with random data
        """
        # Initialize random generator with same seed as Java
        random.seed(0)
        
        # Create a dataframe with 3 columns: age, gender, race
        data = np.zeros((observations, 3))
        
        for i in range(observations):
            # Guarantee feature diversity for age
            data[i, 0] = i % feature_diversity
            # Random gender (0 or 1)
            data[i, 1] = 1 if random.random() > 0.5 else 0
            # Random race (0 or 1)
            data[i, 2] = 1 if random.random() > 0.5 else 0
        
        return data
    
    @staticmethod
    def generate_random_dataframe_drifted(observations: int, feature_diversity: int) -> np.ndarray:
        """
        Generate random test data with drift.
        
        Args:
            observations: Number of observations
            feature_diversity: Range of diversity for age feature
        
        Returns:
            Numpy array with drifted random data
        """
        # Initialize random generator with same seed as Java
        random.seed(0)
        
        # Create a dataframe with 3 columns: age, gender, race
        data = np.zeros((observations, 3))
        
        for i in range(observations):
            # Drifted age: (i % feature_diversity) + feature_diversity
            data[i, 0] = (i % feature_diversity) + feature_diversity
            # Always 0 for gender (drift)
            data[i, 1] = 0
            # Random race (0 or 1)
            data[i, 2] = 1 if random.random() > 0.5 else 0
        
        return data
    
    def read_csv(self, filename: str) -> np.ndarray:
        """
        Read test data from CSV file.
        
        Args:
            filename: Name of the CSV file
        
        Returns:
            Numpy array with data from CSV
        """
        # Get the resource path
        test_dir = Path(__file__).parent
        resource_path = test_dir / "data" / filename
        if not resource_path.exists():
            raise FileNotFoundError(f"Test data file {filename} not found in {test_dir}/data")
    
        
        # Read the CSV file
        with open(resource_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            
            data = []
            for row in reader:
                # Extract numerical features X1-X10 (indices 2-11)
                features = [float(row[i]) for i in range(2, 12)]
                data.append(features)
        
        return np.array(data)
    
    def test_valid_data(self):
        """Test valid data should not show drift."""
        delta_stat = True
        n_test = 100
        n_window = 168
        sig = 10.0
        random_seed = 1234
        n_mode = 512
        epsilon = 1.0e-7
        
        # Column names X1-X10
        column_names = [f"X{i}" for i in range(1, 11)]
        
        # Create FourierMMD
        fourier_mmd = FourierMMD(
            train_data=self.train_df,
            column_names=column_names,
            delta_stat=delta_stat,
            n_test=n_test,
            n_window=n_window,
            sig=sig,
            random_seed=random_seed,
            n_mode=n_mode,
            epsilon=epsilon
        )
        
        threshold = 0.8
        gamma = 1.5
        
        # Calculate drift
        drift = fourier_mmd.calculate(self.valid_df, threshold, gamma)
        
        # Assert not drifted
        assert drift.is_reject() == False, "drifted flag is true"
        assert drift.get_p_value() < 1.0, "drift.pValue >= 1.0"
    
    def test_production_data(self):
        """Test production data should show drift."""
        delta_stat = True
        n_test = 100
        n_window = 168
        sig = 10.0
        random_seed = 1234
        n_mode = 512
        epsilon = 1.0e-7
        
        # Column names X1-X10
        column_names = [f"X{i}" for i in range(1, 11)]
        
        # Create FourierMMD
        fourier_mmd = FourierMMD(
            train_data=self.train_df,
            column_names=column_names,
            delta_stat=delta_stat,
            n_test=n_test,
            n_window=n_window,
            sig=sig,
            random_seed=random_seed,
            n_mode=n_mode,
            epsilon=epsilon
        )
        
        threshold = 0.8
        gamma = 1.5
        
        # Calculate drift
        drift = fourier_mmd.calculate(self.test_df, threshold, gamma)
        
        # Assert drifted
        assert drift.is_reject() == True, "drifted flag is false"
        assert drift.get_p_value() >= 1.0, "drift.pValue < 1.0"
    
    def test_random_data(self):
        """Test random data should show drift."""
        # Generate random training data
        train_tab_x = self.generate_random_dataframe(100, 100)
        
        delta_stat = False
        n_test = 100
        n_window = 20
        sig = 10.0
        random_seed = 1234
        n_mode = 512
        epsilon = 1.0e-7
        
        # Column names
        column_names = ["age", "gender", "race"]
        
        # Create FourierMMD
        fourier_mmd = FourierMMD(
            train_data=train_tab_x,
            column_names=column_names,
            delta_stat=delta_stat,
            n_test=n_test,
            n_window=n_window,
            sig=sig,
            random_seed=random_seed,
            n_mode=n_mode,
            epsilon=epsilon
        )
        
        # Generate drifted test data
        test_tab_x = self.generate_random_dataframe_drifted(100, 100)
        
        threshold = 0.8
        gamma = 4.0
        
        # Calculate drift
        drift = fourier_mmd.calculate(test_tab_x, threshold, gamma)
        
        # Assert drifted
        assert drift.is_reject() == True, "drifted flag is false"
        assert drift.get_p_value() >= 1.0, "drift.pValue < 1.0"