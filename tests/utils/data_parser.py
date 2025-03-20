import numpy as np
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

class PregeneratedNormalData:
    """Helper class for test data generation and loading."""
    
    # Cache for loaded data
    _data = None
    _t_stat_table = None
    _p_value_table = None
    
    @classmethod
    def load_java_array_file(cls, file_path: Union[str, Path]) -> np.ndarray:
        """Load Java-style array from text file."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove all whitespace and newlines
        content = re.sub(r'\s+', '', content)
        
        # Remove outer braces if present
        if content.startswith('{') and content.endswith('}'):
            content = content[1:-1]
        
        # Split into rows
        rows = []
        current = []
        open_braces = 0
        buffer = ""
        
        for char in content:
            if char == '{':
                open_braces += 1
                if open_braces == 1:
                    continue
            elif char == '}':
                open_braces -= 1
                if open_braces == 0:
                    if buffer:
                        current.append(float(buffer))
                        buffer = ""
                    rows.append(current)
                    current = []
                    continue
            elif char == ',' and open_braces == 1:
                if buffer:
                    current.append(float(buffer))
                    buffer = ""
                continue
            
            if open_braces >= 1:
                buffer += char
        
        return np.array(rows)
    
    @classmethod
    def get_data(cls, regenerate=False) -> np.ndarray:
        """Get pregenerated normal distribution data."""
        if cls._data is None or regenerate:
            data_path = Path("tests/data/normal_data.npy")
            if data_path.exists() and not regenerate:
                cls._data = np.load(data_path)
            else:
                # Generate data similar to Java implementation
                cls._data = cls._generate_data()
                # Save for future use
                np.save(data_path, cls._data)
                
        return cls._data
    
    @classmethod
    def get_t_stat_table(cls) -> np.ndarray:
        """Get t-statistic table."""
        if cls._t_stat_table is None:
            # Try loading from .npy file first (faster)
            npy_path = Path("tests/data/t_stat_table.npy")
            if npy_path.exists():
                cls._t_stat_table = np.load(npy_path)
            else:
                # If not found, load from text file
                txt_path = Path("tests/data/t_stat_table.txt")
                if txt_path.exists():
                    cls._t_stat_table = cls.load_java_array_file(txt_path)
                    # Save as numpy array for faster loading next time
                    np.save(npy_path, cls._t_stat_table)
                else:
                    raise FileNotFoundError(f"T-stat table file not found at {txt_path}")
                
        return cls._t_stat_table
    
    @classmethod
    def get_p_value_table(cls) -> np.ndarray:
        """Get p-value table."""
        if cls._p_value_table is None:
            # Try loading from .npy file first (faster)
            npy_path = Path("tests/data/p_value_table.npy")
            if npy_path.exists():
                cls._p_value_table = np.load(npy_path)
            else:
                # If not found, load from text file
                txt_path = Path("tests/data/p_value_table.txt")
                if txt_path.exists():
                    cls._p_value_table = cls.load_java_array_file(txt_path)
                    # Save as numpy array for faster loading next time
                    np.save(npy_path, cls._p_value_table)
                else:
                    raise FileNotFoundError(f"P-value table file not found at {txt_path}")
                
        return cls._p_value_table
    
    @staticmethod
    def _generate_data() -> np.ndarray:
        """Generate synthetic normal distribution data."""
        np.random.seed(42)  # For reproducibility
        
        # Generate data with the same dimensions as the Java test
        n_distributions = 5
        n_cols = 25  # Match the Java test's dimension
        n_rows = 100
        
        data = np.zeros((n_cols, n_rows))
        
        for i in range(n_cols):
            # Use a deterministic pattern for mean and std
            dist_idx = i % n_distributions
            mean = (dist_idx - 2) * 5  # Means: -10, -5, 0, 5, 10
            std = 1 + 0.5 * dist_idx   # Std devs: 1.0, 1.5, 2.0, 2.5, 3.0
            
            data[i] = np.random.normal(mean, std, n_rows)
            
        return data
    
    @classmethod
    def generate(cls, idx: Union[int, List[int]], with_text: bool = False, 
                prefix: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Generate test data with specified columns."""
        data = cls.get_data()
        
        if isinstance(idx, int):
            # Single column case
            selected_data = data[idx].reshape(-1, 1)
            col_names = ["column_0"]  # Use more distinct name
        else:
            # Multi-column case
            selected_data = np.column_stack([data[i] for i in idx])
            col_names = [f"column_{i}" for i in range(len(idx))]  # More distinct names
        
        # Add text column if requested
        if with_text:
            text_col = np.array(['text'] * len(selected_data), dtype=object)
            selected_data = np.column_stack([selected_data, text_col])
            col_names.append("text_column")
        
        # Add prefix to column names if specified
        if prefix:
            col_names = [f"{prefix}_{name}" for name in col_names]
        
        return selected_data, col_names