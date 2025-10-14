"""
Simplified data loader that checks for missing data without filling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import data_config as config

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

class DataLoader:
    """
    Centralized data loader for both Binary and Dual Continuous CQL models
    """
    
    def __init__(self, data_path: str = config.DATA_PATH, verbose: bool = True, random_seed: int = config.RANDOM_SEED):
        """
        Initialize data loader
        
        Args:
            data_path: Path to the CSV data file
            verbose: Whether to print detailed information
            random_seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.verbose = verbose
        self.random_seed = random_seed
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Set seeds for this instance
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            Loaded DataFrame
        """
        if self.verbose:
            print(f"Loading data from {self.data_path}...")
        
        self.data = pd.read_csv(self.data_path)
        
        if self.verbose:
            print(f"Loaded {len(self.data)} records")
            print(f"Number of patients: {self.data[config.PATIENT_ID_COL].nunique()}")
            print(f"Columns found: {len(self.data.columns)} columns")
        
        # Sort by patient and time
        self.data = self.data.sort_values([config.PATIENT_ID_COL, config.TIME_COL])
        
        return self.data
    
    def check_missing_data(self, features: List[str]) -> bool:
        """
        Simple check for missing data - just reports what's missing
        
        Args:
            features: List of feature columns to check
            
        Returns:
            True if no missing data, False otherwise
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("MISSING DATA CHECK")
        print("="*60)
        
        all_good = True
        not_found = []
        has_nulls = []
        
        for feature in features:
            if feature not in self.data.columns:
                print(f"Feature '{feature}' NOT FOUND in data")
                not_found.append(feature)
                all_good = False
            else:
                missing_count = self.data[feature].isnull().sum()
                if missing_count > 0:
                    pct = (missing_count / len(self.data)) * 100
                    print(f" {feature}: {missing_count} missing ({pct:.2f}%)")
                    has_nulls.append((feature, missing_count, pct))
                    all_good = False
                else:
                    if self.verbose:
                        print(f"✓  {feature}: OK")
        
        # Summary
        print("-"*60)
        if not_found:
            print(f"\n{len(not_found)} features NOT FOUND:")
            for f in not_found:
                print(f"   - {f}")
        
        if has_nulls:
            print(f"\n{len(has_nulls)} features have MISSING DATA:")
            for f, count, pct in has_nulls:
                print(f"   - {f}: {count} missing ({pct:.2f}%)")
        
        if all_good:
            print("\n All features present with no missing data!")
        else:
            print("\n Data issues found - cannot proceed with training")
        
        return all_good
    
    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical features (gender, ethnicity) to numbers
        Ensures reproducibility by sorting unique values before encoding
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        for feature in config.CATEGORICAL_FEATURES:
            if feature in self.data.columns:
                if feature not in self.label_encoders:
                    # Convert to string first to handle any mixed types
                    str_values = self.data[feature].astype(str)
                    
                    # Get unique values and sort them for reproducibility
                    unique_values = sorted(str_values.unique())
                    
                    # Create mapping dictionary
                    value_to_int = {val: i for i, val in enumerate(unique_values)}
                    
                    # Apply mapping
                    self.data[feature] = str_values.map(value_to_int)
                    
                    # Store the mapping for potential inverse transform
                    self.label_encoders[feature] = {
                        'classes': unique_values,
                        'mapping': value_to_int
                    }
                    
                    if self.verbose:
                        print(f"Encoded {feature}: {unique_values} → {list(range(len(unique_values)))}")
        
        return self.data
    
    def prepare_features(self, model_type: str = 'binary') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for model training
        
        Args:
            model_type: 'binary' or 'dual' for different feature sets
            
        Returns:
            Tuple of (normalized_features, actions, feature_names)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Select appropriate features based on model type
        if model_type == 'binary':
            state_features = config.BINARY_STATE_FEATURES
            action_features = [config.BINARY_ACTION]
        elif model_type == 'dual':
            state_features = config.DUAL_STATE_FEATURES
            action_features = config.DUAL_ACTIONS
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'binary' or 'dual'")
        
        print(f"\nPreparing {model_type.upper()} CQL features...")
        print(f"State features ({len(state_features)}): {state_features}")
        print(f"Action features: {action_features}")
        
        # Check for missing data
        all_features = state_features + action_features
        if not self.check_missing_data(all_features):
            raise ValueError("Cannot prepare features due to missing data!")
        
        # Encode categorical features
        print("\nEncoding categorical features...")
        self.encode_categorical_features()
        
        # Extract and normalize state features
        X = self.data[state_features].values
        X_normalized = self.scaler.fit_transform(X)
        
        # Extract actions
        if model_type == 'binary':
            y = self.data[config.BINARY_ACTION].values
        else:
            y = self.data[action_features].values
        
        print(f"\n Features prepared successfully!")
        print(f"  State shape: {X_normalized.shape}")
        print(f"  Action shape: {y.shape}")
        
        return X_normalized, y, state_features
    
    def get_patient_ids(self) -> np.ndarray:
        """Get all unique patient IDs"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data[config.PATIENT_ID_COL].unique()
    
    def get_patient_data(self, patient_id: int) -> pd.DataFrame:
        """Get data for a specific patient"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data[self.data[config.PATIENT_ID_COL] == patient_id].copy()
    
    def get_patient_outcomes(self) -> Dict[int, int]:
        """Get patient outcomes (survived=0, died=1)"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        outcomes = {}
        for patient_id in self.get_patient_ids():
            patient_data = self.get_patient_data(patient_id)
            # Death is recorded in the last timestep
            outcomes[patient_id] = int(patient_data[config.DEATH_COL].iloc[-1])
        
        return outcomes


# ==================== Test Script ====================

if __name__ == "__main__":
    print("="*70)
    print(" DATA LOADER TEST - CHECKING FOR MISSING DATA")
    print("="*70)
    
    # Initialize loader
    loader = DataLoader(verbose=True)
    
    # Load data
    data = loader.load_data()
    
    # Show first few columns to verify data loaded
    print(f"\nFirst 10 columns in data: {list(data.columns[:10])}")
    print(f"Data shape: {data.shape}")
    
    # Test Binary CQL features
    print("\n" + "="*70)
    print(" TESTING BINARY CQL FEATURES")
    print("="*70)
    
    try:
        X_binary, y_binary, features_binary = loader.prepare_features(model_type='binary')
        print(f"\nSample of normalized features (first row):")
        print(X_binary[0][:5], "...")  # Show first 5 values
        print(f"Sample action: {y_binary[0]}")
    except ValueError as e:
        print(f"\n Error: {e}")
    
    # Test Dual Continuous CQL features
    print("\n" + "="*70)
    print(" TESTING DUAL CONTINUOUS CQL FEATURES")
    print("="*70)
    
    # Need fresh scaler for different feature set
    loader.scaler = StandardScaler()
    
    try:
        X_dual, y_dual, features_dual = loader.prepare_features(model_type='dual')
        print(f"\nSample of normalized features (first row):")
        print(X_dual[0][:5], "...")  # Show first 5 values
        print(f"Sample actions: {y_dual[0]}")
    except ValueError as e:
        print(f"\n Error: {e}")
    
    # Show patient outcomes
    print("\n" + "="*70)
    print(" PATIENT OUTCOMES SUMMARY")
    print("="*70)
    
    outcomes = loader.get_patient_outcomes()
    n_survived = sum(1 for o in outcomes.values() if o == 0)
    n_died = sum(1 for o in outcomes.values() if o == 1)
    
    print(f"Total patients: {len(outcomes)}")
    print(f"Survived: {n_survived} ({n_survived/len(outcomes)*100:.1f}%)")
    print(f"Died: {n_died} ({n_died/len(outcomes)*100:.1f}%)")
    
    # Show a sample patient trajectory
    print("\n" + "="*70)
    print(" SAMPLE PATIENT TRAJECTORY")
    print("="*70)
    
    sample_patient = loader.get_patient_ids()[0]
    sample_data = loader.get_patient_data(sample_patient)
    print(f"Patient {sample_patient}:")
    print(f"  - Trajectory length: {len(sample_data)} timesteps")
    print(f"  - Outcome: {'Died' if sample_data[config.DEATH_COL].iloc[-1] == 1 else 'Survived'}")
    print(f"  - Mean SOFA: {sample_data['sofa'].mean():.2f}")
    print(f"  - Mean lactate: {sample_data['lactate'].mean():.2f}")

"""
Train/Validation/Test split utility for patient-level splitting
"""

class DataSplitter:
    """
    Handles train/validation/test splitting at the patient level
    IMPORTANT: We split by patient, not by timestep, to avoid data leakage
    """
    
    def __init__(self, random_seed: int = config.RANDOM_SEED):
        """
        Initialize data splitter
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.train_patients = None
        self.val_patients = None
        self.test_patients = None
        
        # Set seeds for this instance
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
    def split_patients(self, patient_ids: np.ndarray, 
                      train_ratio: float = config.TRAIN_RATIO,
                      val_ratio: float = config.VAL_RATIO,
                      test_ratio: float = config.TEST_RATIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split patient IDs into train/val/test sets
        
        Args:
            patient_ids: Array of all patient IDs
            train_ratio: Proportion for training (default 0.70)
            val_ratio: Proportion for validation (default 0.15)
            test_ratio: Proportion for testing (default 0.15)
            
        Returns:
            Tuple of (train_patients, val_patients, test_patients)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # First split: train+val vs test
        train_val_patients, test_patients = train_test_split(
            patient_ids,
            test_size=test_ratio,
            random_state=self.random_seed
        )
        
        # Second split: train vs val
        # Calculate validation size relative to train+val
        val_size_relative = val_ratio / (train_ratio + val_ratio)
        
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=val_size_relative,
            random_state=self.random_seed
        )
        
        self.train_patients = train_patients
        self.val_patients = val_patients
        self.test_patients = test_patients
        
        return train_patients, val_patients, test_patients
    
    def get_split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the full dataset based on patient splits
        
        Args:
            data: Full DataFrame with all patients
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.train_patients is None:
            raise ValueError("Must call split_patients() first")
        
        train_data = data[data[config.PATIENT_ID_COL].isin(self.train_patients)]
        val_data = data[data[config.PATIENT_ID_COL].isin(self.val_patients)]
        test_data = data[data[config.PATIENT_ID_COL].isin(self.test_patients)]
        
        return train_data, val_data, test_data
    
    def get_split_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Get statistics about the splits
        
        Args:
            data: Full DataFrame
            
        Returns:
            Dictionary with split statistics
        """
        if self.train_patients is None:
            raise ValueError("Must call split_patients() first")
        
        train_data, val_data, test_data = self.get_split_data(data)
        
        stats = {
            'train': {
                'n_patients': len(self.train_patients),
                'n_timesteps': len(train_data),
                'n_died': sum(train_data.groupby(config.PATIENT_ID_COL)[config.DEATH_COL].last() == 1),
                'mortality_rate': 0
            },
            'val': {
                'n_patients': len(self.val_patients),
                'n_timesteps': len(val_data),
                'n_died': sum(val_data.groupby(config.PATIENT_ID_COL)[config.DEATH_COL].last() == 1),
                'mortality_rate': 0
            },
            'test': {
                'n_patients': len(self.test_patients),
                'n_timesteps': len(test_data),
                'n_died': sum(test_data.groupby(config.PATIENT_ID_COL)[config.DEATH_COL].last() == 1),
                'mortality_rate': 0
            }
        }
        
        # Calculate mortality rates
        for split in ['train', 'val', 'test']:
            stats[split]['mortality_rate'] = stats[split]['n_died'] / stats[split]['n_patients']
        
        return stats
