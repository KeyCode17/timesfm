# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transfer learning utilities for TimesFM-1.0-200m checkpoint."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from dataclasses import dataclass

import timesfm
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams


@dataclass
class TransferLearningConfig:
  """Configuration for transfer learning with TimesFM-1.0-200m."""
  
  # Model configuration
  context_len: int = 256  # Must be multiple of 32, max 512 for 1.0 model
  horizon_len: int = 96
  backend: str = "auto"  # "auto", "gpu", or "cpu"
  
  # Transfer learning strategy
  approach: str = "fine_tuning"  # "feature_extraction", "fine_tuning", "gradual_unfreezing"
  learning_rate: float = 1e-4
  num_epochs: int = 10
  batch_size: int = 16
  
  # Data preprocessing
  normalize_data: bool = True
  remove_outliers: bool = True
  train_val_split: float = 0.8
  
  # Regularization for transfer learning
  weight_decay: float = 1e-4
  dropout_rate: float = 0.1
  early_stopping_patience: int = 5


class TimesFM200mTransferLearner:
  """Transfer learning utility for TimesFM-1.0-200m checkpoint."""
  
  def __init__(self, config: TransferLearningConfig):
    """Initialize the transfer learning utility.
    
    Args:
      config: Transfer learning configuration
    """
    self.config = config
    self.model = None
    self._validate_config()
  
  def _validate_config(self):
    """Validate the transfer learning configuration."""
    if self.config.context_len > 512:
      raise ValueError("timesfm-1.0-200m supports maximum context length of 512")
    if self.config.context_len % 32 != 0:
      raise ValueError("context_len must be a multiple of 32")
    if self.config.approach not in ["feature_extraction", "fine_tuning", "gradual_unfreezing"]:
      raise ValueError(f"Unknown approach: {self.config.approach}")
  
  def load_pretrained_model(self, use_pytorch: bool = True) -> TimesFm:
    """Load the pre-trained timesfm-1.0-200m model.
    
    Args:
      use_pytorch: Whether to use PyTorch version (True) or JAX version (False)
    
    Returns:
      Loaded TimesFM model
    """
    # Determine backend
    if self.config.backend == "auto":
      backend = "gpu" if torch.cuda.is_available() else "cpu"
    else:
      backend = self.config.backend
    
    # Choose checkpoint based on implementation
    if use_pytorch:
      repo_id = "google/timesfm-1.0-200m-pytorch"
    else:
      repo_id = "google/timesfm-1.0-200m"
    
    # Initialize model
    self.model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=self.config.batch_size,
            horizon_len=self.config.horizon_len,
            context_len=self.config.context_len,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id=repo_id
        ),
    )
    
    logging.info(f"Loaded {repo_id} with:")
    logging.info(f"  - Context length: {self.config.context_len}")
    logging.info(f"  - Horizon length: {self.config.horizon_len}")
    logging.info(f"  - Backend: {backend}")
    
    return self.model
  
  def recommend_strategy(self, 
                        dataset_size: int, 
                        domain_similarity: str = "similar") -> Dict[str, Union[str, float]]:
    """Recommend transfer learning strategy based on dataset characteristics.
    
    Args:
      dataset_size: Number of time series in your dataset
      domain_similarity: How similar your domain is to pre-training data
                        ("very_similar", "similar", "different", "very_different")
    
    Returns:
      Dictionary with recommended strategy and parameters
    """
    strategies = {
        "very_small": {  # < 100 series
            "very_similar": {
                "approach": "feature_extraction",
                "learning_rate": 0,
                "freeze_backbone": True,
                "description": "Use pre-trained model as feature extractor only"
            },
            "similar": {
                "approach": "fine_tuning",
                "learning_rate": 1e-5,
                "freeze_backbone": False,
                "description": "Light fine-tuning with very low learning rate"
            },
            "different": {
                "approach": "fine_tuning",
                "learning_rate": 1e-4,
                "freeze_backbone": False,
                "description": "Careful fine-tuning with regularization"
            }
        },
        "small": {  # 100-1000 series
            "very_similar": {
                "approach": "fine_tuning",
                "learning_rate": 1e-4,
                "freeze_backbone": False,
                "description": "Standard fine-tuning approach"
            },
            "similar": {
                "approach": "gradual_unfreezing",
                "learning_rate": 1e-4,
                "freeze_backbone": False,
                "description": "Gradually unfreeze layers during training"
            },
            "different": {
                "approach": "fine_tuning",
                "learning_rate": 5e-4,
                "freeze_backbone": False,
                "description": "More aggressive fine-tuning"
            }
        },
        "large": {  # > 1000 series
            "very_similar": {
                "approach": "gradual_unfreezing",
                "learning_rate": 1e-4,
                "freeze_backbone": False,
                "description": "Gradual unfreezing with standard parameters"
            },
            "similar": {
                "approach": "fine_tuning",
                "learning_rate": 1e-3,
                "freeze_backbone": False,
                "description": "Full fine-tuning with higher learning rate"
            },
            "different": {
                "approach": "fine_tuning",
                "learning_rate": 1e-3,
                "freeze_backbone": False,
                "description": "Aggressive fine-tuning for domain adaptation"
            }
        }
    }
    
    # Determine dataset size category
    if dataset_size < 100:
      size_category = "very_small"
    elif dataset_size < 1000:
      size_category = "small"
    else:
      size_category = "large"
    
    strategy = strategies[size_category][domain_similarity].copy()
    strategy.update({
        "dataset_size": dataset_size,
        "size_category": size_category,
        "domain_similarity": domain_similarity
    })
    
    return strategy
  
  def prepare_data(self, 
                   time_series_data: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Prepare time series data for transfer learning.
    
    Args:
      time_series_data: List of time series arrays
    
    Returns:
      Tuple of (training_data, validation_data)
    """
    processed_data = []
    min_length = self.config.context_len + self.config.horizon_len
    
    for ts in time_series_data:
      # Skip series that are too short
      if len(ts) < min_length:
        continue
      
      # Handle missing values (simple forward fill)
      if np.any(np.isnan(ts)):
        import pandas as pd
        ts = pd.Series(ts).fillna(method='ffill').fillna(method='bfill').values
      
      # Optional: Remove outliers
      if self.config.remove_outliers:
        q1, q3 = np.percentile(ts, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        ts = np.clip(ts, lower_bound, upper_bound)
      
      # Optional: Normalize data
      if self.config.normalize_data:
        ts_mean = np.mean(ts)
        ts_std = np.std(ts) + 1e-8
        ts = (ts - ts_mean) / ts_std
      
      processed_data.append(ts)
    
    # Split into train/validation
    split_idx = int(self.config.train_val_split * len(processed_data))
    train_data = processed_data[:split_idx]
    val_data = processed_data[split_idx:]
    
    logging.info(f"Prepared {len(train_data)} training series and {len(val_data)} validation series")
    
    return train_data, val_data
  
  def evaluate_zero_shot(self, 
                        test_data: List[np.ndarray],
                        freq_type: int = 0) -> Dict[str, float]:
    """Evaluate zero-shot performance (no fine-tuning).
    
    Args:
      test_data: Test time series data
      freq_type: Frequency type (0=high, 1=medium, 2=low)
    
    Returns:
      Dictionary with evaluation metrics
    """
    if self.model is None:
      raise ValueError("Model not loaded. Call load_pretrained_model() first.")
    
    predictions = []
    ground_truth = []
    
    for ts in test_data:
      if len(ts) < self.config.context_len + self.config.horizon_len:
        continue
      
      # Split into context and target
      context = ts[:self.config.context_len]
      target = ts[self.config.context_len:self.config.context_len + self.config.horizon_len]
      
      try:
        # Generate forecast
        forecast, _ = self.model.forecast([context], freq=[freq_type])
        predictions.append(forecast[0][:self.config.horizon_len])
        ground_truth.append(target)
      except Exception as e:
        logging.warning(f"Error in forecasting: {e}")
        continue
    
    if not predictions:
      return {"error": "No successful predictions"}
    
    # Calculate metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    mae = np.mean(np.abs(predictions - ground_truth))
    mse = np.mean((predictions - ground_truth)**2)
    rmse = np.sqrt(mse)
    
    # MAPE (handling zero values)
    non_zero_mask = ground_truth != 0
    if np.any(non_zero_mask):
      mape = np.mean(np.abs((ground_truth[non_zero_mask] - predictions[non_zero_mask]) / 
                           ground_truth[non_zero_mask])) * 100
    else:
      mape = float('inf')
    
    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "num_predictions": len(predictions)
    }
    
    logging.info("Zero-shot evaluation metrics:")
    for metric, value in metrics.items():
      if metric != "num_predictions":
        logging.info(f"  {metric.upper()}: {value:.4f}")
    
    return metrics


def create_transfer_learning_config(dataset_size: int,
                                   domain_similarity: str = "similar",
                                   context_len: int = 256,
                                   horizon_len: int = 96) -> TransferLearningConfig:
  """Create a transfer learning configuration with recommended settings.
  
  Args:
    dataset_size: Number of time series in your dataset
    domain_similarity: How similar your domain is to pre-training data
    context_len: Context length (must be multiple of 32, max 512)
    horizon_len: Horizon length
  
  Returns:
    Configured TransferLearningConfig
  """
  # Get strategy recommendation
  learner = TimesFM200mTransferLearner(
      TransferLearningConfig(context_len=context_len, horizon_len=horizon_len)
  )
  strategy = learner.recommend_strategy(dataset_size, domain_similarity)
  
  # Create configuration
  config = TransferLearningConfig(
      context_len=context_len,
      horizon_len=horizon_len,
      approach=strategy["approach"],
      learning_rate=strategy["learning_rate"],
      num_epochs=10 if dataset_size < 100 else 15 if dataset_size < 1000 else 20,
      batch_size=8 if dataset_size < 100 else 16 if dataset_size < 1000 else 32,
  )
  
  logging.info(f"Created configuration for {dataset_size} series, {domain_similarity} domain:")
  logging.info(f"  Approach: {config.approach}")
  logging.info(f"  Learning rate: {config.learning_rate}")
  logging.info(f"  Epochs: {config.num_epochs}")
  logging.info(f"  Batch size: {config.batch_size}")
  
  return config


# Convenience function for quick setup
def setup_timesfm_1_0_200m_transfer_learning(dataset_size: int,
                                             domain_similarity: str = "similar",
                                             context_len: int = 256,
                                             horizon_len: int = 96,
                                             use_pytorch: bool = True) -> Tuple[TimesFM200mTransferLearner, TimesFm]:
  """Quick setup for transfer learning with timesfm-1.0-200m.
  
  Args:
    dataset_size: Number of time series in your dataset
    domain_similarity: How similar your domain is to pre-training data
    context_len: Context length (must be multiple of 32, max 512)
    horizon_len: Horizon length
    use_pytorch: Whether to use PyTorch version
  
  Returns:
    Tuple of (transfer_learner, loaded_model)
  """
  # Create configuration
  config = create_transfer_learning_config(
      dataset_size=dataset_size,
      domain_similarity=domain_similarity,
      context_len=context_len,
      horizon_len=horizon_len
  )
  
  # Initialize transfer learner
  transfer_learner = TimesFM200mTransferLearner(config)
  
  # Load model
  model = transfer_learner.load_pretrained_model(use_pytorch=use_pytorch)
  
  return transfer_learner, model