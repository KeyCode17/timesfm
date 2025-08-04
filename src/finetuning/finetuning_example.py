"""
Example usage of the TimesFM Finetuning Framework with Transfer Learning Support.

This example demonstrates how to use TimesFM's finetuning capabilities for 
transfer learning, particularly with the timesfm-1.0-200m checkpoint.

Transfer Learning Modes:
- Feature extraction: Use pre-trained model as fixed feature extractor
- Fine-tuning: Adapt all or some layers to your domain
- Gradual unfreezing: Start frozen and gradually unfreeze layers

For single GPU:
python script.py --training_mode=single --transfer_learning_mode=fine_tuning

For multiple GPUs with transfer learning:
python script.py --training_mode=multi --gpu_ids=0,1,2 --transfer_learning_mode=gradual_unfreezing

For transfer learning with timesfm-1.0-200m:
python script.py --checkpoint_path=google/timesfm-1.0-200m-pytorch --transfer_learning_mode=fine_tuning
"""

import os
from dataclasses import asdict
from os import path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yfinance as yf
from absl import app, flags
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch.utils.data import Dataset

from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import (PatchedTimeSeriesDecoder,
                                             TimesFMConfig)

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "training_mode",
    "single",
    ["single", "multi"],
    'Training mode: "single" for single-GPU or "multi" for multi-GPU training.',
)

flags.DEFINE_list(
    "gpu_ids", ["0"],
    "Comma-separated list of GPU IDs to use for multi-GPU training. Example: 0,1,2"
)

flags.DEFINE_string(
    "local_model_path",
    None,
    "Path to a local .safetensors model file. If provided, overrides Hugging Face download."
)

flags.DEFINE_enum(
    "transfer_learning_mode",
    "fine_tuning",
    ["feature_extraction", "fine_tuning", "gradual_unfreezing"],
    "Transfer learning approach: feature_extraction (frozen backbone), "
    "fine_tuning (adapt all layers), or gradual_unfreezing (progressive training)."
)

flags.DEFINE_string(
    "checkpoint_path",
    "google/timesfm-1.0-200m-pytorch",
    "Hugging Face checkpoint path. Use 'google/timesfm-1.0-200m-pytorch' or "
    "'google/timesfm-1.0-200m' for the 200M parameter model."
)

flags.DEFINE_float(
    "transfer_learning_lr",
    1e-4,
    "Learning rate for transfer learning. Typically lower than training from scratch."
)

class TimeSeriesDataset(Dataset):
  """Dataset for time series data compatible with TimesFM."""

  def __init__(self,
               series: np.ndarray,
               context_length: int,
               horizon_length: int,
               freq_type: int = 0):
    """
        Initialize dataset.

        Args:
            series: Time series data
            context_length: Number of past timesteps to use as input
            horizon_length: Number of future timesteps to predict
            freq_type: Frequency type (0, 1, or 2)
        """
    if freq_type not in [0, 1, 2]:
      raise ValueError("freq_type must be 0, 1, or 2")

    self.series = series
    self.context_length = context_length
    self.horizon_length = horizon_length
    self.freq_type = freq_type
    self._prepare_samples()

  def _prepare_samples(self) -> None:
    """Prepare sliding window samples from the time series."""
    self.samples = []
    total_length = self.context_length + self.horizon_length

    for start_idx in range(0, len(self.series) - total_length + 1):
      end_idx = start_idx + self.context_length
      x_context = self.series[start_idx:end_idx]
      x_future = self.series[end_idx:end_idx + self.horizon_length]
      self.samples.append((x_context, x_future))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(
      self, index: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_context, x_future = self.samples[index]

    x_context = torch.tensor(x_context, dtype=torch.float32)
    x_future = torch.tensor(x_future, dtype=torch.float32)

    input_padding = torch.zeros_like(x_context)
    freq = torch.tensor([self.freq_type], dtype=torch.long)

    return x_context, input_padding, freq, x_future


def prepare_datasets(series: np.ndarray,
                     context_length: int,
                     horizon_length: int,
                     freq_type: int = 0,
                     train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
  """
    Prepare training and validation datasets from time series data.

    Args:
        series: Input time series data
        context_length: Number of past timesteps to use
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
  train_size = int(len(series) * train_split)
  train_data = series[:train_size]
  val_data = series[train_size:]

  # Create datasets with specified frequency type
  train_dataset = TimeSeriesDataset(train_data,
                                    context_length=context_length,
                                    horizon_length=horizon_length,
                                    freq_type=freq_type)

  val_dataset = TimeSeriesDataset(val_data,
                                  context_length=context_length,
                                  horizon_length=horizon_length,
                                  freq_type=freq_type)

  return train_dataset, val_dataset


def get_model(load_weights: bool = False):
  """
  Initialize TimesFM model with transfer learning support.
  
  This function supports loading various TimesFM checkpoints for transfer learning,
  including the timesfm-1.0-200m model which is ideal for transfer learning scenarios.
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  # Adjust hyperparameters based on checkpoint
  if "1.0-200m" in FLAGS.checkpoint_path:
    # TimesFM 1.0 model configuration
    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=32,
        horizon_len=128,
        context_len=192,  # Max 512 for 1.0 model
    )
  else:
    # TimesFM 2.0 model configuration (default)
    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,
        use_positional_embedding=False,
        context_len=192,
    )
  
  print(f"Transfer Learning Configuration:")
  print(f"  Checkpoint: {FLAGS.checkpoint_path}")
  print(f"  Mode: {FLAGS.transfer_learning_mode}")
  print(f"  Learning Rate: {FLAGS.transfer_learning_lr}")
  print(f"  Context Length: {hparams.context_len}")
  
  if load_weights:
    if FLAGS.local_model_path:
      # Load from local file
      tfm_config = TimesFMConfig()
      model = PatchedTimeSeriesDecoder(tfm_config)
      loaded_checkpoint = load_file(FLAGS.local_model_path)
      print(f"Loading from local path: {FLAGS.local_model_path}")
    else:
      # Load from Hugging Face
      repo_id = FLAGS.checkpoint_path
      tfm = TimesFm(hparams=hparams,
              checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

      tfm_config = tfm._model_config
      model = PatchedTimeSeriesDecoder(tfm_config)
      checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
      loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
      print(f"Loading from Hugging Face: {repo_id}")

    model.load_state_dict(loaded_checkpoint)
    
    # Apply transfer learning configuration
    if FLAGS.transfer_learning_mode == "feature_extraction":
      # Freeze backbone for feature extraction
      for name, param in model.named_parameters():
        if "head" not in name.lower():  # Keep head trainable
          param.requires_grad = False
      print("Applied feature extraction mode: backbone frozen, head trainable")
      
    elif FLAGS.transfer_learning_mode == "gradual_unfreezing":
      # Start with everything frozen except head
      for name, param in model.named_parameters():
        if "head" not in name.lower():
          param.requires_grad = False
      print("Applied gradual unfreezing mode: start with backbone frozen")
      
    else:  # fine_tuning
      # All parameters trainable
      for param in model.parameters():
        param.requires_grad = True
      print("Applied fine-tuning mode: all parameters trainable")
  
  return model, hparams, tfm_config


def plot_predictions(
    model: TimesFm,
    val_dataset: Dataset,
    save_path: Optional[str] = "predictions.png",
) -> None:
  """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
      model: Trained TimesFM model
      val_dataset: Validation dataset
      save_path: Path to save the plot
    """
  import matplotlib.pyplot as plt

  model.eval()

  x_context, x_padding, freq, x_future = val_dataset[0]
  x_context = x_context.unsqueeze(0)  # Add batch dimension
  x_padding = x_padding.unsqueeze(0)
  freq = freq.unsqueeze(0)
  x_future = x_future.unsqueeze(0)

  device = next(model.parameters()).device
  x_context = x_context.to(device)
  x_padding = x_padding.to(device)
  freq = freq.to(device)
  x_future = x_future.to(device)

  with torch.no_grad():
    predictions = model(x_context, x_padding.float(), freq)
    predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
    last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

  context_vals = x_context[0].cpu().numpy()
  future_vals = x_future[0].cpu().numpy()
  pred_vals = last_patch_pred[0].cpu().numpy()

  context_len = len(context_vals)
  horizon_len = len(future_vals)

  plt.figure(figsize=(12, 6))

  plt.plot(range(context_len),
           context_vals,
           label="Historical Data",
           color="blue",
           linewidth=2)

  plt.plot(
      range(context_len, context_len + horizon_len),
      future_vals,
      label="Ground Truth",
      color="green",
      linestyle="--",
      linewidth=2,
  )

  plt.plot(range(context_len, context_len + horizon_len),
           pred_vals,
           label="Prediction",
           color="red",
           linewidth=2)

  plt.xlabel("Time Step")
  plt.ylabel("Value")
  plt.title("TimesFM Predictions vs Ground Truth")
  plt.legend()
  plt.grid(True)

  if save_path:
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

  plt.close()


def get_data(context_len: int,
             horizon_len: int,
             freq_type: int = 0) -> Tuple[Dataset, Dataset]:
  df = yf.download("AAPL", start="2010-01-01", end="2019-01-01")
  time_series = df["Close"].values

  train_dataset, val_dataset = prepare_datasets(
      series=time_series,
      context_length=context_len,
      horizon_length=horizon_len,
      freq_type=freq_type,
      train_split=0.8,
  )

  print(f"Created datasets:")
  print(f"- Training samples: {len(train_dataset)}")
  print(f"- Validation samples: {len(val_dataset)}")
  print(f"- Using frequency type: {freq_type}")
  return train_dataset, val_dataset


def single_gpu_example():
  """Transfer learning example using TimesFM with single GPU."""
  model, hparams, tfm_config = get_model(load_weights=True)
  
  # Use transfer learning learning rate from flags
  config = FinetuningConfig(
      batch_size=256,
      num_epochs=5,
      learning_rate=FLAGS.transfer_learning_lr,  # Use transfer learning LR
      use_wandb=True,
      freq_type=1,
      log_every_n_steps=10,
      val_check_interval=0.5,
      use_quantile_loss=True
  )

  print(f"\nTransfer Learning Configuration:")
  print(f"  Mode: {FLAGS.transfer_learning_mode}")
  print(f"  Learning Rate: {config.learning_rate}")
  print(f"  Checkpoint: {FLAGS.checkpoint_path}")

  train_dataset, val_dataset = get_data(128,
                                        tfm_config.horizon_len,
                                        freq_type=config.freq_type)
  finetuner = TimesFMFinetuner(model, config)

  print("\nStarting transfer learning...")
  results = finetuner.finetune(train_dataset=train_dataset,
                               val_dataset=val_dataset)

  print("\nTransfer learning completed!")
  print(f"Training history: {len(results['history']['train_loss'])} epochs")

  plot_predictions(
      model=model,
      val_dataset=val_dataset,
      save_path="timesfm_transfer_learning_predictions.png",
  )


def setup_process(rank, world_size, model, config, train_dataset, val_dataset,
                  return_dict):
  """Setup process function with optimized CUDA handling."""
  try:
    if torch.cuda.is_available():
      torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    if not torch.distributed.is_initialized():
      torch.distributed.init_process_group(backend="nccl",
                                           world_size=world_size,
                                           rank=rank)

    finetuner = TimesFMFinetuner(model, config, rank=rank)

    results = finetuner.finetune(train_dataset=train_dataset,
                                 val_dataset=val_dataset)

    if rank == 0:
      return_dict["results"] = results
      plot_predictions(
          model=model,
          val_dataset=val_dataset,
          save_path="timesfm_predictions.png",
      )

  except Exception as e:
    print(f"Error in process {rank}: {str(e)}")
    raise e
  finally:
    if torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()


def multi_gpu_example():
  """Transfer learning example using TimesFM with multiple GPUs."""
  mp.set_start_method("spawn", force=True)

  gpu_ids = [int(id) for id in FLAGS.gpu_ids]
  world_size = len(gpu_ids)

  model, hparams, tfm_config = get_model(load_weights=True)

  # Create transfer learning config
  config = FinetuningConfig(
      batch_size=256,
      num_epochs=5,
      learning_rate=FLAGS.transfer_learning_lr,  # Use transfer learning LR
      use_wandb=True,
      distributed=True,
      gpu_ids=gpu_ids,
      log_every_n_steps=50,
      val_check_interval=0.5,
  )
  
  print(f"\nMulti-GPU Transfer Learning Configuration:")
  print(f"  Mode: {FLAGS.transfer_learning_mode}")
  print(f"  Learning Rate: {config.learning_rate}")
  print(f"  Checkpoint: {FLAGS.checkpoint_path}")
  print(f"  GPUs: {gpu_ids}")
  
  train_dataset, val_dataset = get_data(128, tfm_config.horizon_len)
  manager = mp.Manager()
  return_dict = manager.dict()

  # Launch processes
  mp.spawn(
      setup_process,
      args=(world_size, model, config, train_dataset, val_dataset, return_dict),
      nprocs=world_size,
      join=True,
  )

  results = return_dict.get("results", None)
  print("\nMulti-GPU transfer learning completed!")
  return results


def main(argv):
  """Main function that selects and runs the appropriate transfer learning mode."""
  
  print("TimesFM Transfer Learning Framework")
  print("=" * 50)
  print(f"Checkpoint: {FLAGS.checkpoint_path}")
  print(f"Transfer Learning Mode: {FLAGS.transfer_learning_mode}")
  print(f"Learning Rate: {FLAGS.transfer_learning_lr}")
  print(f"Training Mode: {FLAGS.training_mode}")
  print("=" * 50)

  try:
    if FLAGS.training_mode == "single":
      print("\nStarting single-GPU transfer learning...")
      single_gpu_example()
    else:
      gpu_ids = [int(id) for id in FLAGS.gpu_ids]
      print(f"\nStarting multi-GPU transfer learning using GPUs: {gpu_ids}...")
      multi_gpu_example()

  except Exception as e:
    print(f"Transfer learning failed: {str(e)}")
    print("\nCommon issues:")
    print("- Ensure TimesFM dependencies are installed")
    print("- Check that the specified checkpoint is available")
    print("- Verify GPU availability if using GPU backend")
    print("- For timesfm-1.0-200m, ensure context_len <= 512")
  finally:
    if torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()


if __name__ == "__main__":
  app.run(main)
