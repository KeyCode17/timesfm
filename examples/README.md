# TimesFM Examples

This directory contains practical examples for using TimesFM, particularly focused on transfer learning with the timesfm-1.0-200m checkpoint.

## Transfer Learning Example

The `transfer_learning_example.py` script demonstrates how to use the timesfm-1.0-200m checkpoint for transfer learning on your custom time series data.

### Features

- **Automatic Strategy Selection**: Recommends the best transfer learning approach based on your dataset size and domain similarity
- **Data Preparation**: Handles missing values, outlier removal, and normalization
- **Zero-shot Evaluation**: Test the pre-trained model without fine-tuning
- **Configuration Management**: Easy setup for different scenarios

### Usage

```bash
# Basic usage with default settings
python examples/transfer_learning_example.py

# Use PyTorch version with custom context length
python examples/transfer_learning_example.py --use-pytorch --context-len 320 --horizon-len 128

# Enable verbose logging
python examples/transfer_learning_example.py --verbose
```

### Command Line Options

- `--use-pytorch`: Use PyTorch implementation (default: True)
- `--context-len`: Context length, must be multiple of 32, max 512 for 1.0 model (default: 256)
- `--horizon-len`: Forecast horizon length (default: 96)
- `--verbose`: Enable detailed logging

### Example Output

```
TimesFM-1.0-200m Transfer Learning Example
============================================================

Generating 50 synthetic time series...
Generated 50 time series with length 400

1. Setting up transfer learning...
Created configuration for 50 series, similar domain:
  Approach: gradual_unfreezing
  Learning rate: 0.0001
  Epochs: 10
  Batch size: 16
✓ Transfer learning setup complete!

2. Preparing data...
Prepared 40 training series and 10 validation series

3. Evaluating zero-shot performance...
Zero-shot metrics:
  MAE: 0.2345
  MSE: 0.0876
  RMSE: 0.2959
  MAPE: 12.34

4. Next steps for fine-tuning:
   - Use TimesFMFinetuner from finetuning.finetuning_torch
   - Apply the recommended configuration
   - Monitor training with validation metrics
   - Save and deploy fine-tuned model
```

## Requirements

To run the examples, you need:

1. **TimesFM Installation**: Install TimesFM with either PyTorch or JAX dependencies
2. **Model Access**: The script will download the timesfm-1.0-200m checkpoint from Hugging Face
3. **Python Dependencies**: numpy, pandas, and other TimesFM dependencies

## Integration with Existing Notebooks

This example complements the existing TimesFM notebooks:

- **[finetuning_torch.ipynb](../notebooks/finetuning_torch.ipynb)**: Detailed PyTorch fine-tuning tutorial
- **[transfer_learning.ipynb](../notebooks/transfer_learning.ipynb)**: Comprehensive transfer learning guide
- **[finetuning.ipynb](../notebooks/finetuning.ipynb)**: JAX-based fine-tuning examples

## Next Steps

After running the transfer learning example:

1. **Load Your Data**: Replace the synthetic data with your own time series
2. **Fine-tune the Model**: Use the generated configuration with TimesFMFinetuner
3. **Evaluate Performance**: Compare zero-shot vs fine-tuned performance
4. **Deploy**: Save and deploy your optimized model

For more detailed guidance, see the [Transfer Learning Notebook](../notebooks/transfer_learning.ipynb).