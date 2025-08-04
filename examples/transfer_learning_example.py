#!/usr/bin/env python3
"""
Simple example demonstrating transfer learning with TimesFM-1.0-200m checkpoint.

This script shows how to:
1. Load the pre-trained timesfm-1.0-200m model
2. Set up transfer learning for your data
3. Evaluate zero-shot performance
4. Configure fine-tuning (requires finetuning framework)

Usage:
    python transfer_learning_example.py [--use-pytorch] [--context-len 256] [--horizon-len 96]
"""

import argparse
import logging
import sys
from typing import List

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required but not installed.")
    print("Please install with: pip install numpy")
    sys.exit(1)

try:
    import timesfm
    from timesfm.transfer_learning import (
        TimesFM200mTransferLearner,
        create_transfer_learning_config,
        setup_timesfm_1_0_200m_transfer_learning
    )
    TIMESFM_AVAILABLE = True
except ImportError as e:
    print(f"Note: Full TimesFM functionality not available: {e}")
    print("Running in demonstration mode (configuration only).")
    TIMESFM_AVAILABLE = False


def generate_synthetic_data(num_series: int = 50, series_length: int = 500) -> List[np.ndarray]:
    """Generate synthetic time series data for demonstration."""
    print(f"Generating {num_series} synthetic time series...")
    
    np.random.seed(42)
    time_series_data = []
    
    for i in range(num_series):
        # Create diverse synthetic patterns
        t = np.linspace(0, 4*np.pi, series_length)
        
        # Mix of trends, seasonality, and noise
        trend = i * 0.01 * t
        seasonal = np.sin(t + i*0.5) + 0.3*np.sin(3*t + i*0.2)
        noise = 0.1 * np.random.normal(0, 1, len(t))
        
        # Add some domain-specific patterns
        if i % 3 == 0:
            # Weekly pattern
            weekly = 0.2 * np.sin(2*np.pi*t/7)
            ts = trend + seasonal + weekly + noise
        elif i % 3 == 1:
            # Growth pattern
            growth = 0.1 * np.exp(0.01*t)
            ts = growth + seasonal + noise
        else:
            # Cyclical pattern
            cycle = 0.5 * np.sin(2*np.pi*t/30)
            ts = trend + seasonal + cycle + noise
        
        time_series_data.append(ts)
    
    print(f"Generated {len(time_series_data)} time series with length {series_length}")
    return time_series_data


def demonstrate_transfer_learning_setup(use_pytorch: bool = True,
                                      context_len: int = 256,
                                      horizon_len: int = 96):
    """Demonstrate transfer learning setup with timesfm-1.0-200m."""
    
    print("=" * 60)
    print("TimesFM-1.0-200m Transfer Learning Example")
    print("=" * 60)
    
    # Generate sample data
    sample_data = generate_synthetic_data(num_series=50, series_length=400)
    
    if not TIMESFM_AVAILABLE:
        return demonstrate_configuration_only(sample_data, context_len, horizon_len)
    
    # Quick setup using convenience function
    print("\n1. Setting up transfer learning...")
    try:
        transfer_learner, model = setup_timesfm_1_0_200m_transfer_learning(
            dataset_size=len(sample_data),
            domain_similarity="similar",
            context_len=context_len,
            horizon_len=horizon_len,
            use_pytorch=use_pytorch
        )
        print("✓ Transfer learning setup complete!")
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        print("Note: This requires proper TimesFM installation and model downloads.")
        return demonstrate_configuration_only(sample_data, context_len, horizon_len)
    
    # Prepare data
    print("\n2. Preparing data...")
    train_data, val_data = transfer_learner.prepare_data(sample_data)
    
    # Evaluate zero-shot performance
    print("\n3. Evaluating zero-shot performance...")
    try:
        metrics = transfer_learner.evaluate_zero_shot(val_data[:5])  # Use subset for demo
        print("Zero-shot metrics:")
        for metric, value in metrics.items():
            if metric != "num_predictions":
                print(f"  {metric.upper()}: {value:.4f}")
    except Exception as e:
        print(f"Zero-shot evaluation failed: {e}")
    
    # Show next steps
    print("\n4. Next steps for fine-tuning:")
    print("   - Use TimesFMFinetuner from finetuning.finetuning_torch")
    print("   - Apply the recommended configuration")
    print("   - Monitor training with validation metrics")
    print("   - Save and deploy fine-tuned model")
    
    return True


def demonstrate_configuration_only(sample_data: List[np.ndarray],
                                 context_len: int = 256,
                                 horizon_len: int = 96):
    """Demonstrate configuration without loading the actual model."""
    
    print("\n1. Creating transfer learning configuration...")
    
    # Create configuration for different scenarios
    scenarios = [
        (20, "very_similar"),
        (100, "similar"),
        (500, "different"),
        (2000, "similar")
    ]
    
    # Simple recommendation logic (without full TimesFM)
    def simple_recommend_strategy(dataset_size: int, domain_similarity: str) -> dict:
        strategies = {
            "very_small": {
                "very_similar": {"approach": "feature_extraction", "learning_rate": 0},
                "similar": {"approach": "fine_tuning", "learning_rate": 1e-5},
                "different": {"approach": "fine_tuning", "learning_rate": 1e-4}
            },
            "small": {
                "very_similar": {"approach": "fine_tuning", "learning_rate": 1e-4},
                "similar": {"approach": "gradual_unfreezing", "learning_rate": 1e-4},
                "different": {"approach": "fine_tuning", "learning_rate": 5e-4}
            },
            "large": {
                "very_similar": {"approach": "gradual_unfreezing", "learning_rate": 1e-4},
                "similar": {"approach": "fine_tuning", "learning_rate": 1e-3},
                "different": {"approach": "fine_tuning", "learning_rate": 1e-3}
            }
        }
        
        size_category = "very_small" if dataset_size < 100 else "small" if dataset_size < 1000 else "large"
        return strategies[size_category][domain_similarity]
    
    for dataset_size, domain_similarity in scenarios:
        print(f"\nScenario: {dataset_size} series, {domain_similarity} domain")
        
        strategy = simple_recommend_strategy(dataset_size, domain_similarity)
        
        print(f"  Recommended approach: {strategy['approach']}")
        print(f"  Learning rate: {strategy['learning_rate']}")
    
    # Show data preparation concept
    print(f"\n2. Data preparation for {len(sample_data)} series...")
    min_length = context_len + horizon_len
    valid_series = [ts for ts in sample_data if len(ts) >= min_length]
    split_idx = int(0.8 * len(valid_series))
    
    print(f"  Training samples: {split_idx}")
    print(f"  Validation samples: {len(valid_series) - split_idx}")
    print(f"  Context length: {context_len}")
    print(f"  Horizon length: {horizon_len}")
    
    print("\n3. Configuration complete!")
    print("   Ready for model loading and fine-tuning")
    print("   Install TimesFM dependencies for full functionality")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="TimesFM-1.0-200m Transfer Learning Example")
    parser.add_argument("--use-pytorch", action="store_true", default=True,
                       help="Use PyTorch version (default: True)")
    parser.add_argument("--context-len", type=int, default=256,
                       help="Context length (must be multiple of 32, max 512)")
    parser.add_argument("--horizon-len", type=int, default=96,
                       help="Forecast horizon length")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Validate arguments
    if args.context_len > 512:
        print("Error: timesfm-1.0-200m supports maximum context length of 512")
        return 1
    if args.context_len % 32 != 0:
        print("Error: context_len must be a multiple of 32")
        return 1
    
    # Run demonstration
    try:
        success = demonstrate_transfer_learning_setup(
            use_pytorch=args.use_pytorch,
            context_len=args.context_len,
            horizon_len=args.horizon_len
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())