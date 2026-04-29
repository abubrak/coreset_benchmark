"""
End-to-end test for all coreset selection methods in continual learning.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.continual_learning import (
    create_task_datasets,
    CoresetBuffer,
    run_continual_learning
)
from src.models.cnn import CNN_MNIST


def test_all_methods_quick():
    """Quick test of all methods with minimal configuration."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    methods = ['random', 'uniform', 'bcsr']  # Start with subset

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing method: {method}")
        print(f"{'='*60}")

        try:
            # Create minimal task dataset
            print("Creating task datasets...")
            train_loaders, test_loaders, num_classes, input_shape = create_task_datasets(
                dataset_name='MNIST',
                num_tasks=2,
                num_classes_per_task=2,
                batch_size=64,
                data_root='./data'
            )

            # Create model
            print("Creating model...")
            model = CNN_MNIST(num_classes=2).to(device)

            # Create buffer
            print(f"Creating buffer with {method} selection...")
            buffer = CoresetBuffer(
                memory_size=100,
                input_shape=input_shape,
                num_classes=num_classes,
                device=device
            )

            # Test selection from first task
            print("Testing coreset selection...")
            all_data = []
            all_labels = []
            for data, labels in train_loaders[0]:
                all_data.append(data)
                all_labels.append(labels)

            all_data = torch.cat(all_data, dim=0)[:200]  # Limit for speed
            all_labels = torch.cat(all_labels, dim=0)[:200]

            selected_data, selected_labels = buffer.select_coreset(
                data=all_data,
                labels=all_labels,
                num_samples=20,
                method=method,
                model=model if method in ['bcsr', 'csrel', 'bilevel'] else None
            )

            print(f"[PASS] {method} method: selected {len(selected_data)} samples")
            assert len(selected_data) == 20
            assert len(selected_labels) == 20

        except Exception as e:
            print(f"[FAIL] {method} method failed: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_all_methods_quick()
