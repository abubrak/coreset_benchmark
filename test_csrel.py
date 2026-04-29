"""
Simple test script for CSReL implementation.
"""

import torch
import numpy as np
from src.coreset import CSReLCoreset, select_by_loss_diff
from src.configs import CSReLConfig


def test_selection_functions():
    """Test selection functions."""
    print("=" * 50)
    print("Testing selection functions...")
    print("=" * 50)

    # Create dummy data
    n_samples = 100
    num_classes = 10

    losses = torch.randn(n_samples).abs()
    reference_losses = torch.randn(n_samples).abs() * 0.5
    labels = torch.randint(0, num_classes, (n_samples,))

    # Test select_by_loss_diff
    print("\n1. Testing select_by_loss_diff (without class balance)")
    selected = select_by_loss_diff(
        losses=losses,
        reference_losses=reference_losses,
        num_samples=20,
        class_balance=False,
        labels=labels,
        num_classes=num_classes
    )
    print(f"   Selected {len(selected)} samples")
    print(f"   Sample indices: {selected[:10].tolist()}...")

    # Test with class balance
    print("\n2. Testing select_by_loss_diff (with class balance)")
    selected_balanced = select_by_loss_diff(
        losses=losses,
        reference_losses=reference_losses,
        num_samples=20,
        class_balance=True,
        labels=labels,
        num_classes=num_classes
    )
    print(f"   Selected {len(selected_balanced)} samples")

    # Check class distribution
    unique, counts = torch.unique(labels[selected_balanced], return_counts=True)
    print(f"   Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    print("\n[PASS] Selection functions test passed!")


def test_csrel_coreset():
    """Test CSReL coreset selection."""
    print("\n" + "=" * 50)
    print("Testing CSReLCoreset...")
    print("=" * 50)

    # Create configuration
    config = CSReLConfig(
        dataset="MNIST",
        num_classes=10,
        num_epochs=5,  # Use small number for testing
        batch_size=32,
        learning_rate=0.001,
        selection_ratio=0.1,
        device="cpu"  # Use CPU for testing
    )

    # Create dummy data (simulating MNIST)
    n_samples = 100
    train_data = torch.randn(n_samples, 1, 28, 28)
    train_labels = torch.randint(0, 10, (n_samples,))

    print(f"\nDataset shape: {train_data.shape}")
    print(f"Labels shape: {train_labels.shape}")

    # Initialize CSReL selector
    selector = CSReLCoreset(config=config)
    print("\n[PASS] CSReL selector initialized")

    # Train reference model
    print("\nTraining reference model...")
    ref_model, ref_losses = selector.train_reference_model(
        train_data=train_data,
        train_labels=train_labels,
        verbose=True
    )
    print(f"Reference losses shape: {ref_losses.shape}")
    print(f"Reference loss range: [{ref_losses.min():.4f}, {ref_losses.max():.4f}]")

    # Select coreset
    print("\nSelecting coreset...")
    from src.models import get_model
    current_model = get_model("MNIST", num_classes=10)
    selected_indices = selector.select(
        train_data=train_data,
        train_labels=train_labels,
        model=current_model,  # 添加 model 参数
        verbose=True
    )

    print(f"\nSelected {len(selected_indices)} samples")
    print(f"Expected: {int(n_samples * config.selection_ratio)} samples")

    # Get selection statistics
    stats = selector.get_selection_stats(train_labels)
    print(f"\nSelection statistics:")
    print(f"  Total selected: {stats['n_selected']}")
    print(f"  Selection ratio: {stats['selection_ratio']:.2%}")
    print(f"  Class distribution: {stats['class_distribution']}")

    # Test incremental selection
    print("\n" + "-" * 50)
    print("Testing incremental selection...")
    # Create a new randomly initialized model for incremental selection
    incremental_model = get_model("MNIST", num_classes=10)
    new_indices = selector.select(
        train_data=train_data,
        train_labels=train_labels,
        model=incremental_model,
        incremental=True,
        current_indices=selected_indices,
        verbose=True
    )
    print(f"Previous selection: {len(selected_indices)} samples")
    print(f"New selection: {len(new_indices)} samples")
    print(f"Added {len(new_indices) - len(selected_indices)} samples")

    print("\n[PASS] CSReL coreset test passed!")


def test_save_load():
    """Test save and load functionality."""
    print("\n" + "=" * 50)
    print("Testing save/load functionality...")
    print("=" * 50)

    # Create simple configuration
    config = CSReLConfig(
        dataset="MNIST",
        num_classes=10,
        num_epochs=2,
        batch_size=32,
        selection_ratio=0.1,
        device="cpu"
    )

    # Create dummy data
    n_samples = 50
    train_data = torch.randn(n_samples, 1, 28, 28)
    train_labels = torch.randint(0, 10, (n_samples,))

    # Create and train selector
    selector = CSReLCoreset(config=config)
    selector.train_reference_model(train_data, train_labels, verbose=False)
    from src.models import get_model
    model = get_model("MNIST", num_classes=10)
    selector.select(train_data, train_labels, model=model, verbose=False)

    # Save
    save_path = "test_csrel_checkpoint.pth"
    selector.save(save_path)
    print(f"\n[PASS] Saved selector to {save_path}")

    # Create new selector and load
    new_selector = CSReLCoreset(config=config)
    from src.models import get_model
    model = get_model("MNIST", num_classes=10)
    new_selector.load(save_path, model)
    print("[PASS] Loaded selector from checkpoint")

    # Verify loaded state
    assert torch.equal(selector.selected_indices, new_selector.selected_indices)
    assert torch.equal(selector.reference_losses, new_selector.reference_losses)
    print("[PASS] State verification passed")

    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"[PASS] Cleaned up {save_path}")

    print("\n[PASS] Save/load test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("CSReL Implementation Tests")
    print("=" * 50)

    try:
        test_selection_functions()
        test_csrel_coreset()
        test_save_load()

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
