"""
Integration test for BCSR in continual learning.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.continual_learning import CoresetBuffer


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_bcsr_in_buffer():
    """Test BCSR method works in CoresetBuffer."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create buffer
    buffer = CoresetBuffer(
        memory_size=100,
        input_shape=(1, 28, 28),
        num_classes=10,
        device=device
    )

    # Create sample data
    torch.manual_seed(42)
    data = torch.randn(50, 1, 28, 28).to(device)
    labels = torch.randint(0, 5, (50,)).to(device)

    # Create model
    model = SimpleCNN(num_classes=5).to(device)

    # Test BCSR selection
    selected_data, selected_labels = buffer.select_coreset(
        data=data,
        labels=labels,
        num_samples=10,
        method='bcsr',
        model=model
    )

    assert selected_data.shape[0] == 10
    assert selected_labels.shape[0] == 10
    print("BCSR integration test passed!")


if __name__ == '__main__':
    test_bcsr_in_buffer()
