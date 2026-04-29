"""
Unit tests for continual learning coreset adapters.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coreset.continual_adapters import BCSRContinualAdapter


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def device():
    """Get test device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sample_data(device):
    """Create sample data for testing."""
    torch.manual_seed(42)
    n_samples = 100
    data = torch.randn(n_samples, 1, 28, 28).to(device)  # Use 28x28 for MNIST model
    labels = torch.randint(0, 5, (n_samples,)).to(device)
    return data, labels


@pytest.fixture
def sample_model(device):
    """Create sample model for testing."""
    model = SimpleCNN(num_classes=5).to(device)
    return model


def test_bcsr_adapter_initialization(device):
    """Test BCSR adapter can be initialized."""
    adapter = BCSRContinualAdapter(device=device)
    assert adapter is not None
    assert adapter.device == device


def test_bcsr_adapter_select(sample_data, sample_model, device):
    """Test BCSR adapter can select samples."""
    data, labels = sample_data
    adapter = BCSRContinualAdapter(device=device, num_outer_steps=2)

    num_samples = 20
    selected_data, selected_labels = adapter.select(
        data=data,
        labels=labels,
        num_samples=num_samples,
        model=sample_model
    )

    # Check output shapes
    assert selected_data.shape[0] == num_samples
    assert selected_labels.shape[0] == num_samples
    assert selected_data.device.type == device
    assert selected_labels.device.type == device


def test_bcsr_adapter_select_size_larger_than_data(sample_data, sample_model, device):
    """Test BCSR adapter handles num_samples > len(data)."""
    data, labels = sample_data
    adapter = BCSRContinualAdapter(device=device)

    num_samples = 200  # Larger than data size (100)
    selected_data, selected_labels = adapter.select(
        data=data,
        labels=labels,
        num_samples=num_samples,
        model=sample_model
    )

    # Should return all data
    assert selected_data.shape[0] == data.shape[0]
    assert selected_labels.shape[0] == labels.shape[0]


@pytest.fixture
def csrel_adapter(device):
    """Create CSReL adapter for testing."""
    from src.coreset.continual_adapters import CSReLContinualAdapter
    return CSReLContinualAdapter(
        num_epochs=2,  # Reduced for testing
        device=device
    )


def test_csrel_adapter_select(sample_data, sample_model, csrel_adapter, device):
    """Test CSReL adapter can select samples."""
    data, labels = sample_data

    num_samples = 20
    selected_data, selected_labels = csrel_adapter.select(
        data=data,
        labels=labels,
        num_samples=num_samples,
        model=sample_model
    )

    # Check output shapes
    assert selected_data.shape[0] <= num_samples  # CSReL uses ratio
    assert selected_labels.shape[0] == selected_data.shape[0]
    assert selected_data.device.type == device
    assert selected_labels.device.type == device


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
