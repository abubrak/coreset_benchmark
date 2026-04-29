"""快速测试脚本，验证MNIST修复是否有效"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.cnn import CNN_MNIST
from src.models.resnet import ResNet18

def test_mnist_feature_extraction():
    """测试MNIST特征提取是否正常工作"""
    print("Testing MNIST feature extraction...")

    # 创建MNIST模型
    model = CNN_MNIST(num_classes=10)
    model.eval()

    # 创建MNIST输入 (1, 1, 28, 28)
    x = torch.randn(4, 1, 28, 28)

    # 测试正常前向传播
    with torch.no_grad():
        output = model(x)
        print(f"  Normal forward pass output shape: {output.shape}")
        assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"

    # 测试特征提取
    def extract_features(model, x):
        # 第一个卷积块
        x = F.relu(model.bn1(model.conv1(x)))
        x = model.pool(x)
        # 第二个卷积块
        x = F.relu(model.bn2(model.conv2(x)))
        x = model.pool(x)
        # 第三个卷积块
        x = F.relu(model.bn3(model.conv3(x)))
        x = model.pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        return x

    with torch.no_grad():
        features = extract_features(model, x)
        print(f"  Feature extraction output shape: {features.shape}")
        assert features.shape == (4, 128 * 3 * 3), f"Expected (4, 1152), got {features.shape}"

    print("  [PASS] MNIST feature extraction works!")

def test_cifar10_feature_extraction():
    """测试CIFAR-10特征提取是否正常工作"""
    print("Testing CIFAR-10 feature extraction...")

    # 创建CIFAR-10模型
    model = ResNet18(num_classes=10)
    model.fc = nn.Identity()
    model.eval()

    # 创建CIFAR-10输入 (3, 32, 32)
    x = torch.randn(4, 3, 32, 32)

    # 测试特征提取
    with torch.no_grad():
        features = model(x)
        print(f"  Feature extraction output shape: {features.shape}")
        # ResNet18的最终特征维度是512
        assert features.shape == (4, 512), f"Expected (4, 512), got {features.shape}"

    print("  [PASS] CIFAR-10 feature extraction works!")

if __name__ == '__main__':
    test_mnist_feature_extraction()
    test_cifar10_feature_extraction()
    print("\n[SUCCESS] All feature extraction tests passed!")
