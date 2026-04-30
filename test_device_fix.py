"""
测试设备管理修复

验证BCSR适配器不会改变原始模型的设备
"""
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.coreset.continual_adapters import BCSRContinualAdapter
from src.models.cnn import CNN_MNIST

def test_device_management():
    """测试BCSR适配器的设备管理"""
    print("测试BCSR适配器设备管理修复")
    print("="*50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建模型并放在GPU上
    model = CNN_MNIST(num_classes=2).to(device)
    print(f"模型初始设备: {next(model.parameters()).device}")

    # 创建测试数据
    torch.manual_seed(42)
    data = torch.randn(50, 1, 28, 28).to(device)
    labels = torch.randint(0, 2, (50,)).to(device)

    # 创建BCSR适配器
    adapter = BCSRContinualAdapter(
        learning_rate_inner=0.01,
        learning_rate_outer=3.0,
        num_inner_steps=1,
        num_outer_steps=1,  # 减少到1次以加快测试
        beta=0.1,
        device=device
    )

    print(f"适配器设备: {adapter.device}")

    # 测试选择
    print("\n开始BCSR选择...")
    try:
        selected_data, selected_labels = adapter.select(
            data=data,
            labels=labels,
            num_samples=10,
            model=model
        )

        # 检查模型设备是否仍然是正确的
        model_device_after = next(model.parameters()).device
        print(f"选择后模型设备: {model_device_after}")

        if model_device_after == torch.device(device):
            print(f"[OK] 模型设备保持不变: {model_device_after}")
        else:
            print(f"[FAIL] 模型设备被改变: {model_device_after} != {torch.device(device)}")
            return False

        # 检查选择的数据是否在正确设备上
        print(f"选择的数据设备: {selected_data.device}")
        print(f"选择的标签设备: {selected_labels.device}")

        if selected_data.device == torch.device(device) and selected_labels.device == torch.device(device):
            print(f"[OK] 选择的数据在正确设备上")
        else:
            print(f"[FAIL] 选择的数据不在正确设备上")
            return False

        # 尝试使用模型进行前向传播（验证没有设备不匹配）
        print("\n验证模型仍然可以正常使用...")
        model.eval()
        with torch.no_grad():
            test_data = torch.randn(5, 1, 28, 28).to(device)
            outputs = model(test_data)
            print(f"[OK] 模型前向传播成功，输出形状: {outputs.shape}")

        print("\n[OK] 所有设备管理测试通过！")
        return True

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_device_management()
    sys.exit(0 if success else 1)
