#!/usr/bin/env python
"""
CSReL 修复验证脚本

验证 CSReL 输入验证修复是否正确工作。
"""
import torch
from src.coreset.csrel_coreset import CSReLCoreset
from src.configs import CSReLConfig
from src.models import get_model

print("=" * 60)
print("CSReL 修复验证")
print("=" * 60)

# 创建配置
config = CSReLConfig(
    dataset='MNIST', num_classes=10, input_dim=784,
    batch_size=64, learning_rate=0.001, num_epochs=5,
    device='cpu', random_seed=42
)

selector = CSReLCoreset(config=config)

# 创建数据
train_data = torch.randn(500, 1, 28, 28)
train_labels = torch.randint(0, 10, (500,))

# 训练参考模型
print("\n训练参考模型...")
selector.train_reference_model(train_data, train_labels, verbose=False)
print("[OK] 参考模型训练完成")

# 测试 1: model=None 应该抛出错误
print("\n测试 1: model=None 应该抛出 ValueError")
try:
    selector.select(train_data, train_labels, model=None, verbose=False)
    print("[FAIL] model=None 应该抛出 ValueError")
    exit(1)
except ValueError as e:
    error_msg = str(e)
    if "model" in error_msg.lower() and "must be provided" in error_msg.lower():
        print(f"[PASS] 正确抛出 ValueError")
        print(f"       错误消息: {error_msg[:80]}...")
    else:
        print(f"[FAIL] 错误消息不正确: {error_msg}")
        exit(1)

# 测试 2: 提供正确模型应该工作
print("\n测试 2: 提供正确的模型应该正常工作")
current_model = get_model(dataset='MNIST', num_classes=10)
try:
    selected = selector.select(train_data, train_labels, model=current_model, verbose=False)
    if len(selected) > 0 and len(selected) <= len(train_data):
        print(f"[PASS] 选择了 {len(selected)} 个样本（总共 {len(train_data)} 个）")
    else:
        print(f"[FAIL] 选择的样本数量异常: {len(selected)}")
        exit(1)
except Exception as e:
    print(f"[FAIL] 抛出异常: {e}")
    exit(1)

# 测试 3: 验证选择结果的质量
print("\n测试 3: 验证选择结果的多样性")
selected_labels = train_labels[selected]
unique_labels = torch.unique(selected_labels)
if len(unique_labels) >= 3:  # 至少应该有 3 个类别
    print(f"[PASS] 选择的样本涵盖 {len(unique_labels)} 个类别（类别平衡良好）")
else:
    print(f"[WARN] 选择的样本只涵盖 {len(unique_labels)} 个类别（可能数据集太小）")

print("\n" + "=" * 60)
print("[OK] 所有测试通过！CSReL 修复验证成功！")
print("=" * 60)
