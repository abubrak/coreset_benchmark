#!/bin/bash
# Colab设备管理修复测试脚本

echo "=========================================="
echo "设备管理修复验证"
echo "=========================================="

# 测试1: 快速设备管理测试
echo ""
echo "测试1: 设备管理验证"
echo "------------------------------------------"
python test_device_fix.py

if [ $? -eq 0 ]; then
    echo "[OK] 设备管理测试通过"
else
    echo "[FAIL] 设备管理测试失败"
    exit 1
fi

# 测试2: 小规模BCSR实验（原失败场景）
echo ""
echo "测试2: 小规模BCSR持续学习实验"
echo "------------------------------------------"
python experiments/continual_learning.py \
    --dataset MNIST \
    --num_tasks 2 \
    --num_classes_per_task 2 \
    --selection_method bcsr \
    --memory_size 1000 \
    --num_epochs 5 \
    --save_results

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[OK] 所有测试通过！修复成功！"
    echo "=========================================="
    echo ""
    echo "现在可以运行完整实验："
    echo ""
    echo "python experiments/continual_learning.py \\"
    echo "    --dataset MNIST --num_tasks 3 \\"
    echo "    --selection_method bcsr --memory_size 2000"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "[FAIL] 实验失败"
    echo "=========================================="
    echo "请检查错误信息并报告问题"
    exit 1
fi
