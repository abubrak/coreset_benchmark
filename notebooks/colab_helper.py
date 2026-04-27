#!/usr/bin/env python3
"""
Colab辅助脚本

这个脚本提供了在Google Colab中使用项目时的辅助功能
"""

import sys
import os
from pathlib import Path


def setup_project_path():
    """
    设置项目路径

    在Colab中，如果从GitHub克隆项目，项目会在/content/coreset_benchmark
    在本地，项目路径是当前目录的父目录

    Returns:
        Path: 项目根目录路径
    """
    if 'google.colab' in sys.modules:
        # Colab环境
        project_path = Path('/content/coreset_benchmark')

        # 如果项目不存在，提示用户克隆
        if not project_path.exists():
            print("⚠ 项目目录不存在")
            print("请先运行以下命令克隆项目：")
            print("!git clone https://github.com/yourusername/coreset_benchmark.git /content/coreset_benchmark")
            return None
    else:
        # 本地环境
        # 假设脚本在notebooks目录下
        project_path = Path(__file__).parent.parent.absolute()

    # 添加项目路径到sys.path
    project_str = str(project_path)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)
        print(f"✓ 已添加项目路径到sys.path: {project_str}")

    return project_path


def verify_project_structure(project_root=None):
    """
    验证项目结构是否完整

    Args:
        project_root: 项目根目录路径，如果为None则自动检测

    Returns:
        bool: 项目结构是否完整
    """
    if project_root is None:
        project_root = setup_project_path()

    if project_root is None:
        return False

    print(f"\n验证项目结构...")
    print(f"项目根目录: {project_root}")

    # 检查必需的目录
    required_dirs = [
        'src',
        'notebooks',
        'data',
    ]

    required_files = [
        'src/__init__.py',
        'src/data.py',
        'src/coreset.py',
        'src/models.py',
        'src/utils.py',
    ]

    all_exist = True

    # 检查目录
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"✓ 目录存在: {dir_path}")
        else:
            print(f"✗ 目录缺失: {dir_path}")
            all_exist = False

    # 检查文件
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            print(f"✓ 文件存在: {file_path}")
        else:
            print(f"✗ 文件缺失: {file_path}")
            all_exist = False

    if all_exist:
        print("\n✓ 项目结构完整")
    else:
        print("\n✗ 项目结构不完整，请检查")

    return all_exist


def print_environment_info():
    """打印环境信息"""
    print("\n" + "=" * 60)
    print("环境信息")
    print("=" * 60)

    # 检查是否在Colab中
    if 'google.colab' in sys.modules:
        print("✓ Google Colab环境")
    else:
        print("✓ 本地环境")

    # Python版本
    print(f"Python版本: {sys.version}")

    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA可用: 是")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        else:
            print(f"CUDA可用: 否")
    except ImportError:
        print("PyTorch: 未安装")

    # 项目路径
    project_root = setup_project_path()
    if project_root:
        print(f"项目路径: {project_root}")

    print("=" * 60)


def main():
    """主函数 - 用于测试"""
    print_environment_info()
    verify_project_structure()


if __name__ == "__main__":
    main()
