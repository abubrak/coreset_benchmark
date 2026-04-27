#!/usr/bin/env python3
"""
Colab环境设置脚本

这个脚本用于在Google Colab环境中设置必要的依赖和环境配置。
"""

import subprocess
import sys
import os
from pathlib import Path


def get_project_path():
    """
    自动检测项目路径

    在Colab环境中返回 /content/coreset_benchmark
    在本地环境中返回当前工作目录
    """
    if 'google.colab' in sys.modules:
        # Colab环境
        return Path('/content/coreset_benchmark')
    else:
        # 本地环境，返回脚本所在目录的父目录
        return Path(__file__).parent.parent.absolute()


def install_dependencies():
    """安装必要的Python包"""
    print("=" * 60)
    print("开始安装依赖包...")
    print("=" * 60)

    packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "ipywidgets>=7.6.0",
    ]

    for package in packages:
        print(f"\n正在安装 {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", package
            ])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"✗ {package} 安装失败: {e}")
            continue

    print("\n" + "=" * 60)
    print("依赖包安装完成！")
    print("=" * 60)


def check_gpu():
    """检查GPU可用性"""
    print("\n" + "=" * 60)
    print("检查GPU可用性...")
    print("=" * 60)

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ GPU可用！")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  PyTorch版本: {torch.__version__}")
        else:
            print("✗ GPU不可用，将使用CPU")
            print("  提示: 在Colab中，请点击 '运行时' -> '更改运行时类型' -> 选择 'GPU'")
    except ImportError:
        print("✗ PyTorch未安装，无法检查GPU")


def create_directory_structure():
    """创建必要的目录结构"""
    print("\n" + "=" * 60)
    print("创建目录结构...")
    print("=" * 60)

    # 获取项目根目录
    project_root = get_project_path()
    print(f"项目根目录: {project_root}")

    # 定义需要创建的目录
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "results",
        "results/summaries",
        "results/continual_learning",
        "results/benchmarks",
        "checkpoints",
        "logs",
        "figures",
    ]

    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")

    # 创建.gitkeep文件以保持空目录在git中
    for dir_path in directories:
        full_path = project_root / dir_path / ".gitkeep"
        if not full_path.exists():
            full_path.touch()
            print(f"✓ 创建.gitkeep: {dir_path}/.gitkeep")

    print("\n" + "=" * 60)
    print("目录结构创建完成！")
    print(f"基础路径: {project_root}")
    print("=" * 60)


def verify_installation():
    """验证关键包的安装"""
    print("\n" + "=" * 60)
    print("验证安装...")
    print("=" * 60)

    packages_to_check = {
        "numpy": "np",
        "scipy": "sp",
        "sklearn": "sklearn",
        "torch": "torch",
        "matplotlib": "plt",
        "pandas": "pd",
    }

    failed = []
    for package_name, import_name in packages_to_check.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} 未安装")
            failed.append(package_name)

    if failed:
        print(f"\n警告: 以下包未能正确安装: {', '.join(failed)}")
        return False
    else:
        print("\n所有必要的包都已正确安装！")
        return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Coreset Benchmark Colab环境设置")
    print("=" * 60)

    # 检查是否在Colab环境中
    in_colab = 'google.colab' in sys.modules
    if in_colab:
        print("✓ 检测到Google Colab环境")
    else:
        print("⚠ 未检测到Google Colab环境")
        print("  此脚本主要为Colab设计，但也可在其他环境中使用")

    # 执行设置步骤
    install_dependencies()
    check_gpu()
    create_directory_structure()

    # 验证安装
    success = verify_installation()

    print("\n" + "=" * 60)
    if success:
        print("✓ 环境设置完成！")
        print("  现在可以运行实验notebook了。")
    else:
        print("⚠ 环境设置完成，但部分包可能需要手动安装")
    print("=" * 60)

    # 显示下一步操作提示
    print("\n下一步:")
    print("1. 运行 'Data_Summarization_Experiment.ipynb' 进行数据摘要实验")
    print("2. 运行 'Continual_Learning_Experiment.ipynb' 进行持续学习实验")
    print("\n提示: 使用ipywidgets进行交互式参数调整")


if __name__ == "__main__":
    main()
