#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试路径检测功能

这个脚本验证路径自动检测是否正常工作
"""

import sys
from pathlib import Path


def test_path_detection():
    """测试路径检测"""
    print("=" * 60)
    print("路径检测测试")
    print("=" * 60)

    # 测试环境检测
    in_colab = 'google.colab' in sys.modules
    print(f"\n当前环境: {'Colab' if in_colab else '本地'}")

    # 测试项目路径检测
    def get_project_path():
        """自动检测项目路径"""
        if 'google.colab' in sys.modules:
            # Colab环境
            return Path('/content/coreset_benchmark')
        else:
            # 本地环境
            return Path(__file__).parent.parent.absolute()

    project_root = get_project_path()
    print(f"检测到的项目根目录: {project_root}")
    print(f"路径是否存在: {project_root.exists()}")

    # 测试路径添加到sys.path
    project_str = str(project_root)
    if project_str in sys.path:
        print(f"[OK] 项目路径已在sys.path中")
    else:
        sys.path.insert(0, project_str)
        print(f"[OK] 项目路径已添加到sys.path")

    # 测试相对路径解析
    test_paths = [
        'src',
        'notebooks',
        'data',
        'results',
    ]

    print("\n测试相对路径解析:")
    for path_str in test_paths:
        full_path = project_root / path_str
        exists = full_path.exists()
        status = "[OK]" if exists else "[FAIL]"
        print(f"  {status} {path_str}: {full_path}")

    # 测试导入
    print("\n测试模块导入:")
    try:
        from src import data, coreset, models, utils
        print("  [OK] src.data")
        print("  [OK] src.coreset")
        print("  [OK] src.models")
        print("  [OK] src.utils")
    except ImportError as e:
        print(f"  [FAIL] 导入失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_path_detection()
