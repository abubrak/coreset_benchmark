"""
结果可视化脚本

用于加载、分析和可视化实验结果。
支持数据摘要和持续学习实验的结果比较。

主要功能：
1. load_results(): 从日志目录加载结果
2. plot_data_summarization(): 绘制数据摘要结果比较图
3. plot_continual_learning(): 绘制持续学习结果比较图
4. create_comparison_table(): 创建比较表格
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def load_results(log_dir: str, pattern: str = "*.json") -> List[Dict]:
    """
    从日志目录加载实验结果

    参数:
        log_dir: 日志目录路径
        pattern: 文件匹配模式

    返回:
        结果字典列表
    """
    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"警告: 日志目录不存在: {log_dir}")
        return []

    results = []
    for json_file in log_path.glob(pattern):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                # 添加文件路径信息
                result['_source_file'] = str(json_file)
                results.append(result)
        except Exception as e:
            print(f"警告: 无法加载文件 {json_file}: {e}")

    return results


def filter_results(
    results: List[Dict],
    dataset: Optional[str] = None,
    method: Optional[str] = None,
    selection_ratio: Optional[float] = None
) -> List[Dict]:
    """
    根据条件过滤结果

    参数:
        results: 结果列表
        dataset: 数据集名称
        method: 方法名称
        selection_ratio: 选择比例

    返回:
        过滤后的结果列表
    """
    filtered = results

    if dataset is not None:
        filtered = [r for r in filtered if r.get('dataset') == dataset]

    if method is not None:
        filtered = [r for r in filtered if r.get('method') == method]

    if selection_ratio is not None:
        filtered = [r for r in filtered
                   if abs(r.get('selection_ratio', 0) - selection_ratio) < 1e-6]

    return filtered


def aggregate_results(results: List[Dict], group_by: List[str]) -> pd.DataFrame:
    """
    聚合多次运行的结果

    参数:
        results: 结果列表
        group_by: 分组字段列表

    返回:
        聚合后的DataFrame
    """
    df = pd.DataFrame(results)

    if df.empty:
        return df

    # 计算统计量
    numeric_cols = ['test_acc_full', 'test_acc_coreset', 'performance_drop',
                   'selection_time', 'train_time_full', 'train_time_coreset']

    # 只保留存在的列
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    grouped = df.groupby(group_by)[numeric_cols].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ])

    return grouped


def plot_data_summarization(
    results: List[Dict],
    output_dir: str = "./figures",
    dataset: Optional[str] = None,
    metrics: List[str] = ['test_acc_coreset', 'performance_drop']
):
    """
    绘制数据摘要实验结果比较图

    参数:
        results: 结果列表
        output_dir: 输出目录
        dataset: 指定数据集（None表示使用所有）
        metrics: 要绘制的指标列表
    """
    if not results:
        print("没有可绘制的结果")
        return

    # 过滤数据集
    if dataset is not None:
        plot_results = filter_results(results, dataset=dataset)
        datasets = [dataset]
    else:
        plot_results = results
        datasets = sorted(set(r.get('dataset') for r in results))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 为每个数据集创建图表
    for ds in datasets:
        ds_results = filter_results(plot_results, dataset=ds)

        if not ds_results:
            continue

        # 准备数据
        methods = sorted(set(r.get('method') for r in ds_results))
        ratios = sorted(set(r.get('selection_ratio') for r in ds_results))

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{ds} 数据集 - Coreset选择方法比较', fontsize=16, fontweight='bold')

        # 1. 测试准确率比较（按方法）
        ax1 = axes[0, 0]
        for method in methods:
            method_results = [r for r in ds_results if r.get('method') == method]
            if method_results:
                ratios_data = [r.get('selection_ratio') * 100 for r in method_results]
                accs = [r.get('test_acc_coreset') for r in method_results]
                ax1.plot(ratios_data, accs, marker='o', label=method.upper(), linewidth=2)

        ax1.set_xlabel('Coreset大小 (%)', fontsize=11)
        ax1.set_ylabel('测试准确率 (%)', fontsize=11)
        ax1.set_title('不同选择比例下的准确率', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. 性能下降比较（按方法）
        ax2 = axes[0, 1]
        for method in methods:
            method_results = [r for r in ds_results if r.get('method') == method]
            if method_results:
                ratios_data = [r.get('selection_ratio') * 100 for r in method_results]
                drops = [r.get('performance_drop') for r in method_results]
                ax2.plot(ratios_data, drops, marker='s', label=method.upper(), linewidth=2)

        ax2.set_xlabel('Coreset大小 (%)', fontsize=11)
        ax2.set_ylabel('性能下降 (%)', fontsize=11)
        ax2.set_title('性能下降比较（越低越好）', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. 选择时间比较
        ax3 = axes[1, 0]
        for method in methods:
            method_results = [r for r in ds_results if r.get('method') == method]
            if method_results:
                ratios_data = [r.get('selection_ratio') * 100 for r in method_results]
                times = [r.get('selection_time', 0) for r in method_results]
                ax3.plot(ratios_data, times, marker='^', label=method.upper(), linewidth=2)

        ax3.set_xlabel('Coreset大小 (%)', fontsize=11)
        ax3.set_ylabel('选择时间 (秒)', fontsize=11)
        ax3.set_title('计算时间比较', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # 4. 训练时间比较
        ax4 = axes[1, 1]
        for method in methods:
            method_results = [r for r in ds_results if r.get('method') == method]
            if method_results:
                ratios_data = [r.get('selection_ratio') * 100 for r in method_results]
                times = [r.get('train_time_coreset', 0) for r in method_results]
                ax4.plot(ratios_data, times, marker='d', label=method.upper(), linewidth=2)

        ax4.set_xlabel('Coreset大小 (%)', fontsize=11)
        ax4.set_ylabel('训练时间 (秒)', fontsize=11)
        ax4.set_title('训练时间比较', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ds}_data_summarization_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {filepath}")

        plt.close()


def plot_continual_learning(
    results: List[Dict],
    output_dir: str = "./figures",
    dataset: Optional[str] = None
):
    """
    绘制持续学习实验结果比较图

    参数:
        results: 结果列表
        output_dir: 输出目录
        dataset: 指定数据集
    """
    if not results:
        print("没有可绘制的结果")
        return

    # 过滤数据集
    if dataset is not None:
        plot_results = filter_results(results, dataset=dataset)
        datasets = [dataset]
    else:
        plot_results = results
        datasets = sorted(set(r.get('dataset') for r in results))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 为每个数据集创建图表
    for ds in datasets:
        ds_results = [r for r in plot_results if r.get('dataset') == ds]

        if not ds_results:
            continue

        # 准备数据
        methods = sorted(set(r.get('selection_method') for r in ds_results))

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{ds} 数据集 - 持续学习性能', fontsize=16, fontweight='bold')

        # 1. 平均准确率比较
        ax1 = axes[0, 0]
        for method in methods:
            method_results = [r for r in ds_results if r.get('selection_method') == method]
            if method_results:
                avg_accs = [r.get('average_accuracy', 0) for r in method_results]
                x_pos = range(len(method_results))
                ax1.plot(x_pos, avg_accs, marker='o', label=method.upper(), linewidth=2)

        ax1.set_xlabel('实验运行', fontsize=11)
        ax1.set_ylabel('平均准确率 (%)', fontsize=11)
        ax1.set_title('平均准确率比较', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. 遗忘度量比较
        ax2 = axes[0, 1]
        for method in methods:
            method_results = [r for r in ds_results if r.get('selection_method') == method]
            if method_results:
                forgettings = [r.get('forgetting_measure', 0) for r in method_results]
                x_pos = range(len(method_results))
                ax2.plot(x_pos, forgettings, marker='s', label=method.upper(), linewidth=2)

        ax2.set_xlabel('实验运行', fontsize=11)
        ax2.set_ylabel('遗忘度量 (%)', fontsize=11)
        ax2.set_title('遗忘度量比较（越低越好）', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. 准确率矩阵热图（选择第一个结果）
        ax3 = axes[1, 0]
        if ds_results:
            first_result = ds_results[0]
            acc_matrix = np.array(first_result.get('accuracy_matrix', []))
            if acc_matrix.size > 0:
                num_tasks = acc_matrix.shape[0]
                im = ax3.imshow(acc_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
                ax3.set_xticks(range(num_tasks))
                ax3.set_yticks(range(num_tasks))
                ax3.set_xlabel('任务ID', fontsize=11)
                ax3.set_ylabel('学习任务', fontsize=11)
                ax3.set_title(f'准确率矩阵 - {first_result.get("selection_method", "unknown").upper()}',
                             fontsize=12)

                # 添加数值标注
                for i in range(num_tasks):
                    for j in range(num_tasks):
                        if j <= i:
                            text = ax3.text(j, i, f'{acc_matrix[i, j]:.1f}',
                                          ha="center", va="center", color="black", fontsize=8)

                plt.colorbar(im, ax=ax3, label='准确率 (%)')

        # 4. 任务间准确率演变
        ax4 = axes[1, 1]
        if ds_results:
            first_result = ds_results[0]
            acc_matrix = np.array(first_result.get('accuracy_matrix', []))
            if acc_matrix.size > 0:
                num_tasks = acc_matrix.shape[0]
                for task_id in range(num_tasks):
                    # 获取该任务在各个学习阶段的准确率
                    task_accuracies = []
                    for learned_task in range(task_id, num_tasks):
                        task_accuracies.append(acc_matrix[learned_task, task_id])

                    x_pos = range(task_id, num_tasks)
                    ax4.plot(x_pos, task_accuracies, marker='o',
                            label=f'Task {task_id}', linewidth=2, alpha=0.7)

                ax4.set_xlabel('已学习的任务数', fontsize=11)
                ax4.set_ylabel('准确率 (%)', fontsize=11)
                ax4.set_title('各任务准确率演变', fontsize=12)
                ax4.legend(fontsize=8, ncol=2)
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ds}_continual_learning_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {filepath}")

        plt.close()


def create_comparison_table(
    results: List[Dict],
    output_dir: str = "./tables",
    experiment_type: str = "data_summarization"
):
    """
    创建结果比较表格

    参数:
        results: 结果列表
        output_dir: 输出目录
        experiment_type: 实验类型 ('data_summarization' 或 'continual_learning')
    """
    if not results:
        print("没有可创建表格的结果")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if experiment_type == "data_summarization":
        # 数据摘要实验表格
        df = pd.DataFrame(results)

        # 选择关键列
        key_cols = ['dataset', 'method', 'selection_ratio', 'coreset_size',
                   'test_acc_full', 'test_acc_coreset', 'performance_drop',
                   'selection_time', 'train_time_coreset']

        # 只保留存在的列
        table_cols = [col for col in key_cols if col in df.columns]
        table_df = df[table_cols].copy()

        # 格式化数值列
        if 'selection_ratio' in table_df.columns:
            table_df['selection_ratio'] = table_df['selection_ratio'] * 100

        # 重命名列
        column_mapping = {
            'dataset': 'Dataset',
            'method': 'Method',
            'selection_ratio': 'Ratio (%)',
            'coreset_size': 'Size',
            'test_acc_full': 'Full Acc (%)',
            'test_acc_coreset': 'Coreset Acc (%)',
            'performance_drop': 'Drop (%)',
            'selection_time': 'Sel Time (s)',
            'train_time_coreset': 'Train Time (s)'
        }

        table_df.rename(columns=column_mapping, inplace=True)

        # 按数据集、方法、选择比例排序
        if 'Dataset' in table_df.columns and 'Method' in table_df.columns and 'Ratio (%)' in table_df.columns:
            table_df = table_df.sort_values(['Dataset', 'Method', 'Ratio (%)'])

        # 保存为CSV
        csv_file = os.path.join(output_dir, f"data_summarization_{timestamp}.csv")
        table_df.to_csv(csv_file, index=False, float_format='%.2f')
        print(f"表格已保存: {csv_file}")

        # 保存为LaTeX
        latex_file = os.path.join(output_dir, f"data_summarization_{timestamp}.tex")
        table_df.to_latex(latex_file, index=False, float_format='%.2f',
                         column_format='l' + 'c' * (len(table_df.columns) - 1))
        print(f"LaTeX表格已保存: {latex_file}")

        # 打印表格预览
        print("\n数据摘要实验结果表:")
        print(table_df.to_string(index=False))

    elif experiment_type == "continual_learning":
        # 持续学习实验表格
        df = pd.DataFrame(results)

        # 选择关键列
        key_cols = ['dataset', 'selection_method', 'memory_size',
                   'num_tasks', 'average_accuracy', 'forgetting_measure']

        # 只保留存在的列
        table_cols = [col for col in key_cols if col in df.columns]
        table_df = df[table_cols].copy()

        # 重命名列
        column_mapping = {
            'dataset': 'Dataset',
            'selection_method': 'Method',
            'memory_size': 'Memory',
            'num_tasks': 'Tasks',
            'average_accuracy': 'Avg Acc (%)',
            'forgetting_measure': 'Forgetting (%)'
        }

        table_df.rename(columns=column_mapping, inplace=True)

        # 按数据集、方法、内存大小排序
        if 'Dataset' in table_df.columns and 'Method' in table_df.columns:
            table_df = table_df.sort_values(['Dataset', 'Method'])

        # 保存为CSV
        csv_file = os.path.join(output_dir, f"continual_learning_{timestamp}.csv")
        table_df.to_csv(csv_file, index=False, float_format='%.2f')
        print(f"表格已保存: {csv_file}")

        # 保存为LaTeX
        latex_file = os.path.join(output_dir, f"continual_learning_{timestamp}.tex")
        table_df.to_latex(latex_file, index=False, float_format='%.2f',
                         column_format='l' + 'c' * (len(table_df.columns) - 1))
        print(f"LaTeX表格已保存: {latex_file}")

        # 打印表格预览
        print("\n持续学习实验结果表:")
        print(table_df.to_string(index=False))


def print_summary_statistics(results: List[Dict]):
    """
    打印结果摘要统计

    参数:
        results: 结果列表
    """
    if not results:
        print("没有可统计的结果")
        return

    print("\n" + "=" * 80)
    print("结果摘要统计")
    print("=" * 80)

    # 按数据集分组
    datasets = sorted(set(r.get('dataset') for r in results))

    for dataset in datasets:
        ds_results = [r for r in results if r.get('dataset') == dataset]
        print(f"\n数据集: {dataset}")
        print("-" * 60)

        # 按方法分组
        methods = sorted(set(r.get('method') or r.get('selection_method') for r in ds_results))

        for method in methods:
            method_results = [r for r in ds_results
                            if (r.get('method') or r.get('selection_method')) == method]

            if method_results:
                print(f"\n  方法: {method.upper()} ({len(method_results)} 次运行)")

                # 打印统计信息
                if 'test_acc_coreset' in method_results[0]:
                    accs = [r.get('test_acc_coreset', 0) for r in method_results]
                    print(f"    测试准确率: {np.mean(accs):.2f}% (±{np.std(accs):.2f}%)")

                if 'performance_drop' in method_results[0]:
                    drops = [r.get('performance_drop', 0) for r in method_results]
                    print(f"    性能下降: {np.mean(drops):.2f}% (±{np.std(drops):.2f}%)")

                if 'average_accuracy' in method_results[0]:
                    avg_accs = [r.get('average_accuracy', 0) for r in method_results]
                    print(f"    平均准确率: {np.mean(avg_accs):.2f}% (±{np.std(avg_accs):.2f}%)")

                if 'forgetting_measure' in method_results[0]:
                    forgettings = [r.get('forgetting_measure', 0) for r in method_results]
                    print(f"    遗忘度量: {np.mean(forgettings):.2f}% (±{np.std(forgettings):.2f}%)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='结果可视化脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 输入参数
    parser.add_argument('--log_dir', type=str, default='./results',
                       help='结果日志目录')
    parser.add_argument('--experiment_type', type=str, default='data_summarization',
                       choices=['data_summarization', 'continual_learning', 'both'],
                       help='实验类型')

    # 过滤参数
    parser.add_argument('--dataset', type=str, default=None,
                       help='指定数据集')
    parser.add_argument('--method', type=str, default=None,
                       help='指定方法')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='图表输出目录')
    parser.add_argument('--table_dir', type=str, default='./tables',
                       help='表格输出目录')

    # 绘图参数
    parser.add_argument('--dpi', type=int, default=300,
                       help='图表DPI')
    parser.add_argument('--style', type=str, default='paper',
                       choices=['paper', 'talk', 'poster'],
                       help='绘图样式')

    # 其他参数
    parser.add_argument('--summary_only', action='store_true',
                       help='仅打印摘要统计')
    parser.add_argument('--create_tables', action='store_true',
                       help='创建比较表格')

    args = parser.parse_args()

    # 设置绘图样式
    if args.style == 'talk':
        sns.set_context("talk", font_scale=1.2)
    elif args.style == 'poster':
        sns.set_context("poster", font_scale=1.4)
    else:
        sns.set_context("paper", font_scale=1.2)

    # 加载结果
    print(f"从目录加载结果: {args.log_dir}")
    all_results = load_results(args.log_dir)

    if not all_results:
        print("错误: 没有找到任何结果文件")
        return

    print(f"加载了 {len(all_results)} 个结果文件")

    # 过滤结果
    filtered_results = filter_results(
        all_results,
        dataset=args.dataset,
        method=args.method
    )

    print(f"过滤后剩余 {len(filtered_results)} 个结果")

    if not filtered_results:
        print("错误: 过滤后没有结果")
        return

    # 打印摘要统计
    print_summary_statistics(filtered_results)

    if args.summary_only:
        return

    # 创建表格
    if args.create_tables:
        if args.experiment_type in ['data_summarization', 'both']:
            create_comparison_table(
                filtered_results,
                output_dir=args.table_dir,
                experiment_type='data_summarization'
            )

        if args.experiment_type in ['continual_learning', 'both']:
            create_comparison_table(
                filtered_results,
                output_dir=args.table_dir,
                experiment_type='continual_learning'
            )

    # 绘制图表
    if args.experiment_type in ['data_summarization', 'both']:
        plot_data_summarization(
            filtered_results,
            output_dir=args.output_dir,
            dataset=args.dataset
        )

    if args.experiment_type in ['continual_learning', 'both']:
        plot_continual_learning(
            filtered_results,
            output_dir=args.output_dir,
            dataset=args.dataset
        )

    print("\n可视化完成!")


if __name__ == '__main__':
    main()
