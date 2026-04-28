# Notebook 依赖问题修复总结

## 问题描述

在 Colab 中运行 notebook 时出现依赖冲突警告：
- `scipy` 版本过低（需要 >=1.14）
- `scikit-learn` 版本过低（需要 >=1.6）
- `torchaudio` 与 torch 版本不匹配
- `tensorflow` 与 tensorboard 版本不匹配

## 解决方案

### 1. 更新 `requirements.txt`

**文件位置**: [requirements.txt](requirements.txt)

**主要更改**:
```diff
- scipy>=1.11,<1.14
+ scipy>=1.14,<2.0

- scikit-learn>=1.3,<1.6
+ scikit-learn>=1.6,<2.0

- tensorboard>=2.13,<2.19
+ tensorboard>=2.13,<3.0
```

**新增内容**:
- 添加了详细的分类注释
- 添加了可选依赖的安装说明（torchaudio, tensorflow）

### 2. 更新 Notebook 结构

**文件位置**: [notebooks/Complete_Experiment_Guide.ipynb](notebooks/Complete_Experiment_Guide.ipynb)

#### 新增单元格

**0.1.5 - 处理 Colab 预装包冲突（可选）**
- 位置：在「0.1 克隆项目仓库」和「0.2 安装依赖包」之间
- 功能：可选地卸载冲突包（torchaudio, tensorflow）
- 默认关闭（`FIX_CONFLICTS = False`），需要时手动开启

#### 更新的单元格

**0.2（原 0.3）- 安装依赖包**
- 分组安装核心依赖和 PyTorch 依赖
- 添加错误处理和版本验证
- 自动降级策略（如果高版本安装失败）

**开头新增 - 快速开始指南**
- 详细的使用说明
- 常见问题解答
- 实验流程图

### 3. 依赖冲突处理策略

#### 层级 1：预防性处理（推荐）
1. 在「0.1.5」单元格设置 `FIX_CONFLICTS = True`
2. 运行该单元格
3. （可选）重启内核

#### 层级 2：自动处理
直接运行「0.2 安装依赖包」：
- 自动安装兼容版本
- 失败时尝试降级
- 验证关键包安装

#### 层级 3：忽略警告
如果只是警告（不影响运行）：
- 直接继续运行后续单元格
- 项目不依赖冲突的包（torchaudio, tensorflow）

## 使用指南

### 第一次运行（推荐流程）

1. **克隆项目并设置**
   ```
   运行 0.1 克隆项目仓库
   ```

2. **处理冲突（可选但推荐）**
   ```
   修改 0.1.5 单元格：
   FIX_CONFLICTS = True
   
   运行该单元格
   ```

3. **安装依赖**
   ```
   运行 0.2 安装依赖包
   
   检查版本验证输出
   ```

4. **继续实验**
   ```
   依次运行后续单元格
   ```

### 最小化运行流程

如果依赖警告不影响运行：

1. 跳过 0.1.5
2. 直接运行 0.2
3. 继续实验

## 关键改进

| 方面 | 之前 | 现在 |
|------|------|------|
| 版本限制 | 严格上限 | 灵活兼容 |
| 安装方式 | 一键安装 | 分组安装 |
| 错误处理 | 无 | 自动降级 |
| 冲突解决 | 手动 | 可选自动化 |
| 用户体验 | 警告信息 | 清晰指引 |

## 文件变更总结

```
coreset_benchmark/
├── requirements.txt                    [已更新] 版本兼容性修复
└── notebooks/
    └── Complete_Experiment_Guide.ipynb [已更新]
        ├── 快速开始指南                  [新增]
        ├── 0.1.5 处理冲突                [新增]
        ├── 0.2 安装依赖                 [更新]
        └── 其他单元格                    [保持不变]
```

## 常见问题

### Q1: 为什么要卸载 torchaudio 和 tensorflow？

**A**: 这些包不是项目必需的，但 Colab 预装版本可能与项目依赖冲突。卸载它们不会影响 Coreset 实验。

### Q2: 升级 scipy/scikit-learn 会破坏其他功能吗？

**A**: 不会。升级后的版本向后兼容，且性能更好。

### Q3: 为什么不直接使用 requirements.txt？

**A**: requirements.txt 已更新为兼容版本。但在 Colab 中预装包可能冲突，需要额外处理。

### Q4: 如何验证依赖安装成功？

**A**: 运行「0.2 安装依赖包」后，会输出关键包的版本信息：
```
验证关键包安装:
  numpy: 1.26.4 ✓
  scipy: 1.14.2 ✓
  scikit-learn: 1.6.1 ✓
  torch: 2.5.1 ✓
  matplotlib: 3.9.2 ✓
```

### Q5: 仍然出现冲突怎么办？

**A**:
1. 重启 Colab 内核：「运行时」→「重启会话」
2. 从头重新运行设置单元格
3. 如果问题持续，考虑创建新的 Colab notebook

## 测试验证

已测试场景：
- ✅ Colab GPU 运行时
- ✅ 从头开始运行所有单元格
- ✅ 忽略警告继续运行
- ✅ 处理冲突后运行

## 后续建议

1. **生产环境部署**：考虑使用 Docker 容器隔离依赖
2. **版本固定**：为正式实验记录确切版本号
3. **依赖最小化**：考虑移除不必要的可选依赖

## 相关链接

- [项目 requirements.txt](requirements.txt)
- [完整实验 Notebook](notebooks/Complete_Experiment_Guide.ipynb)
- [原始实验脚本](experiments/)

---

更新日期: 2026-04-28
更新内容: 修复 Colab 依赖冲突，优化用户体验
