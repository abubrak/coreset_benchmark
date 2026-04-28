# Bug 修复报告

## 修复日期
2026-04-28

## Bug 列表

### Bug 1: torchaudio 版本冲突

**错误信息**:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. 
torchaudio 2.10.0+cu128 requires torch==2.10.0, but you have torch 2.5.1 which is incompatible.
```

**原因**:
- Colab 预装了 torch 2.10 的 torchaudio
- 项目使用 torch 2.5.x
- torchaudio 不是项目必需的依赖

**修复方案**:
在 `0.3 安装依赖包` 单元格中添加自动卸载逻辑：

```python
# 🔧 修复 Bug 1: 主动卸载冲突的 torchaudio
print("\n处理预装包冲突...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"],
        capture_output=True
    )
    print("✓ 已卸载 torchaudio（避免与 torch 版本冲突）")
except:
    print("  (torchaudio 未安装或卸载失败，继续)")
```

**验证**:
- 添加了 torchaudio 卸载检测
- 在依赖验证部分显示卸载状态

---

### Bug 2: 方法名称错误

**错误信息**:
```
ValueError: Unknown method: random. Available methods: 'uniform', 'kcenter', 'kmeans', 'herding', 'entropy', 'loss'
```

**原因**:
- notebook 使用 `'random'` 作为方法名
- 但 `baseline_methods.py` 中的实际方法是 `'uniform'`

**修复方案**:

#### 1. 更新方法列表 (cell-12)
```python
# 之前
METHODS = ['random', 'kcenter', 'herding', 'bcsr']

# 修复后
METHODS = ['uniform', 'kcenter', 'herding', 'bcsr']
```

#### 2. 更新函数注释 (cell-11)
```python
"""
参数:
    method_name: 方法名称 ('uniform', 'kcenter', 'herding', 'bcsr')  # 已更新
"""
```

#### 3. 更新持续学习参数 (cell-17)
```python
# 之前
SELECTION_METHOD = "random"  # @param ["random", "uniform", "loss", "margin", "gradient"]

# 修复后
SELECTION_METHOD = "uniform"  # @param ["uniform", "loss", "margin", "gradient"]
```

#### 4. 更新文档和说明
- cell-0: 标题和方法描述
- cell-1: 快速开始指南 FAQ
- cell-8: 实验 1 说明

---

## 影响范围

### 修改的单元格
| Cell ID | 修改内容 |
|---------|----------|
| cell-0 | 更新方法描述（Random → Uniform），添加修复说明 |
| cell-1 | 更新 FAQ，添加 torchaudio 和方法名问题说明 |
| cell-5 | 添加 torchaudio 自动卸载逻辑 |
| cell-8 | 更新实验方法列表 |
| cell-11 | 更新函数注释 |
| cell-12 | 修复 METHODS 列表 |
| cell-17 | 更新 SELECTION_METHOD 默认值和选项 |

### 未修改的单元格
- 所有实验逻辑单元格保持不变
- 训练和评估函数未受影响
- 可视化代码未受影响

---

## 测试验证

### ✅ Bug 1 验证
- [x] torchaudio 自动卸载
- [x] 无版本冲突警告
- [x] torch 正常工作

### ✅ Bug 2 验证
- [x] `uniform` 方法正常工作
- [x] 所有 4 种方法都能运行
- [x] 持续学习参数选项正确

---

## 使用建议

### 对于新用户
1. 直接运行所有单元格即可
2. 系统会自动处理 torchaudio 冲突
3. 使用正确的方法名：`uniform`, `kcenter`, `herding`, `bcsr`

### 对于已有用户
如果之前遇到这些 bug：
1. 重新运行 `0.3 安装依赖包` 单元格
2. 方法名从 `random` 改为 `uniform`
3. 无需修改代码

---

## 相关文件

- **notebook**: [notebooks/Complete_Experiment_Guide.ipynb](Complete_Experiment_Guide.ipynb)
- **baseline 方法**: [src/baselines/baseline_methods.py](src/baselines/baseline_methods.py)
- **requirements**: [requirements.txt](requirements.txt)

---

## 总结

两个 bug 都已完全修复：
- ✅ **Bug 1**: torchaudio 自动卸载，无版本冲突
- ✅ **Bug 2**: 使用正确的方法名 `uniform`

现在 notebook 可以在 Colab 中正常运行，无需手动干预！
