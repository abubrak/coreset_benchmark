# Coreset 选择方法详细说明

## 目录
1. [持续学习中的选择方法](#持续学习中的选择方法)
2. [双层优化方法](#双层优化方法)
3. [方法在项目中的位置](#方法在项目中的位置)

---

## 持续学习中的选择方法

在持续学习实验中，`CoresetBuffer` 使用以下方法从新任务中选择样本添加到缓冲区：

### 1. Uniform（均匀随机采样）

**文件位置**: [experiments/continual_learning.py:238-253](experiments/continual_learning.py#L238-L253)

**核心思想**: 按类别均匀随机选择样本

**算法流程**:
```python
def select_uniform(data, labels, num_samples):
    # 1. 计算每个类别应该选择的样本数
    num_classes = labels.max().item() + 1
    samples_per_class = num_samples // num_classes
    
    # 2. 对每个类别，随机选择 samples_per_class 个样本
    for c in range(num_classes):
        class_mask = (labels == c)  # 找到该类别的所有样本
        class_indices = torch.where(class_mask)[0]
        
        # 随机排列并选择前 n 个
        perm = torch.randperm(len(class_indices))[:samples_per_class]
        selected_indices.append(class_indices[perm])
    
    # 3. 合并所有类别的选择结果
    return torch.cat(selected_indices)
```

**特点**:
- ✅ 保证类别平衡
- ✅ 简单高效
- ❌ 不考虑样本难度或信息量

**适用场景**: 
- 基线对比
- 类别分布不均衡的情况

---

### 2. Loss（基于损失的选择）

**文件位置**: [experiments/continual_learning.py:263-269](experiments/continual_learning.py#L263-L269)

**核心思想**: 选择训练损失最高的样本（最难学习的样本）

**算法流程**:
```python
def select_by_loss(data, labels, model, num_samples):
    # 1. 计算所有样本的预测损失
    outputs = model(data)
    losses = nn.functional.cross_entropy(
        outputs, labels, reduction='none'  # 每个样本一个损失值
    )
    
    # 2. 选择损失最大的 top-k 样本
    _, indices = torch.topk(losses, num_samples)
    
    return indices
```

**特点**:
- ✅ 选择模型最不自信/最困惑的样本
- ✅ 主动学习策略
- ❌ 需要模型前向传播
- ❌ 可能选择噪声样本

**适用场景**:
- 主动学习
- 难样本挖掘
- 提升模型对困难样本的学习

**损失可视化**:
```
高损失 ← 选择的样本（困难样本）
    │
    ├── 样本A: loss=2.5 (模型不确定)
    ├── 样本B: loss=2.1 (模型困惑)
    └── 样本C: loss=1.9 (边界样本)
    │
低损失（容易样本，不选择）
```

---

### 3. Margin（基于边界余量的选择）

**文件位置**: [src/coreset/selection_functions.py:113-196](src/coreset/selection_functions.py#L113-L196)

**核心思想**: 选择分类边界余量最小的样本（最接近决策边界的样本）

**算法流程**:
```python
def select_by_margin(logits, labels, num_samples):
    # 1. 计算每个样本的分类余量
    #    Margin = 正确类别logit - 最大错误类别logit
    
    correct_logits = logits[torch.arange(n), labels]  # 正确类别的分数
    incorrect_logits = logits.masked_select(mask).view(n, -1)
    max_incorrect = incorrect_logits.max(dim=1)[0]  # 最大错误类别的分数
    
    margins = correct_logits - max_incorrect
    
    # 2. 余量越小 → 样本越接近决策边界 → 越不确定
    #    选择余量最小的样本（negate 使余量最小的得分最高）
    scores = -margins
    _, indices = torch.topk(scores, num_samples)
    
    return indices
```

**余量示意图**:
```
正确类别: logits[猫] = 5.0
错误类别: logits[狗] = 4.8  ← 最大错误类别
          logits[鸟] = 2.1
          logits[车] = 0.5

Margin = 5.0 - 4.8 = 0.2  ← 很小的余量，样本接近边界！

正确类别: logits[猫] = 8.0
错误类别: logits[狗] = 1.2  ← 最大错误类别
          logits[鸟] = 0.5
          logits[车] = 0.1

Margin = 8.0 - 1.2 = 6.8  ← 很大的余量，样本远离边界
```

**特点**:
- ✅ 考虑模型的不确定性
- ✅ 专注于边界样本
- ✅ 类别平衡选择
- ❌ 需要模型前向传播

**适用场景**:
- 边界样本挖掘
- 提升分类器的泛化能力
- 半监督学习

---

### 4. Gradient（基于梯度范数的选择）

**文件位置**: [src/coreset/selection_functions.py:199-296](src/coreset/selection_functions.py#L199-L296)

**核心思想**: 选择损失梯度范数最大的样本（对模型参数影响最大的样本）

**算法流程**:
```python
def select_by_gradient_norm(model, data, labels, num_samples):
    gradient_norms = []
    
    # 1. 对每个样本单独计算梯度范数
    for i in range(n_total):
        model.zero_grad()
        
        # 前向传播
        output = model(data[i:i+1])
        loss = nn.functional.cross_entropy(output, labels[i:i+1])
        
        # 反向传播
        loss.backward()
        
        # 计算所有参数梯度的L2范数
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2) ** 2
        grad_norm = grad_norm ** 0.5
        
        gradient_norms.append(grad_norm)
    
    # 2. 选择梯度范数最大的样本
    _, indices = torch.topk(gradient_norms, num_samples)
    
    return indices
```

**梯度范数示意图**:
```
样本A:
  ∂L/∂w1 = 0.5
  ∂L/∂w2 = 1.2
  ∂L/∂w3 = 0.8
  ──────────────
  ||∇L|| = 1.53  ← 大梯度，对参数影响大

样本B:
  ∂L/∂w1 = 0.1
  ∂L/∂w2 = 0.05
  ∂L/∂w3 = 0.2
  ──────────────
  ||∇L|| = 0.24  ← 小梯度，对参数影响小
```

**特点**:
- ✅ 选择对模型更新最重要的样本
- ✅ 考虑样本对整个模型的影响
- ❌ 计算成本高（需要逐样本反向传播）
- ❌ 需要模型完整的前向+反向传播

**适用场景**:
- 重要样本识别
- 数据集压缩
- 提升训练效率

---

## 方法对比总结

| 方法 | 选择依据 | 需要模型 | 计算成本 | 类别平衡 | 核心理念 |
|------|---------|---------|---------|---------|---------|
| **Uniform** | 随机均匀 | ❌ | ⚡ 低 | ✅ | 基线方法 |
| **Loss** | 训练损失 | ✅ (仅前向) | 🔸 中 | ✅ | 选择困难样本 |
| **Margin** | 分类余量 | ✅ (仅前向) | 🔸 中 | ✅ | 选择边界样本 |
| **Gradient** | 梯度范数 | ✅ (前向+反向) | 🐢 高 | ✅ | 选择重要样本 |

---

## 双层优化方法

项目中实现了两种基于双层优化的 Coreset 选择方法：

### 1. BilevelCoreset（经典双层优化）

**文件位置**: [src/coreset/bilevel_coreset.py](src/coreset/bilevel_coreset.py)

**核心思想**: 通过双层优化同时优化 Coreset 选择和模型参数

**优化问题**:
```
min_{S ⊆ D}  L_val(θ*(S))
  
s.t.  θ*(S) = argmin_θ L_train(S, θ)
```

其中：
- **外层优化**: 选择 Coreset S，最小化验证集损失
- **内层优化**: 给定 S，训练模型参数 θ

**算法框架**:
```python
def solve_bilevel_opt_representer_proxy(K_train, y_train, K_val, y_val):
    """
    使用 Representer Theorem 求解双层优化
    
    步骤:
    1. 初始化系数 alpha
    2. 外层循环（选择 Coreset）:
       for outer_iter in range(max_outer_it):
           a. 计算验证集损失
           b. 使用隐式微分计算梯度
           c. 更新 alpha（选择新的 Coreset）
    
    3. 返回最优的 alpha（对应选中的样本）
    """
    # 外层优化
    for outer_iter in range(max_outer_it):
        # 计算验证集损失
        val_logits = K_val @ alpha
        val_loss = F.cross_entropy(val_logits, y_val)
        
        # 隐式微分：计算 dL_val/dα
        grad_alpha = torch.autograd.grad(val_loss, alpha)[0]
        
        # 梯度下降更新
        alpha = alpha - lr * grad_alpha
    
    return alpha
```

**贪心选择策略**:
```python
def build_with_representer_proxy(X, y, m, kernel_fn):
    """
    贪心前向选择 Coreset
    
    for k in range(m):  # 逐步选择 m 个样本
        best_idx = None
        best_loss = float('inf')
        
        # 尝试每个剩余样本
        for idx in remaining_indices:
            # 将 idx 加入当前选择
            current_selected = selected_indices + [idx]
            
            # 求解双层优化，计算验证损失
            val_loss = solve_and_evaluate(current_selected)
            
            if val_loss < best_loss:
                best_idx = idx
        
        selected_indices.append(best_idx)
    
    return selected_indices
```

**特点**:
- ✅ 理论基础扎实（Representer Theorem）
- ✅ 显式建模验证集性能
- ❌ 计算复杂度高 O(n²m)
- ❌ 需要计算和存储核矩阵

**适用场景**:
- 小数据集
- 需要严格验证性能的场景

---

### 2. BCSRCoreset（BCSR: Bilevel Coreset Selection with Reweighting）

**文件位置**: [src/coreset/bcsr_coreset.py](src/coreset/bcsr_coreset.py)

**核心思想**: 通过双层优化学习样本权重，然后选择权重最高的样本

**与 BilevelCoreset 的区别**:
| 方面 | BilevelCoreset | BCSRCoreset |
|------|----------------|--------------|
| 优化目标 | 选择样本子集 | 学习样本权重 |
| 离散/连续 | 离散选择 | 连续权重 |
| 计算方式 | 贪心搜索 | 梯度下降 |
| 内存需求 | 需要核矩阵 | 可分批计算 |

**优化框架**:
```python
class BCSRCoreset:
    def coreset_select(X, y, coreset_size, model):
        """
        双层优化学习样本权重
        
        外层: 最小化验证集损失
        内层: 在加权训练集上更新模型参数
        
        for outer_step in range(num_outer_steps):
            # 内层优化：在加权数据上训练
            for inner_step in range(num_inner_steps):
                loss = Σ_i w_i * L(model(x_i), y_i)  # 加权损失
                θ = θ - lr_inner * ∇_θ loss
            
            # 外层优化：更新样本权重
            val_loss = L_val(θ)
            w = w - lr_outer * ∇_w val_loss
            
            # 投影到单纯形（w ≥ 0, Σw = 1）
            w = projection_onto_simplex(w)
        
        # 选择权重最高的 top-k 样本
        top_k_indices = argsort(w)[-coreset_size:]
        return X[top_k_indices], y[top_k_indices]
    """
```

**权重投影（单纯形投影）**:
```python
def projection_onto_simplex(v):
    """
    将向量投影到单纯形: w ≥ 0, Σw = 1
    
    算法 (Duchi et al., 2008):
    1. 找到阈值 θ，使得 Σ max(v_i - θ, 0) = 1
    2. 投影: w_i = max(v_i - θ, 0)
    """
    # 排序
    u = np.sort(v)[::-1]
    
    # 找到满足条件的阈值
    rho = max{j : u_j - (Σ_{k≤j} u_k - 1)/j > 0}
    theta = (Σ_{k≤rho} u_k - 1) / rho
    
    # 投影
    w = np.maximum(v - theta, 0)
    return w / w.sum()  # 归一化
```

**核方法版本（无模型）**:
```python
def _optimize_weights_kernel(X, y):
    """
    简化版本：基于多样性和类别平衡学习权重
    
    # 1. 计算多样性得分（RBF 核均值）
    K = exp(-γ * ||x_i - x_j||²)
    diversity_scores = K.mean(dim=1)  # 每个样本的平均相似度
    
    # 2. 计算类别平衡权重
    class_weights = 1.0 / class_counts
    sample_class_weights = class_weights[labels]
    
    # 3. 组合权重
    weights = sample_class_weights * diversity_scores
    
    # 4. 投影到单纯形
    weights = projection_onto_simplex(weights)
    
    return weights
```

**特点**:
- ✅ 内存高效（可分批计算）
- ✅ 连续优化（可微分）
- ✅ 支持基于模型的方法
- ✅ 同时考虑多样性和重要性
- ❌ 需要调优超参数

**适用场景**:
- 大规模数据集
- 需要内存高效的方法
- 希望结合模型信息

---

## 方法在项目中的位置

### 实验 1: 数据摘要

**Notebook 位置**: cell-12 `1.4 运行所有 Coreset 方法对比`

**方法列表**:
```python
METHODS = ['uniform', 'kcenter', 'herding', 'bcsr']
```

**对应实现**:
| Notebook 方法名 | 实际调用 | 文件位置 |
|---------------|---------|---------|
| `'uniform'` | `UniformSelector` | [src/baselines/baseline_methods.py:48](src/baselines/baseline_methods.py#L48) |
| `'kcenter'` | `KCenterSelector` | [src/baselines/baseline_methods.py:80](src/baselines/baseline_methods.py#L80) |
| `'herding'` | `HerdingSelector` | [src/baselines/baseline_methods.py:201](src/baselines/baseline_methods.py#L201) |
| `'bcsr'` | `BCSRCoreset` | [src/coreset/bcsr_coreset.py:13](src/coreset/bcsr_coreset.py#L13) |

### 实验 2: 持续学习

**Notebook 位置**: cell-17 `2.1 持续学习实验参数`

**方法列表**:
```python
SELECTION_METHOD = "uniform"  # @param ["uniform", "loss", "margin", "gradient"]
```

**对应实现**:
| Notebook 方法名 | 实际调用 | 文件位置 |
|---------------|---------|---------|
| `'uniform'` | `select_coreset(..., method="uniform")` | [experiments/continual_learning.py:238](experiments/continual_learning.py#L238) |
| `'loss'` | 内联实现（计算损失） | [experiments/continual_learning.py:263](experiments/continual_learning.py#L263) |
| `'margin'` | `select_by_margin()` | [src/coreset/selection_functions.py:113](src/coreset/selection_functions.py#L113) |
| `'gradient'` | `select_by_gradient_norm()` | [src/coreset/selection_functions.py:199](src/coreset/selection_functions.py#L199) |

---

## 双层优化在哪里使用？

### 在项目中

1. **实验 1**: `BCSRCoreset` 使用双层优化学习样本权重
   - 文件: [src/coreset/bcsr_coreset.py](src/coreset/bcsr_coreset.py)
   - 优化目标: 学习样本权重 w，使得验证集损失最小

2. **理论研究**: `BilevelCoreset` 实现了经典的双层优化方法
   - 文件: [src/coreset/bilevel_coreset.py](src/coreset/bilevel_coreset.py)
   - 优化目标: 直接选择样本子集 S

### 核心差异

```
┌─────────────────────────────────────────────────────────────┐
│                    双层优化框架                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  min_S⊆D  L_val(θ*(S))                                    │
│                                                             │
│  s.t.    θ*(S) = argmin_θ L_train(S, θ)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
┌───────────────┐          ┌──────────────────┐
│ BilevelCoreset│          │  BCSRCoreset    │
├───────────────┤          ├──────────────────┤
│ 贪心选择样本  │          │ 优化连续权重    │
│ 离散优化     │          │ 梯度下降        │
│ O(n²m) 时间  │          │ O(nm) 时间      │
│ 需要核矩阵   │          │ 内存高效        │
└───────────────┘          └──────────────────┘
```

---

## 总结

### 持续学习方法特点

- **Uniform**: 简单基线，类别平衡
- **Loss**: 选择困难样本，主动学习
- **Margin**: 选择边界样本，提升泛化
- **Gradient**: 选择重要样本，考虑参数影响

### 双层优化方法特点

- **BilevelCoreset**: 理论严格，适合小数据
- **BCSRCoreset**: 实用高效，适合大数据

两者都通过双层优化框架（外层选样本/权重，内层训练模型）来选择最有效的 Coreset！

---

**建议**: 
- 快速实验 → 使用 `Uniform`
- 性能优先 → 尝试 `BCSR` (已在实验 1 中)
- 持续学习 → 对比 4 种方法（已在实验 2 中）
