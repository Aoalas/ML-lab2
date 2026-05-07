# 机器学习实验二报告（二元分类）

## 基本信息

- 学号：`（填写你的学号）`
- 系级：`（填写你的系级）`
- 姓名：`（填写你的姓名）`

---

## 1. 在 `ex2_logistic_original.ipynb` 中的参数调试与分析（占分 25%）

在训练 logistic 模型时，本实验重点调试了以下超参数：

- `max_iter = 10`（可上下调）
- `batch_size = 8`（可上下调）
- `learning_rate = 0.2`（可上下调）

### 1.1 配置性能特征分析

1. 训练速度优先配置
   - 训练时间：约 6~8 秒（约 4.8 万训练样本）
   - 资源占用：较低（小批量训练）
   - 计算环境：CPU 可运行

2. 预期模型表现（示例，按你的实测结果填写）

| 指标 | 参考值区间 | 说明 |
|---|---|---|
| 训练精度 | 0.85 ~ 0.89 | 训练过程较稳定 |
| 测试精度 | 0.84 ~ 0.86 | 具有一定泛化能力 |
| 训练损失 | 约 0.27 ~ 0.31 | 随迭代下降 |
| 收敛情况 | 前 10 轮变化明显 | 后续提升趋缓 |

3. 参数调试结论（可按实测修改）
   - `learning_rate` 过小：更稳定但收敛慢。
   - `learning_rate` 过大：收敛快但可能振荡。
   - `batch_size` 减小：更新更频繁，耗时上升，波动变大。
   - `max_iter` 增大：训练时间近似线性增加，后期收益递减。

### 1.2 配置优势与局限

优势：

- 训练速度较快
- 小批量训练具备一定隐式正则化效果
- 对资源要求较低

局限：

- 参数设置不当时容易欠收敛
- 测试精度仍有提升空间
- 学习曲线后期趋稳，提升有限

### 1.3 数据结果与 PR 曲线图

训练日志截图（训练集大小、测试集大小、最终 loss、最终 accuracy）：

![original_train_log](./original_train_log.png)

Loss 与 Accuracy 曲线：

![loss_acc](./loss_acc.png)

Logistic PR 曲线：

![logistic_pr](./logistic_pr.png)

---

## 2. 在 `ex2_logistic_xgboost.ipynb` 中的四模型对比与问答（占分 75%）

### 2.1 四种模型 PR 曲线图

本实验对比模型：

- Logistic Regression
- Decision Tree
- Random Forest
- （若最终代码含 XGBoost，可补充 XGBoost）

PR 对比图如下：

![pr_comparison](./pr_comparison.png)

---

### 2.2 问题（1）：采用了哪种特征工程？原理、实现过程、影响

本实验采用的特征工程流程：

1. 方差筛选（VarianceThreshold）
   - 原理：删除方差过小特征，减少低信息量噪声。
   - 实现：`VarianceThreshold(threshold=0.1)`。

2. 标准化（StandardScaler）
   - 原理：统一量纲，避免少数大数值特征主导模型。
   - 实现：`fit_transform / transform`，并配合 `np.clip(-10, 10)` 抑制极端值。

3. PCA 降维
   - 原理：在尽量保留主要信息的前提下压缩维度，降低冗余与过拟合风险。
   - 实现：高维时保留 95% 方差，低维时最多保留 30 维。

量化影响（模板，按实测填写）：

| 处理阶段 | 维度变化 | 训练时间 | 训练精度 | 测试精度 |
|---|---:|---:|---:|---:|
| 原始数据 | 510 |  |  |  |
| 方差筛选后 |  |  |  |  |
| 标准化后 |  |  |  |  |
| PCA 后 |  |  |  |  |

与超参数协同（可选）：

- 与 `max_iter`：降维后通常更快收敛。
- 与 `batch_size`：低维输入下小批量更新更稳定。
- 与 `learning_rate`：标准化后可使用更稳定的学习率区间。

---

### 2.3 问题（2）：正则化的作用，以及对模型影响

正则化通过约束参数规模来抑制过拟合，提升泛化能力。

- L1：产生稀疏解，具备特征选择效果。
- L2：参数更平滑，训练更稳定。

对本实验模型的影响：

- 训练集精度可能略降
- 测试集稳定性通常提升
- PR 曲线在测试集上更可靠

---

### 2.4 问题（3）：XGBoost 相对 Logistic 的优势

主要优势：

- 能捕捉非线性关系与特征交互
- Boosting 框架通常有更高性能上限
- 自带正则化与采样策略，泛化能力较强
- 对复杂特征分布更鲁棒

代价：

- 参数更多，调参复杂
- 训练时间通常更长
- 可解释性通常弱于 Logistic

---

## 3. 图片清单：需要截哪些图 + 怎么得到

1. `original_train_log.png`
   - 内容：`ex2_logistic_original.ipynb` 训练日志（训练集大小、测试集大小、loss、accuracy）。
   - 获取：运行训练单元后，截图输出区域。

2. `loss_acc.png`
   - 内容：Loss 曲线 + Accuracy 曲线。
   - 获取：运行对应绘图单元；也可加保存语句：
```python
plt.savefig('loss_acc.png', dpi=150, bbox_inches='tight')
```

3. `logistic_pr.png`
   - 内容：original notebook 中的 logistic PR 曲线。
   - 获取：
```python
plt.savefig('logistic_pr.png', dpi=150, bbox_inches='tight')
```

4. `pr_comparison.png`
   - 内容：`ex2_logistic_xgboost.ipynb` 的多模型 PR 对比图。
   - 获取：运行完整流程，代码中通常已自动保存。

5. （可选）`feature_steps.png`
   - 内容：方差筛选、标准化、PCA 三段核心代码截图。
   - 获取：截图对应代码单元，拼接或分图插入。

---

## 4. 复现与提交建议

1. 先运行 `ex2_logistic_original.ipynb`，保存训练日志图、loss/acc 图、logistic PR 图。
2. 再运行 `ex2_logistic_xgboost.ipynb`，保存 `pr_comparison.png`。
3. 用你的实测数据补全本报告中的表格和结论。
4. 检查图片路径无误后，导出为 `report.pdf` 并提交。
