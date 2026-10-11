# task1 - make_moons 训练 MLP

用 PyTorch 搭建一个简单的 MLP（多层感知机），在 `make_moons` 二分类数据集上完成训练、评估与可视化，并对比不同噪声、网络深度下的实验效果。详细分析见 [note2.md](./note2.md)。

## 项目结构

```
task1/
├── make_moons训练模型.ipynb   # 主程序：数据加载 → 建模型 → 训练 → 画图
├── note2.md                   # 实验笔记（激活/损失函数、过拟合、控制变量实验等）
├── README.md
└── attachment/                # 笔记中引用的图片
    ├── cat.png
    └── confusion_matrix.png
```

## 环境依赖

- Python 3.10
- torch、scikit-learn、numpy、matplotlib
- jupyter lab（开发环境）

```bash
pip install torch scikit-learn numpy matplotlib jupyterlab
```

## 运行方式

```bash
cd task1
jupyter lab
```

然后打开 `make_moons训练模型.ipynb`，从上到下依次运行所有单元格即可。在 Jupyter 中可按 `Restart Kernel and Run All` 一键跑完整流程。

GPU 不是必需的：代码会自动检测 `torch.cuda.is_available()`，没有 CUDA 时回落到 CPU（只是慢一些）。

## 内容概览

1. **数据**：`make_moons(n_samples=10000, noise=0.1)`，按 8:1:1 划分训练/验证/测试集，标准化后转张量装入 `DataLoader`。
2. **模型**：两层隐藏层的 MLP（2 → 64 → 64 → 2），损失函数 `CrossEntropyLoss`，优化器 `Adam(lr=1e-3)`。
3. **训练**：20 个 epoch，每个 epoch 后在测试集上评估，并按验证集精度选最佳模型。
4. **可视化**：原始数据散点图、数据集分布、loss/acc 曲线、决策边界、混淆矩阵。

## 笔记要点

- 激活函数选 **ReLU**，损失函数选 **CrossEntropyLoss**（内部含 LogSoftmax，数值更稳定）。
- 三个关键步骤：`zero_grad()` 清空旧梯度 → `backward()` 反向传播算梯度 → `step()` 更新参数。
- 过拟合的典型表现：loss 降到低点后又回升，且 train / test 曲线之间出现明显间隙。
- 控制变量实验：`noise` 由 0.01 → 0.1 → 0.3 时难度递增（0.3 时 test 与 train 精度约差 0.1）；在 noise=0.1 下加深网络（+2 层）后，test 曲线收敛后出现波动，呈现轻微过拟合。
