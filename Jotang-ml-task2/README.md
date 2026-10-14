# Task2 — 卷积与猫狗分类实践

> 本项目是 ML 学习路线的第二阶段作业：**从手写卷积函数起步，到仿 AlexNet 搭建一个猫狗二分类网络并完成训练、可视化与推理**。
> 所有实验过程的思考与结论记录在 [note3.md](./note3.md) 中，本文件只负责说明「项目怎么跑、每个文件干什么」。

---

## 一、项目简介

整个 task2 分成两个实践：

| 实践 | 目标 | 产出 |
| --- | --- | --- |
| 实践1 · 卷积核 | 理解像素/RGB/灰度、Numpy 与 Tensor 的维度差异、卷积到底在算什么 | 手写的 `MyConv2d`，以及均值/高斯/锐化/Sobel 四种卷积核的对照图 |
| 实践2 · 猫狗分类器 | 端到端跑通一个图像分类任务：下载数据 → 划分数据集 → 搭网络 → 训练 → 分析错误 → 做推理 | 仿 AlexNet 的 `CatsDogsModel`、训练曲线、`best_model.pth`、推理脚本 |

模型最终形态是一个 **5 层卷积 + 3 层全连接** 的网络（用 `BatchNorm2d` 替代了原版 AlexNet 的 LRN），输入 `224×224×3`，输出猫/狗两类 logits。数据集为 Kaggle `dogs-vs-cats`，训练集 25000 张（猫狗各 12500），按 8:2 划分为 train / val。

---

## 二、环境依赖

- Python 3.x
- PyTorch、torchvision
- numpy、matplotlib、Pillow
- PyYAML（配置文件解析）
- kagglehub（**仅** `download_data_1.py` 下载数据时需要，且要求本机 `~/.kaggle/` 已放好 API token）
- Jupyter / Notebook（运行 `.ipynb`）

> `matplotlib.use("TkAgg")` 在 `picture_2.py` 中被显式指定，因此在无图形界面的环境里直接运行该脚本可能报错，建议在桌面环境或注释掉该行运行。

---

## 三、快速开始（推荐执行顺序）

所有命令均在 **`cats_dogs_task/` 目录下**执行（脚本统一按相对路径读取同目录的 `config.yaml`）：

```bash
cd cats_dogs_task

python download_data_1.py     # ① 下载、解压、清理、按文件名分好 cat/ 和 dog/
python picture_2.py           # ② 统计数量 + 随机抽样可视化（可选）
python data_processor_3.py    # ③ 自检：查看 train/val 的数据增强效果是否正确
python net_4.py               # ④ 自检：打印每张特征图的尺寸变化

# ⑤ 训练：直接用 VSCode/Jupyter 打开 train.ipynb 从头跑到尾
# ⑥ 推理：改好 config.yaml 里的 data_root 与 n_show 后
python predictor_8.py
```

---

## 四、项目结构总览

```
Jotang-ml-task2/
├── README.md                     ← 本文件，项目说明
├── note3.md                      ← 作业报告：所有思考题的答案与实验结论
│
├── 卷积核-实践1.ipynb             ← 实践1：像素/RGB/灰度 → Numpy → Tensor
├── 手搓卷积函数.ipynb             ← 实践1：手写卷积 + 四类卷积核对照
│
├── attachment/                   ← 报告与实验用到的所有图片素材
│   ├── outer1.png                    实践1的测试原图（手搓卷积用的就是它）
│   ├── MyModel.png / MyModel.drawio.svg   本项目网络结构图
│   ├── 预测错误.png                 16 张预测错样本拼图
│   ├── 分布起伏.png                 VGG16 浅层/深层 L2 范数分布对比
│   ├── AlexNet.png / AlexNet_2.png / AlexNet3.png
│   ├── VGGNet.png / VGGNet_division.png / VGGNet_structure.png
│   └── ResNet.png
│
└── cats_dogs_task/               ← 实践2：猫狗分类的主工程目录
    ├── config.yaml               ★ 全局路径与推理参数配置（唯一需要改的地方）
    │
    ├── download_data_1.py        ① 下载并整理原始数据
    ├── picture_2.py              ② 数据量统计与抽样展示
    ├── data_processor_3.py       ③ 数据增强 + train/val 划分 + DataLoader
    ├── net_4.py                  ④ 网络定义 + 逐层形状自检
    ├── predictor_8.py            ⑧ 加载训练好的权重做单张/批量推理
    │
    ├── train.ipynb               ⑤ 主训练脚本（20 epoch + 保存最优 + 错例可视化）
    ├── Linear.ipynb              对照实验：只用全连接层会怎样
    ├── 可视化特征提取_5.ipynb      hook 抓取 conv1 / conv_last 的特征图
    ├── 数据增强对照试验_6.ipynb     翻转 / 裁剪 / 颜色 / 组合增强的效果对照
    ├── VGGNet.ipynb              用预训练 VGG16 对比浅层与深层特征分布
    ├── 接入Bottleneck.ipynb      把第 4 段卷积替换成 ResNet Bottleneck 的替换实验
    │
    ├── cats_and_dogs/            数据集（体积大，不在版本库中）
    │   ├── train/cat/                12500 张
    │   ├── train/dog/                12500 张
    │   └── test1/                    12500 张（无标签，仅用于推理演示）
    ├── saved_model/
    │   └── best_model.pth        训练过程中 val acc 最优的一份权重（约 222 MB）
    ├── photo_of_result.jpg       train.ipynb 输出的 loss / accuracy 曲线
    ├── 只用全连接层对比图.jpg      Linear.ipynb 输出的对比曲线
    ├── __pycache__/              解释器自动生成，可忽略
    └── .ipynb_checkpoints/       Jupyter 自动备份，可忽略
```

> 文件名里的数字（`_1`、`_2`、`_3`…）代表它们在流水线中的**执行顺序**，方便按顺序阅读。

---

## 五、文件详解

### 5.1 根目录：实践1（理解卷积）

**`卷积核-实践1.ipynb`**
打通「图片 → Numpy 数组 → Tensor」这条链路。读取 `attachment/outer1.png` 后，分别打印 Numpy 的 `H×W×C` 形状和 Tensor 的 `C×H×W` 形状，并对比同一个像素点（第 100 行、第 150 列）在两种表示下的 RGB 值，直观说明为什么要 `transpose(2, 0, 1)`，以及为什么 `ToTensor()` 内部已经帮我们做了这件事。

**`手搓卷积函数.ipynb`**
不调用框架，纯 Numpy 实现卷积：

- `MyPad()` — 零填充；
- `MyConv2d()` — 滑窗取块、逐元素相乘再求和，输出尺寸按 `(H + 2p - k) // stride + 1` 计算；
- 随后定义并对照四种典型卷积核：**均值模糊**、**高斯模糊**、**锐化**、**Sobel 边缘检测**（gx / gy 分别卷积后再求模），最后拼成一张对比图。
- 边缘检测部分额外实现了 `to_uint8()`：卷积结果先裁剪或归一化到 `[0, 255]` 再转 `uint8`，避免溢出导致的黑白反转。

### 5.2 cats_dogs_task：实践2（训练分类器）

| 文件 | 作用 | 关键实现 |
| --- | --- | --- |
| `download_data_1.py` | 一键把 Kaggle 数据整理成可直接喂给 `ImageFolder` 的形态 | 目录非空则跳过下载；解压时自动拉平多余的一层目录；删掉 zip、`.complete`、`sampleSubmission.csv`；最后按文件名前缀把图片 `move` 进 `train/cat`、`train/dog`、`test1/cat`、`test1/dog` |
| `picture_2.py` | 数据集体检 | `os.walk` 遍历并按文件名前缀计数，打印猫/狗数量，再随机各抽 4 张画图，观察尺寸、清晰度、背景差异 |
| `data_processor_3.py` | 数据管线核心 | `_build_transforms()` 定义训练/验证两套变换；`split_train_val()` 从 train 里划分 8:2；`process_image()` 是对外入口；`random_check()` 反归一化后抽样作图，用来肉眼验证增强有没有搞坏图片 |
| `net_4.py` | 网络定义 | `CatsDogsModel`：`features`（5 卷积 + 3 池化，每层卷积后接 BN + ReLU）+ `classifier`（Dropout → 4096 → 4096 → 2）。`__main__` 里会逐层打印张量形状，是最直观的 shape 自检工具 |
| `predictor_8.py` | 推理入口 | 从 config 里读 `model_path` / `data_root` / `n_show` / `class_names`，`data_root` 可以是**单张图片路径**也可以是**文件夹**；用 softmax 输出各类别概率并画图 |
| `train.ipynb` | 主训练流程 | 20 个 epoch，Adam + CrossEntropyLoss；`run_epoch()` 统一处理训练/验证；每轮保存 val acc 最优的权重到 `saved_model/best_model.pth`；训练结束后调用 `collect_wrong()` / `show_misclassified()` 把预测错的样本（上限 16 张）画出来；最后 `draw_result()` 输出 loss/acc 曲线并自动避免覆盖旧的 `photo_of_result.jpg` |
| `Linear.ipynb` | 对照实验：删掉所有卷积，只留 `Flatten + Linear(3×224×224 → 2)` | 结论：效果几乎等于瞎猜，直观说明卷积层提取局部相关性、参数共享的必要性 |
| `可视化特征提取_5.ipynb` | 用 `forward hook` 抓取首层卷积（`CONV1_IDX=0`）和末层卷积（`CONV_LAST_IDX=14`）的输出 | 观察结论：第一层还能看出轮廓，最后一层已抽象成色块；不同通道关注不同特征 |
| `数据增强对照试验_6.ipynb` | 同一张样本分别施加翻转、裁剪、颜色抖动以及三者组合，横向对比 | 用来判断哪些增强会破坏标签（如垂直翻转、大角度旋转、裁剪范围过小） |
| `VGGNet.ipynb` | 加载 ImageNet 预训练 VGG16，取 `features[0]` 与 `features[0:29]` 的输出 | 计算每个通道的 L2 范数并归一化作图：浅层起伏剧烈、深层相对平坦 |
| `接入Bottleneck.ipynb` | 定义 ResNet 的 `Bottleneck`（1×1 → 3×3 → 1×1，expansion=4），替换原网络第 4 段卷积后重新训练 | 结论：卷积层参数下降，但训练速度没有明显变化 |

### 5.3 数据与产物

- **`cats_and_dogs/`**：Kaggle `dogs-vs-cats` 的原始内容。原始 `test1` 是比赛用的无标签集（文件名只有序号），脚本按文件名前缀分类时它实际上没有 cat/dog 前缀，因此训练与验证**全部来自 `train/` 的 25000 张**（猫狗各 12500），按 8:2 切成 20000 / 5000。`test1` 只用来演示推理。
- **`saved_model/best_model.pth`**：`train.ipynb` 训练过程中 val accuracy 最高时的 `state_dict`，由 `predictor_8.py` 加载。
- **`attachment/`**：纯图片资源，既服务于 `note3.md` 的排版，也是 `卷积核-实践1.ipynb` / `手搓卷积函数.ipynb` 的输入。

---

## 六、config.yaml 配置说明

路径不再硬编码在各脚本里，统一由 `cats_dogs_task/config.yaml` 管理，修改这一处即可：

```yaml
model_path: ./saved_model/best_model.pth   # 推理时加载的权重
data_root: ./cats_and_dogs/test1           # 推理数据来源：单张图片路径 或 图片文件夹
dataset_root: ./cats_and_dogs              # 数据集根目录（下载目标、train/ 的上一级）
train_root: ./cats_and_dogs/train          # 已按 cat/ dog/ 分好类的训练集
n_show: 1                                  # 推理时展示几张（>1 时从文件夹中随机抽）
class_names:                               # 下标 → 类别名，顺序须与训练时一致
  - cat
  - dog
```

各文件的读取关系：

| 配置项 | 被谁使用 |
| --- | --- |
| `model_path` | `predictor_8.py` |
| `data_root` | `predictor_8.py`、`可视化特征提取_5.ipynb` |
| `dataset_root` | `download_data_1.py`、`data_processor_3.py`、`net_4.py`、`train.ipynb`、`Linear.ipynb`、`数据增强对照试验_6.ipynb` |
| `train_root` | `picture_2.py`、`VGGNet.ipynb` |
| `n_show` / `class_names` | `predictor_8.py` |

---

## 七、数据处理流程

```
kagglehub 下载 dogs-vs-cats
        │  download_data_1.py：解压 → 拉平 → 删 zip/垃圾文件 → 按 cat./dog. 前缀归类
        ▼
cats_and_dogs/train/{cat,dog}   （25000 张）
        │  data_processor_3.py：ImageFolder 读取
        ├── 同一份数据加载两次，分别套 train / eval 两种 transform
        ├── random_split(seed=42)：20000 train / 5000 val
        └── DataLoader(batch_size=32, shuffle, num_workers=2)
                ▼
            net_4.CatsDogsModel → 训练 → best_model.pth → predictor_8.py 推理
```

**数据增强策略**（训练集专用）：

- 训练：`RandomResizedCrop(224, scale=(0.8, 1.0))` + `RandomHorizontalFlip()` + `ToTensor()` + `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
- 验证/推理：`Resize(256)` + `CenterCrop(224)` + `ToTensor()` + 同样的 `Normalize`

> 验证集与测试集只做确定性变换：一是保证结果可复现，二是测试时的目标就是看模型在「原样图片」上的真实水平。

---

## 八、网络结构与张量形状

输入 `(N=1, C=3, H=224, W=224)`，输出尺寸公式 `(H + 2p - k) // stride + 1`：

| 层 | 配置 | 输出形状 |
| --- | --- | --- |
| Conv1 (+BN+ReLU) | `Conv2d(3→96, k=11, s=4, p=2)` | `(1, 96, 55, 55)` |
| MaxPool1 | `k=3, s=2` | `(1, 96, 27, 27)` |
| Conv2 (+BN+ReLU) | `Conv2d(96→256, k=5, s=1, p=2)` | `(1, 256, 27, 27)` |
| MaxPool2 | `k=3, s=2` | `(1, 256, 13, 13)` |
| Conv3 (+BN+ReLU) | `Conv2d(256→384, k=3, s=1, p=1)` | `(1, 384, 13, 13)` |
| Conv4 (+BN+ReLU) | `Conv2d(384→384, k=3, s=1, p=1)` | `(1, 384, 13, 13)` |
| Conv5 (+BN+ReLU) | `Conv2d(384→256, k=3, s=1, p=1)` | `(1, 256, 13, 13)` |
| MaxPool3 | `k=3, s=2` | `(1, 256, 6, 6)` |
| Flatten | — | `(1, 9216)` ← `256×6×6` |
| FC1 / FC2 / FC3 | `9216→4096→4096→2`，每层前接 `Dropout(p=0.5)` | `(1, 2)` |

可以直接运行 `python net_4.py` 复现上面这张表。结构图见 `attachment/MyModel.png`。

主要超参：`epochs=20`、`lr=1e-4`、`weight_decay=1e-4`、优化器 Adam、损失 CrossEntropyLoss。

---

## 九、实验结论速查

| 想看的内容 | 去哪里 |
| --- | --- |
| 八个思考题的完整回答（卷积怎么滑、输出尺寸公式、溢出怎么处理等） | [note3.md](./note3.md) 实践1 / 趁热打铁部分 |
| 网络每一层为什么这么设计、通道数为什么越往后越多 | [note3.md](./note3.md) 趁热打铁部分 |
| 纯全连接层 vs 卷积层 | `Linear.ipynb` + `只用全连接层对比图.jpg` |
| 特征图逐层抽象过程 | `可视化特征提取_5.ipynb` |
| 哪些数据增强有害 | `数据增强对照试验_6.ipynb` |
| 16 张预测错样本的共性分析 | `train.ipynb`（`show_misclassified`）+ `attachment/预测错误.png` |
| AlexNet / VGGNet / ResNet 的笔记与参数量公式 | [note3.md](./note3.md) 补充部分 |

预测出错的典型原因（肉眼归因）：一图多只宠物、宠物与环境融为一体、只露出局部（半张脸/只有耳朵）、姿势怪异、光线过暗、主人脸部占比过大等。

---

## 十、注意事项

1. **工作目录**：所有脚本都用相对路径 `"config.yaml"` 读取配置，必须在 `cats_dogs_task/` 下运行，换目录会找不到配置文件。
2. **下载前置条件**：`download_data_1.py` 依赖 kagglehub，需要本机已配置 Kaggle API；数据集本身较大（约 800 MB）不入版本库，`data_root` 等路径都指向本地相对目录。
3. **类别顺序**：`class_names` 的顺序必须与 `ImageFolder` 建出来的 `class_to_idx` 一致（默认按文件夹名字典序：`cat=0, dog=1`），改动会导致预测结果标签反转。
4. **不计入流水线的目录**：`__pycache__/`、`.ipynb_checkpoints/` 均为自动生成，勿提交、勿手改。
