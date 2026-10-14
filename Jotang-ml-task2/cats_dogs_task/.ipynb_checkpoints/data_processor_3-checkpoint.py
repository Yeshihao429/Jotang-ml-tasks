from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch


def _build_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, eval_transform

# 方便从train集中划分两个数据集，因为原数据集中的test1的标签是无效的
def split_train_val(train_root, val_ratio=0.2, seed=42, batch_size=32):
    if not os.path.exists(train_root):
        raise FileNotFoundError(f"没有 {train_root}")

    train_tf, eval_tf = _build_transforms()

    # 同一份数据加载两次，用不同 transform
    full_train = datasets.ImageFolder(train_root, transform=train_tf)
    full_eval  = datasets.ImageFolder(train_root, transform=eval_tf)

    n = len(full_train)
    n_val   = int(val_ratio * n)
    n_train = n - n_val

    g = torch.Generator().manual_seed(seed)
    train_sub, _ = random_split(full_train, [n_train, n_val], generator=g)
    _, val_sub   = random_split(full_eval,  [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,  num_workers=2
    )
    val_loader = DataLoader(
        val_sub,   batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"[split] train: {n_train}, val: {n_val}, "
          f"val_ratio={val_ratio}, seed={seed}")
    return train_loader, val_loader


def process_image(root, val_ratio=0.2, seed=42, batch_size=32):
    train_root = os.path.join(root, "train")
    return split_train_val(train_root, val_ratio, seed, batch_size)

# 检验函数
def random_check(train_loader, val_loader, n=5, seed=42):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def denormalize(tensor):
        img = tensor.permute(1, 2, 0).numpy()
        img = img * std + mean
        return np.clip(img, 0, 1)

    random.seed(seed)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for row, (loader, split_name) in enumerate([(train_loader, "Train"),
                                                (val_loader,   "Val")]):
        dataset = loader.dataset
        base = dataset.dataset if hasattr(dataset, "dataset") else dataset
        idx_to_class = {v: k for k, v in base.class_to_idx.items()}

        indices = random.sample(range(len(dataset)), n)

        for col, idx in enumerate(indices):
            img_tensor, label = dataset[idx]
            img = denormalize(img_tensor)

            ax = axes[row, col] if n > 1 else axes[row]
            ax.imshow(img)
            ax.set_title(idx_to_class[label], fontsize=10)
            ax.axis("off")

        axes[row, 0].set_ylabel(split_name, fontsize=14)

    plt.suptitle("Random Check: Train (top) vs Val (bottom)", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root = "./cats_and_dogs"
    train_loader, val_loader = process_image(root)
    random_check(train_loader, val_loader, n=5)