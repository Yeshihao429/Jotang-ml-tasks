import os
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

img_dir = cfg["train_root"]

cats = []
dogs = []

for root, _, files in os.walk(img_dir):
    for f in files:
        name = f.lower()
        if not name.endswith((".jpg", ".jpeg", ".png")):
            continue
        full_path = os.path.join(root, f)
        if name.startswith("cat"):
            cats.append(full_path)
        elif name.startswith("dog"):
            dogs.append(full_path)


print("=" * 30)
print(f"猫的图片数量: {len(cats)}")
print(f"狗的图片数量: {len(dogs)}")
print(f"总计: {len(cats) + len(dogs)}")
print("=" * 30)

# 随机抽样，每类四个
n_per_class = 4
sample_cats = random.sample(cats, min(n_per_class, len(cats)))
sample_dogs = random.sample(dogs, min(n_per_class, len(dogs)))


fig, axes = plt.subplots(2, n_per_class, figsize=(12, 6))

# 第一行：猫
for ax, img_path in zip(axes[0], sample_cats):
    ax.imshow(Image.open(img_path))
    ax.set_title(os.path.basename(img_path), fontsize=8)
    ax.axis("off")

# 第二行：狗
for ax, img_path in zip(axes[1], sample_dogs):
    ax.imshow(Image.open(img_path))
    ax.set_title(os.path.basename(img_path), fontsize=8)
    ax.axis("off")

plt.suptitle("Cats (top) vs Dogs (bottom)", fontsize=14)
plt.tight_layout()
plt.show()