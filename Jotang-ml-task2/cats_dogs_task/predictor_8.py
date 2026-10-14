from net_4 import CatsDogsModel
import torch
import yaml
from data_processor_3 import _build_transforms
import os
import random
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 从配置文件中调用路径
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    MODEL_PATH = cfg["model_path"]
    DATA_ROOT = cfg["data_root"]
    class_names = cfg["class_names"]
    n_show = cfg["n_show"]

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CatsDogsModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    # 预处理
    _, val_transforms = _build_transforms()

    # 加载数据
    exts = ('.jpg', '.jpeg', '.png')
    if os.path.isfile(DATA_ROOT):
        images = [DATA_ROOT]
    else:
        images = []
        for f in os.listdir(DATA_ROOT):
            if f.lower().endswith(exts):
                images.append(os.path.join(DATA_ROOT, f))

    random.seed(1)

    if n_show < len(images):
        images = random.sample(images, n_show)

    # 分别预测
    for i, path in enumerate(images):
        img = Image.open(path).convert('RGB')
        x = val_transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = F.softmax(model(x), dim=1)[0].cpu()

            idx = int(prob.argmax())
            name = class_names[idx]
            conf = float(prob[idx])

            print(f"\n({i + 1}/{len(images)}) {path}")
            print(f"  预测: {name}  ({conf:.4f})  "
                  + "  ".join(f"{class_names[j]}={prob[j]:.4f}" for j in range(len(class_names))))

            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.title(f"{os.path.basename(path)}\n {name} ({conf * 100:.2f}%)")
            plt.axis("off")
            plt.tight_layout()
            plt.show()