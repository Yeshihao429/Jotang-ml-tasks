import torch
import torch.nn as nn
import yaml


class CatsDogsModel(nn.Module):
    def __init__(self):
        super(CatsDogsModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    from data_processor_3 import process_image
    import random

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    root = cfg["dataset_root"]
    train_loader, _ = process_image(root)
    dataset = train_loader.dataset

    idx = random.randint(0, len(dataset) - 1)
    img, label = dataset[idx]
    x = img.unsqueeze(0)  # 在第0维补上batch

    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    idx_to_class = {v: k for k, v in base.class_to_idx.items()}
    print(f"随机样本: idx={idx}, 类别={idx_to_class[label]}")
    print(f"{'输入':<28s} -> {tuple(x.shape)}   (N, C, H, W)")

    model = CatsDogsModel()
    model.eval()

    with torch.no_grad():
        # feature层
        for name, layer in model.features.named_children():
            x = layer(x)
            tag = f"features[{name}] {layer.__class__.__name__}"
            print(f"{tag:<28s} -> {tuple(x.shape)}")

        # flatten层
        x = torch.flatten(x, 1)
        print(f"{'flatten':<28s} -> {tuple(x.shape)}")

        # classifier层
        for name, layer in model.classifier.named_children():
            x = layer(x)
            tag = f"classifier[{name}] {layer.__class__.__name__}"
            print(f"{tag:<28s} -> {tuple(x.shape)}")