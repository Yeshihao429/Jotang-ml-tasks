"""
文件作用：下载并解压文件，自动删除zip，根据图片名自动分到cats或dogs文件夹中
"""

import kagglehub
import os
import zipfile
import shutil

custom_save_path = "./cats_and_dogs/"

# 判断非空
if os.path.isdir(custom_save_path) and os.listdir(custom_save_path):
    print(f"[跳过下载] 目录 {custom_save_path} 已存在且非空。")
    path = custom_save_path
else:
    os.makedirs(custom_save_path, exist_ok=True)
    path = kagglehub.competition_download(
        'dogs-vs-cats',
        output_dir=custom_save_path
    )
    print("Path to competition files:", path)


# 解压并拉平目录
def extract_and_flatten(zip_path, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if n.strip()]
        top_level = {n.split('/')[0] for n in names}
        single_top = (len(top_level) == 1) and all('/' in n for n in names)

        if single_top:
            tmp_dir = target_dir + "_tmp"
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)
            zf.extractall(tmp_dir)

            inner = os.path.join(tmp_dir, top_level.pop())
            for item in os.listdir(inner):
                shutil.move(os.path.join(inner, item),
                            os.path.join(target_dir, item))
            shutil.rmtree(tmp_dir)
        else:
            zf.extractall(target_dir)


for file_name in os.listdir(path):
    if file_name.endswith(".zip"):
        zip_full_path = os.path.join(path, file_name)
        extract_target_dir = os.path.join(path, os.path.splitext(file_name)[0])
        extract_and_flatten(zip_full_path, extract_target_dir)
        print(f"已完成解压并拉平：{file_name} -> {extract_target_dir}")


# -删zip
for file_name in os.listdir(path):
    if file_name.endswith(".zip"):
        os.remove(os.path.join(path, file_name))
        print(f"已清理原始压缩包：{file_name}")


# 清理没用的文件，例如.csv、.complete之类的
for item in [".complete", "sampleSubmission.csv"]:
    item_path = os.path.join(path, item)
    if os.path.exists(item_path):
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)
        print(f"已删除：{item}")


# 按名称分类
def split_by_prefix(folder):
    if not os.path.isdir(folder):
        print(f"[跳过] {folder} 不存在")
        return

    existing = {d.lower() for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))}
    if "cat" in existing and "dog" in existing:
        print(f"[跳过分类] {folder} 已有 cat/dog 子目录")
        return

    cat_dir = os.path.join(folder, "cat")
    dog_dir = os.path.join(folder, "dog")
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    n_cat, n_dog, n_skip = 0, 0, 0
    for f in os.listdir(folder):
        src = os.path.join(folder, f)
        if not os.path.isfile(src):
            continue
        low = f.lower()
        if low.startswith("cat") and low.endswith((".jpg", ".jpeg", ".png")):
            shutil.move(src, os.path.join(cat_dir, f))
            n_cat += 1
        elif low.startswith("dog") and low.endswith((".jpg", ".jpeg", ".png")):
            shutil.move(src, os.path.join(dog_dir, f))
            n_dog += 1
        else:
            n_skip += 1

    print(f"{folder}: cat {n_cat} 张, dog {n_dog} 张, 跳过 {n_skip} 个")


train_dir = os.path.join(path, "train")
test_dir  = os.path.join(path, "test1")

split_by_prefix(train_dir)
split_by_prefix(test_dir)



print("\n=== 最终目录结构 ===")
for sub in ["train/cat", "train/dog", "test1/cat", "test1/dog"]:
    d = os.path.join(path, sub)
    if os.path.isdir(d):
        print(f"{sub}: {len(os.listdir(d))} 张")
    else:
        print(f"{sub}: 不存在" )