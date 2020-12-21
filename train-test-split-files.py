import os
import shutil

# Stratified 80/20 split of an "image folder" dataset.

root = "data/ImageNet"
class_names = ["cat", "dog", "bear"]

def str_hash(s: str) -> int:
    h = 5381
    for c in s:
        h = (h * 33 + ord(c)) % (1 << 64)
    return h

for class_name in class_names:
    print(f"{class_name}...")
    train_dir = os.path.join(root, class_name + "_train", class_name)
    test_dir = os.path.join(root, class_name + "_test", class_name)
    for dirname in [train_dir, test_dir]:
        os.makedirs(dirname, exist_ok=True)
    all_dir = os.path.join(root, class_name, class_name)
    for filename in os.listdir(all_dir):
        target_dir = train_dir if str_hash(filename) % 10 < 8 else test_dir
        shutil.copyfile(os.path.join(all_dir, filename), os.path.join(target_dir, filename))