import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

root_project = Path(__file__).resolve().parent.parent

data_path = root_project / "data/raw/Architecture_art_styles"
train_data_path = root_project / "data/processed/train"
validation_data_path = root_project / "data/processed/val"
test_data_path = root_project / "data/processed/test"

classes = sorted(os.listdir(data_path))

dataset = [
    (img_path, class_name)
    for class_name in classes
    for img_path in (data_path / class_name).glob("*.jpg")
]

class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

dataset_numeric = [
    (img_path, class_to_idx[class_name])
    for img_path, class_name in dataset
]

train_files, temp_files  = train_test_split(
    dataset_numeric,
    test_size=0.3,
    stratify=[label for _, label in dataset_numeric],
    random_state=SEED
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=0.5,
    stratify=[label for _, label in temp_files],
    random_state=SEED
)

for p in [train_data_path, validation_data_path, test_data_path]:
    if p.exists():
        shutil.rmtree(p)

for split_path in [train_data_path, validation_data_path, test_data_path]:
    for cls_name in classes:
        (split_path / cls_name).mkdir(parents=True, exist_ok=True)

def copy_files(file_list, dest_root):
    for img_path, label in file_list:
        cls_name = classes[label]
        shutil.copy(img_path, dest_root / cls_name / img_path.name)

copy_files(train_files, train_data_path)
copy_files(val_files, validation_data_path)
copy_files(test_files, test_data_path)

#------------------------------------------Agumentacja-----------------------------------------#

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = ImageFolder(
    root=train_data_path,
    transform=None
)

val_dataset = ImageFolder(
    root=validation_data_path,
    transform=None
)

test_dataset = ImageFolder(
    root=test_data_path,
    transform=None
)