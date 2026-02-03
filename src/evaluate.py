import os
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_preprocessing import test_dataset, val_test_transform, classes
from src.model import prepare_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

root_project = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MODELS_ROOT = root_project / "models"
RESULTS_ROOT = root_project / "models"
ALL_HISTORY_PATH = root_project / "src/all_history.json"


def list_models():
    model_folders = [d for d in os.listdir(MODELS_ROOT) if os.path.isdir(os.path.join(MODELS_ROOT, d))]
    model_folders.sort()
    print("Wybierz model do ewaluacji:")
    for i, name in enumerate(model_folders, 1):
        print(f"{i}. {name}")
    return model_folders


def find_model_file(folder_path, folder_name):
    for f in os.listdir(folder_path):
        if f.startswith(folder_name) and f.endswith(".pth"):
            return os.path.join(folder_path, f)
    raise FileNotFoundError(f"Nie znaleziono pliku modelu w folderze {folder_path} zaczynającego się od {folder_name}")


def prepare_test_loader():
    test_dataset.transform = val_test_transform
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def evaluate_model(model, test_loader):
    model.eval()
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    test_acc = report["accuracy"]

    return test_loss, test_acc, cm, report


def save_metrics(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)


def save_model_history(all_history_path, model_id_prefix, output_dir):
    with open(all_history_path, "r") as f:
        all_history = json.load(f)

    filtered_history = {k: v for k, v in all_history.items() if k.startswith(model_id_prefix)}

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "finetuning_history_data.json")
    with open(save_path, "w") as f:
        json.dump(filtered_history, f, indent=4)

    print(f"Zapisano historię dla '{model_id_prefix}' do {save_path}")
    return filtered_history[list(filtered_history.keys())[0]] if filtered_history else None


def plot_and_save_results(output_dir, cm, history, classes):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()

if __name__ == "__main__":
    model_folders = list_models()
    choice = int(input(">> ")) - 1
    folder_name = model_folders[choice]
    folder_path = MODELS_ROOT / folder_name

    model_file_path = find_model_file(folder_path, folder_name)
    print(f"Wczytywanie modelu: {model_file_path}")

    parts = folder_name.split("_")

    if parts[0] in ["efficientnet", "vit"]:
        model_name = "_".join(parts[:2])
        strategy = "_".join(parts[2:])
    else:  # mobilenetv3, resnet50
        model_name = parts[0]
        strategy = "_".join(parts[1:])

    model = prepare_model({
        "model_name": model_name,
        "strategy": strategy
    })
    model.load_state_dict(torch.load(model_file_path, map_location=DEVICE))

    test_loader = prepare_test_loader()

    test_loss, test_acc, cm, report = evaluate_model(model, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    results_dir = RESULTS_ROOT / folder_name
    os.makedirs(results_dir, exist_ok=True)

    save_metrics(report, results_dir)
    history = save_model_history(ALL_HISTORY_PATH, folder_name, results_dir)

    if history:
        plot_and_save_results(results_dir, cm, history, classes)
