import copy
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_preprocessing import train_dataset, val_dataset, train_transform, val_test_transform
from src.model import prepare_model

root_project = Path(__file__).resolve().parent.parent


def configure_experiment():
    print("\n-----------------------------------------KONFIGURACJA EKSPERYMENTU-----------------------------------------\n")

#-------------------------------------------Modele-------------------------------------------#
    models = {
        1: "resnet50",
        2: "efficientnet_b0",
        3: "mobilenetv3",
        4: "vit_b16"
    }

    print("Wybierz model bazowy:")
    for k, v in models.items():
        print(f"{k}. {v}")

    model_choice = int(input(">> "))
    model_name = models[model_choice]

#-----------------------------------------Strategie------------------------------------------#
    strategies = {
        1: "full_finetuning",
        2: "head_only",
        3: "adapters",
        4: "lora",
        5: "freeze_n_layers"
    }

    print("\nWybierz strategię fine-tuningu:")
    for k, v in strategies.items():
        print(f"{k}. {v}")

    strategy_choice = int(input(">> "))
    strategy = strategies[strategy_choice]

    freeze_layers = None
    if strategy == "freeze_n_layers":
        freeze_layers = int(input("Podaj liczbę zamrożonych warstw: "))

#-----------------------------------------Agumentacja----------------------------------------#
    augmentations = {
        1: "augmentacja",
        2: "brak_augmentacji"
    }

    print("\nWybierz preprocessing:")
    print("1. Augmentacja danych (dla treningu)")
    print("2. Brak augmentacji (np. dla walidacji/testu)")

    augmentation_choice = int(input(">> "))
    augmentation = augmentations[augmentation_choice]

#---------------------------------------Hiperparametry---------------------------------------#
    print("\nHiperparametry treningu:")
    learning_rate = float(input("Learning rate: "))
    batch_size = int(input("Batch size: "))
    num_epochs = int(input("Liczba epok: "))

#----------------------------------------Optymalizator---------------------------------------#
    optimizers = {
        1: "Adam",
        2: "SGD",
        3: "RMSprop"
    }

    print("\nOptymalizator:")
    for k, v in optimizers.items():
        print(f"{k}. {v}")

    optimizer_name = optimizers[int(input(">> "))]

#-----------------------------------------Scheduler------------------------------------------#
    schedulers = {
        0: None,
        1: "StepLR",
        2: "CosineAnnealing",
        3: "ReduceLROnPlateau"
    }

    print("\nScheduler LR:")
    print("0. Brak")
    print("1. StepLR")
    print("2. CosineAnnealing")
    print("3. ReduceLROnPlateau")

    scheduler = schedulers[int(input(">> "))]

#---------------------------------------Regularyzacja----------------------------------------#
    print("\nRegularyzacja:")
    weight_decay = float(input("Weight decay (0.0 jeśli brak): "))
    label_smoothing = input("Label smoothing? (t/n): ").lower() == "t"

#-----------------------------------------Monitoring-----------------------------------------#
    print("\nMonitorowana metryka:")
    print("1. val_acc")
    print("2. val_loss")
    monitor = "val_acc" if int(input(">> ")) == 1 else "val_loss"

#-------------------------------------------Ogólne-------------------------------------------#
    device = input("Device (cpu / cuda): ")

    print("\n-----------------------------------------PODSUMOWANIE KONFIGURACJI-----------------------------------------\n")
    print(f"Model bazowy:            {model_name}")
    print(f"Strategia fine-tuningu:  {strategy}")
    if strategy == "freeze_n_layers":
        print(f"Zamrożone warstwy:       {freeze_layers}")

    print(f"Augmentacja danych:      {augmentation}")
    print(f"Learning rate:           {learning_rate}")
    print(f"Batch size:              {batch_size}")
    print(f"Liczba epok:             {num_epochs}")
    print(f"Optymalizator:           {optimizer_name}")
    print(f"Scheduler LR:            {scheduler}")
    print(f"Weight decay:            {weight_decay}")
    print(f"Label smoothing:         {label_smoothing}")
    print(f"Monitorowana metryka:    {monitor}")
    print(f"Device:                  {device}")

    print("\n-----------------------------------------------------------------------------------------------------------\n")

    confirm = input("Czy uruchomić eksperyment z tą konfiguracją? (t/n): ").lower()
    if confirm != "t":
        print("\nEksperyment anulowany.")
        sys.exit(0)

    print("\nKonfiguracja zatwierdzona. Start treningu...\n")

    return {
        "model_name": model_name,
        "strategy": strategy,
        "freeze_layers": freeze_layers,
        "augmentation": augmentation,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "optimizer_name": optimizer_name,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "monitor": monitor,
        "device": device
    }

def train_model(model, config, train_loader, val_loader):
    device = config["device"]
    model.to(device)

    if config["label_smoothing"]:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    if config["optimizer_name"].lower() == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer_name"].lower() == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"], momentum=0.9)
    elif config["optimizer_name"].lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Nieznany optymalizator: {config['optimizer_name']}")

    scheduler = None
    if config["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif config["scheduler"] == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min' if config["monitor"]=="val_loss" else 'max')

    best_metric = -float("inf") if config["monitor"] == "val_acc" else float("inf")
    best_model_state = copy.deepcopy(model.state_dict())

    metric_key = "best_val_acc" if config["monitor"] == "val_acc" else "best_val_loss"

    current_history = {
        'experiment_id': f"{config['model_name']}_{config['strategy']}_lr{config['learning_rate']}_ep{config['num_epochs']}",
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        metric_key: best_metric,
        'model_name': config['model_name'],
        'strategy': config['strategy'],
        'learning_rate': config['learning_rate'],
        'num_epochs': config['num_epochs'],
        'batch_size': config['batch_size'],
        'augmentation': [config['augmentation']],
        'optimizer': config['optimizer_name'],
        'dataset_name': 'architectural_styles'
    }

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_corrects += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_corrects / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)

        current_metric = val_acc if config["monitor"] == "val_acc" else val_loss

        if scheduler is not None:
            if config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(val_loss if config["monitor"] == "val_loss" else val_acc)
            else:
                scheduler.step()

        if (config["monitor"] == "val_acc" and current_metric > best_metric) or \
                (config["monitor"] == "val_loss" and current_metric < best_metric):
            best_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())
            DIR_NAME = config["model_name"] + "_" + config["strategy"]
            save_path = root_project / "models" / DIR_NAME / f"{config['model_name']}_{config['strategy']}_best.pth"

            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(best_model_state, save_path)
            current_history[metric_key] = best_metric

        current_history["train_loss"].append(train_loss)
        current_history["train_acc"].append(train_acc)
        current_history["val_loss"].append(val_loss)
        current_history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    model.load_state_dict(best_model_state)
    return model, current_history


def save_to_all_history(current_history, filename="all_history.json"):
    try:
        with open(filename, "r") as f:
            all_history = json.load(f)
    except FileNotFoundError:
        all_history = {}

    key = current_history["experiment_id"]
    all_history[key] = current_history

    with open(filename, "w") as f:
        json.dump(all_history, f, indent=4)


def get_data_loaders(config):
    if config["augmentation"] == "augmentacja":
        train_dataset.transform = train_transform
    else:
        train_dataset.transform = val_test_transform

    val_dataset.transform = val_test_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader

if __name__ == "__main__":
    config = configure_experiment()

    train_loader, val_loader = get_data_loaders(config)

    model = prepare_model(config)

    trained_model, history = train_model(model, config, train_loader, val_loader)

    save_to_all_history(history)