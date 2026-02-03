import torch.nn as nn
import torchvision.models as models
from src.adapters import Adapter
from src.lora import LoRALinear

def build_model(model_name: str, num_classes: int = 7):
    if model_name.startswith("resnet"):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("mobilenet"):
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("vit"):
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model

def apply_freeze_strategy(model: nn.Module, model_name: str, strategy: str, freeze_layers: int = None):
    if strategy in ['lora', 'adapters']:
        return model

#-----------------------------------------------------Head_Only-----------------------------------------------------#
    if strategy == 'head_only':
        for param in model.parameters():
            param.requires_grad = False
        if model_name.startswith("resnet"):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name.startswith("efficientnet"):
            for param in model.classifier[1].parameters():
                param.requires_grad = True
        elif model_name.startswith("mobilenet"):
            for param in model.classifier[3].parameters():
                param.requires_grad = True
        elif model_name.startswith("vit"):
            for param in model.heads.head.parameters():
                param.requires_grad = True
#-------------------------------------------------------Full--------------------------------------------------------#
    elif strategy == 'full_finetuning':
        for param in model.parameters():
            param.requires_grad = True
#--------------------------------------------------freeze_n_layers--------------------------------------------------#
    elif strategy == 'freeze_n_layers':

        for param in model.parameters():
            param.requires_grad = True

        if model_name.startswith("resnet"):
            layers_for_resnet = [
                model.conv1,
                model.bn1,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4
            ]
            for i in range(freeze_layers):
                for param in layers_for_resnet[i].parameters():
                    param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

        elif model_name.startswith("efficientnet"):
            layers_for_efficientnet_b0 = [
                model.features[0],
                model.features[1],
                model.features[2],
                model.features[3],
                model.features[4],
                model.features[5],
                model.features[6],
                model.features[7],
                model.features[8],
            ]
            for i in range(freeze_layers):
                for param in layers_for_efficientnet_b0[i].parameters():
                    param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

        elif model_name.startswith("mobilenet"):
            layers = [
                model.features[0],
                model.features[1],
                model.features[2],
                model.features[3],
                model.features[4],
                model.features[5],
                model.features[6],
                model.features[7],
                model.features[8],
                model.features[9],
                model.features[10],
                model.features[11],
                model.features[12],
            ]
            for i in range(freeze_layers):
                for param in layers[i].parameters():
                    param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

        elif model_name.startswith("vit"):
            layers = [
                model.conv_proj,
                model.encoder.layers[0],
                model.encoder.layers[1],
                model.encoder.layers[2],
                model.encoder.layers[3],
                model.encoder.layers[4],
                model.encoder.layers[5],
                model.encoder.layers[6],
                model.encoder.layers[7],
                model.encoder.layers[8],
                model.encoder.layers[9],
                model.encoder.layers[10],
                model.encoder.layers[11],
                model.encoder.ln,
            ]
            for i in range(freeze_layers):
                for param in layers[i].parameters():
                    param.requires_grad = False
            for param in model.heads.parameters():
                param.requires_grad = True

    else:
        raise ValueError(f"Nieznana strategia: {strategy}")

    return model

def auto_lora_strategy_config(model_name: str, base_layer: nn.Linear):
    in_features = base_layer.in_features
    out_features = base_layer.out_features
    layer_size = min(in_features, out_features)

    if model_name.startswith("resnet"):
        r = min(8, layer_size // 64)
        dropout = 0.0

    elif model_name.startswith("efficientnet"):
        r = min(8, layer_size // 64)
        dropout = 0.05

    elif model_name.startswith("mobilenet"):
        r = min(4, layer_size // 128)
        dropout = 0.1

    elif model_name.startswith("vit"):
        r = min(16, layer_size // 32)
        dropout = 0.1

    else:
        r = 4
        dropout = 0.1

    r = max(1, r)

    alpha = r

    return {
        "r": r,
        "alpha": alpha,
        "dropout": dropout
    }

def auto_adapter_strategy_config(model_name: str, in_features: int):

    if model_name.startswith("resnet"):
        bottleneck_dim = max(16, in_features // 16)
        dropout = 0.0
    elif model_name.startswith("efficientnet"):
        bottleneck_dim = max(16, in_features // 16)
        dropout = 0.05
    elif model_name.startswith("mobilenet"):
        bottleneck_dim = max(8, in_features // 32)
        dropout = 0.1
    elif model_name.startswith("vit"):
        bottleneck_dim = max(32, in_features // 16)
        dropout = 0.1
    else:
        bottleneck_dim = 16
        dropout = 0.1

    return {
        "bottleneck_dim": bottleneck_dim,
        "dropout": dropout
    }

def get_parent_module(model: nn.Module, layer_name: str):
    parent = model
    parts = layer_name.split(".")
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def apply_adapter_strategy(model: nn.Module, model_name: str, strategy: str):
    if strategy == "lora":

        """
        LoRA:

        LoRA to technika wprowadzania korekcji wagi 'W' warstwy liniowej, poprzed dodawanie 'ΔW'
        do już istniejącego 'W', dzięki temu oszczędzamy na liczbie trenowanych parametrów. 'ΔW'
        Natomiast to inaczej iloczyn 'A * B', gdzie 'A' i 'B' to trenowalne macierze
        niskiego rzędu, które aproksymują aktualizację wag warstwy liniowej. 'A' - wybiera
        najważniejsze kierunki w których chcemy zmienić zachowanie wwarstwy, robi to przez
        mapowanie wejścia warstwy liniowej z przestrzeni wejściowej do przestrzeni niskowymiarowej,
        'B' - mówi natomiast jak te kierunki wpływają na wyjście warstwy, robi to  przez mapowanie
        korekty wyjścia warstwy z przestrzeni niskowymiarowej do przestrzeni wyjściowej warstwy,
        generując przy tym przyrost aktywacji dodawany do oryginalnego wyniku.

        W strategii LoRA wszystkie oryginalne parametry modelu pozostają zamrożone, a adaptacja do
        nowego zadania realizowana jest wyłącznie poprzez trenowanie niskowymiarowych macierzy
        korekcyjnych dodawanych do wybranej warstwy liniowej. W projekcie wybraną warstwą liniową
        jest warstwa klasyfikacyjna, przy całkowitym zamrożeniu parametrów modelu bazowego, co
        umożliwia efektywną adaptację przy minimalnym koszcie obliczeniowym.
        """
        for param in model.parameters():
            param.requires_grad = False

        if model_name.startswith("resnet"):
            layer_names = ["fc"]

        elif model_name.startswith("efficientnet"):
            layer_names = ["classifier.1"]

        elif model_name.startswith("mobilenet"):
            layer_names = ["classifier.3"]

        elif model_name.startswith("vit"):
            layer_names = [
                "encoder.layers.10.mlp.0",
                "encoder.layers.11.mlp.0",
                "heads.head"
            ]

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        for layer_name in layer_names:
            parent, attr = get_parent_module(model, layer_name)
            target_layer = getattr(parent, attr)

            cfg = auto_lora_strategy_config(model_name, target_layer)

            setattr(
                parent,
                attr,
                LoRALinear(
                    base_layer=target_layer,
                    r=cfg["r"],
                    alpha=cfg["alpha"],
                    dropout=cfg["dropout"],
                )
            )
    elif strategy == "adapters":

        """
        Adapters:

        Adaptery to technika adaptacji modelu polegająca na wstawieniu niewielkich,
        trenowalnych modułów sieci neuronowej pomiędzy warstwy modelu bazowego,
        przy jednoczesnym zamrożeniu wszystkich oryginalnych parametrów modelu.

        Typowy adapter składa się z dwóch warstw liniowych: pierwsza wykonuje
        projekcję aktywacji do przestrzeni niskowymiarowej, a druga mapuje je
        z powrotem do oryginalnego wymiaru. Wynik adaptera dodawany jest do
        oryginalnych aktywacji za pomocą połączenia resztkowego.

        W przeciwieństwie do LoRA, która wprowadza korekty bezpośrednio do wag
        warstw liniowych, adaptery operują na aktywacjach w trakcie przejścia
        sygnału przez sieć.

        W strategii adapterów wszystkie parametry modelu bazowego pozostają
        zamrożone, a adaptacja do nowego zadania realizowana jest poprzez
        trenowanie wyłącznie parametrów adapterów (oraz opcjonalnie warstwy
        klasyfikacyjnej), co znacząco redukuje liczbę trenowanych parametrów
        i koszt obliczeniowy.

        Dla większośći modeli przy małym datasecie, powszechnie przyjętą i logiczną 
        praktyką jest wstawianie adapterów przed warstwą klasyfikacyjną, w miejscu, 
        gdzie model kończy ekstrakcję cech.
        """

        for param in model.parameters():
            param.requires_grad = False

        if model_name.startswith("resnet"):
            layer_names = ["fc"]
        elif model_name.startswith("efficientnet"):
            layer_names = ["classifier.1"]
        elif model_name.startswith("mobilenet"):
            layer_names = ["classifier.3"]
        elif model_name.startswith("vit"):
            layer_names = [
                "encoder.layers.10.mlp.0",
                "encoder.layers.11.mlp.0",
                "heads.head"
            ]
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        for layer_name in layer_names:
            parent, attr = get_parent_module(model, layer_name)
            target_layer = getattr(parent, attr)

            cfg = auto_adapter_strategy_config(model_name, target_layer.in_features)

            adapter_module = Adapter(
                in_features=target_layer.out_features,
                bottleneck_dim=cfg["bottleneck_dim"],
                dropout=cfg["dropout"]
            )

            new_module = nn.Sequential(target_layer, adapter_module)

            setattr(parent, attr, new_module)

            for p in target_layer.parameters():
                p.requires_grad = False

    return model

def prepare_model(config: dict) -> nn.Module:
    """
    Buduje i przygotowuje model do treningu na podstawie konfiguracji eksperymentu.

    Kroki:
    1. Budowa modelu bazowego
    2. Wstrzyknięcie LoRA / Adapterów (jeśli wybrana strategia)
    3. Zastosowanie strategii zamrażania parametrów
    """

    model = build_model(
        model_name=config["model_name"]
    )

    if config["strategy"] in ["lora", "adapters"]:
        model = apply_adapter_strategy(
            model=model,
            model_name=config["model_name"],
            strategy=config["strategy"]
        )

    model = apply_freeze_strategy(
        model=model,
        model_name=config["model_name"],
        strategy=config["strategy"],
        freeze_layers=config.get("freeze_layers", 7)
    )

    return model
