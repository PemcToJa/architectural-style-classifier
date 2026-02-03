# Analiza porównawcza metod adaptacji głębokich sieci neuronowych w klasyfikacji stylów architektonicznych

## Autor: Przemysław Rządkowski

### Abstrakt
Niniejszy projekt przedstawia proces adaptacji zaawansowanych modeli wizyjnych do zadania klasyfikacji stylów architektonicznych. W ramach badań przeprowadzono analizę porównawczą tradycyjnego procesu fine-tuningu z nowoczesnymi metodami Parameter-Efficient Fine-Tuning (PEFT), takimi jak LoRA (Low-Rank Adaptation) oraz moduły Adapterów. Badania objęły zarówno architektury splotowe (CNN), jak i modele oparte na mechanizmie uwagi (Vision Transformers). Najwyższą skuteczność klasyfikacji na poziomie 94.1% uzyskano przy wykorzystaniu modelu ViT-B16 z zaimplementowanymi modułami adapterów.

---

## 1. Cel i zakres badań
Głównym celem projektu jest ocena efektywności różnych strategii douczania modeli w warunkach ograniczonej liczby danych treningowych. Zakres prac obejmował:
* Implementację i porównanie czterech architektur bazowych: ResNet50, EfficientNet-B0, MobileNetV3 oraz ViT-B16.
* Analizę pięciu strategii uczenia: Full Fine-Tuning, Head-Only, Freeze-n-layers, LoRA oraz Adapters.
* Ewaluację wpływu architektury modelu na zdolność do adaptacji niskowymiarowej.

## 2. Metodyka i przeprowadzone eksperymenty
Eksperymenty zostały podzielone na trzy główne kategorie:
1. **Tradycyjny Fine-tuning**: Obejmujący warianty pełnego douczania, trenowania wyłącznie klasyfikatora oraz mrożenia początkowych bloków warstw.
2. **Parameter-Efficient Fine-Tuning (PEFT)**: Autorska implementacja modułów LoRA oraz Adapterów wstrzykiwanych w warstwy liniowe modeli.
3. **Analiza Baseline**: Wykorzystanie modeli jako ekstraktorów cech dla klasyfikatora SVM (Support Vector Machine).



## 3. Zbiór danych
Wykorzystano zbiór danych powstały z integracji trzech niezależnych źródeł([1-dataset](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset), [2-dataset](https://www.kaggle.com/datasets/josephgoksu/architectural-styles), [3-dataset](https://www.kaggle.com/datasets/jungseolin/international-architectural-styles-combined)), obejmujący 7 klas stylów architektonicznych:
* Achaemenid, American Craftsman, Ancient Egyptian, Edo Period, Baroque, Beaux-Arts, Gothic.
* **Preprocessing**: Zastosowano skrypty do usuwania duplikatów oraz ujednolicenia formatów danych.
* **Podział danych**: Dane podzielono w proporcji 70% trening / 15% walidacja / 15% test.
* **Augmentacja**: W celu zwiększenia generalizacji zastosowano techniki takie jak: obrót losowy, przycinanie (cropping), flip poziomy oraz modyfikacje jasności, kontrastu i saturacji.

## 4. Wyniki eksperymentów
Poniższa tabela przedstawia najwyższe uzyskane wyniki skuteczności (Validation Accuracy) dla poszczególnych kombinacji modelu i strategii:

| Architektura | Head-Only | Full FT | Freeze-n-Layers | LoRA | Adapter | Baseline (SVM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT-B16** | 93.3% | 54.1% | 92.8% | 93.8% | **94.1%** | 91.5% |
| **ResNet50** | 87.9% | 89.5% | 93.3% | 17.4% | 33.8% | 92.1% |
| **EfficientNet-B0** | 86.4% | 93.6% | 92.8% | 30.5% | 34.6% | 87.1% |
| **MobileNetV3** | 85.4% | 89.5% | 90.3% | 36.7% | 27.2% | 91.0% |



## 5. Kluczowe wnioski
1. **Efektywność modeli Transformer**: Model ViT-B16 wykazał najwyższą adaptowalność przy użyciu metod PEFT. Wynik 94.1% dla adapterów znacząco przewyższa pełny fine-tuning (54.1%), co wskazuje na uniknięcie zjawiska katastrofalnego zapominania.
2. **Ograniczenia PEFT w CNN**: Metody LoRA i Adapters (skupione na warstwach liniowych) okazały się mało efektywne dla sieci splotowych w tej konkretnej implementacji, co wynika z mrożenia kluczowych warstw konwolucyjnych.
3. **Wartość cech bazowych**: Wysoki wynik Baseline (92.1% dla ResNet50) potwierdza wysoką transferowalność cech wyuczonych na zbiorze ImageNet do dziedziny architektury.

## 6. Technologie i narzędzia
* **Framework**: PyTorch
* **Przetwarzanie danych**: NumPy, Pandas, Scikit-learn
* **Wizualizacja**: Matplotlib, Seaborn

## 7. Jak skonfigurować projekt
### Wyniki ewaluacji modeli:
Ze względu na duży rozmiar plików w folderu, wyniki ewaluacji są przechowywane zewnętrznie.
* Wyniki ewaluacji: [Google Drive - Architectural-style-classifier-results](https://drive.google.com/drive/folders/1ny-utCc9cVr1yNQx75y2jhQNUfgQY8kH?usp=sharing), są one gotowe do pobrania. W przypadu ich pobrania należy umieścić je w głównym folderze projektowym.

### Instalacja zależności:
Zalecane jest użycie środowiska wirtualnego (Python 3.10+):
```bash
python -m venv .venv
```
### Windows:
```bash
.venv\Scripts\activate
```
### Linux/Mac:
```bash
source .venv/bin/activate
```
### Requirements:
```bash
pip install -r requirements.txt
```