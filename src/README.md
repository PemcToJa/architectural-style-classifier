# Crowd Counting System
**Autor:** Przemysław Rządkowski 

**Temat:** Estymacja liczby osób w gęstym tłumie oraz w tłumie bardziej rozproszonym (ShanghaiTech part A & B)

---
## 1. Opis Projektu i Filozofia Architektury
Projekt realizuje zadanie liczenia ludzi w ekstremalnie gęstych scenach. Zastosowany model to autorska implementacja CSRNet, pracująca w paradygmacie **Encoder-Decoder**.

### 1.1. Fundament Wiedzy: ImageNet i problem małego zbioru danych
**Na czym trenowano VGG16?**
Baza ta została pretrenowana na zbiorze **ImageNet** – gigantycznej bazie danych zawierającej ponad **14 milionów 
obrazów**. Dzięki temu model posiada "gotową wiedzę" o krawędziach, kształtach i teksturach, której nie musiał 
uczyć się od zera na wybranym do problemu datasecie.
&nbsp;

**Dlaczego trenowanie od zera byłoby niemożliwe?**

Zbiór [**ShanghaiTech**](https://www.kaggle.com/datasets/tthien/shanghaitech-with-people-density-map/data) (łącznie ok. 1200 zdjęć w part A i B) jest zbyt mały, by wytrenować tak głęboką sieć jak 
CSRNet od podstaw. 

* Gdyby zaczęto trening od losowych wag, model zamiast zrozumieć cechy tłumu, po prostu wykułby zdjęcia na 
pamięć (**overfitting**). 
* Wykorzystanie VGG16 pozwala na **Transfer Learning** – bierzemy model, który "umie widzieć", i 
nakierowujemy go na liczenie ludzi.

**Charakterystyka Datasetu:**
Zbiór zawiera adnotacje punktowe (współrzędne głów), które zostały przetworzone na **mapy gęstości (H5)**. 
Wykorzystano specyficzne kernele Gaussa, aby zamienić pojedyncze punkty w rozmyte plamy energii, co pozwala 
modelowi uczyć się rozkładu przestrzennego tłumu, a nie tylko pojedynczych pikseli.

**Kompletne wytłumacznie:**

### 1.2. Encoder (Ekstrakcja Cech - VGG16 Frontend)
Jako "mózg" systemu wykorzystano pierwsze 10 warstw splotowych sieci **VGG16** oraz odcięcie go od reszty modelu w `block4_conv3`, robimy to z 3 powodów:

1. **Sieci neuronowe działają jak lejek. Każda warstwa MaxPooling zmniejsza obraz o połowę.** 
   * Wejście:   &nbsp;&nbsp;&nbsp;&nbsp;   512x512
   * Po Block 1: 256x256
   * Po Block 2: 128x128
   * Po Block 3: 64x64
   * Po Block 4: 32x32 (gdybyśmy poszli dalej)
   
   Zatrzymując się na poziomie **'block4_conv3'** (który jest przed poolingiem czwartego bloku), pracujemy na mapie **64x64**. Dlaczego 
to ważne? Bo jeśli obrazek jest zbyt mały (np. 32x32), to małe głowy ludzi, które mają po kilka pikseli, po prostu "znikają" – model nie ma 
fizycznie miejsca, żeby je narysować. 64x64 to najmniejszy rozmiar, który wciąż pozwala odróżnić od siebie ludzi w gęstym tłumie.



2. **Filtry w sieciach splotowych uczą się hierarchicznie:**
   * **Początkowe warstwy (Block 1-2):** Widzą krawędzie, kropki i kolory. To za mało – model myliłby głowę z kamieniem lub liściem.
   * **Środkowe warstwy (Block 3-4):** Zaczynają rozpoznawać złożone kształty: owale, struktury przypominające twarze, ramiona. To jest idealne do liczenia ludzi.
   * **Końcowe warstwy (Block 5):** Szukają całych obiektów (np. "cały samochód"). W gęstym tłumie rzadko widać całego człowieka, więc te filtry są tu bezużyteczne.



3. **Wybór optymalnego punktu - "Złoty Środek":**

   Gdybyśmy ucięli model wcześniej (np. na Block 2), mielibyśmy świetną rozdzielczość (256x256), ale model byłby "głupi" – nie odróżniałby głowy od 
innego okrągłego przedmiotu. Gdybyśmy ucięli model później (np. na Block 5), model byłby bardzo "mądry", ale miałby fatalną rozdzielczość (32x32 lub 16x16) i 
nie wiedziałby dokładnie, gdzie stoją ludzie. Wybierając **block4_conv3** bierzemy co najlepsze z obu światów: mamy już filtry, które "rozumieją", jak wygląda 
człowiek oraz jeszcze wystarczająco dużo pikseli, żeby ich precyzyjnie policzyć. **Wynik:** Tensor (None, 64, 64, 512), czyli **"książka o 512 stronach"**, gdzie 
każda strona opisuje inną cechę wizualną.

### 1.2.1 Dlaczego NIE używamy warstw Flatten?
**W przeciwieństwie do klasycznych autoencoderów (np. dla MNIST z labu 9), w tym modelu całkowicie zrezygnowano z warstw `Flatten()`.**

**Zachowanie informacji przestrzennej:** Warstwa `Flatten` "rozjeżdża" obraz do jednego wektora, niszcząc informację o tym, gdzie obok siebie znajdowały się 
dane piksele. W liczeniu tłumu informacja "gdzie" jest kluczowa do poprawnej lokalizacji osób.

### 1.3. Decoder (Agregacja Kontekstu i "Rekonstrukcja")
**Agregacja Kontekstu (Dilated Backend – Pętla for):**

Warstwy te mają dwa cele: stopniowe **"odchudzanie"** danych (redukcja filtrów z 512 do 64) oraz drastyczne 
zwiększenie pola widzenia (**Receptive Field**). **Mechanizm Dylatacji:** Zamiast standardowego 
splotu(dilation_rate=1), używamy dylatacji 2 i 4. Filtr 3x3 **"rozczapierza palce"** – nie zwiększamy liczby 
wag(nadal tylko 9 punktów), ale zwiększamy zasięg (do 5x5 i 9x9). **Dlaczego to kluczowe?** Dzięki temu model 
unika **"wizji tunelowej"**. Widząc beżowy owal, filtr dylatacyjny widzi też kontekst (ramiona, inne głowy obok). 
Pozwala to odróżnić człowieka od podobnych obiektów w tle (np. kamieni) oraz poprawnie ocenić gęstość tłumu bez 
użycia warstw Pooling, które zniszczyłyby rozdzielczość. Rola Batch Normalization (BN): Wprowadzenie warstw BN po 
każdym splocie dylatacyjnym było punktem zwrotnym projektu. Działają one jak "filtr stabilizujący" – normalizują 
wartości płynące z "rozczapierzonych" filtrów, zapobiegając ich nadmiernej ekscytacji (tzw. Internal Covariate Shift). 
Dzięki temu model nie "pływa" w obliczeniach i potrafi zachować matematyczny rygor. To właśnie ten element pozwolił na
drastyczne zredukowanie błędu z początkowych 79% (przeszacowanie wyniku) do zaledwie 0.17% błędu względem 
rzeczywistości co do zliczania osób z obrazów. BN sprawia, że mapa gęstości jest "czysta" – tło pozostaje martwe, a energia 
modelu skupia się wyłącznie na głowach ludzi.


**Generowanie Mapy Gęstości (Output Layer):** 

Warstwa Conv2D(1, (1, 1)) wykonuje ostateczną syntezę. Bierze 64 wyselekcjonowane cechy z każdego punktu i "sprasowuje"
je do jednej wartości. Wynik: Tensor (64, 64, 1). To jest nasza surowa mapa gęstości. Jasne 
punkty (rozmyte plamy Gaussa) oznaczają obecność ludzi, a ciemne – tło. Zależność: Każda wartość w tej mapie odpowiada
przewidywanej liczbie osób w danym fragmencie obrazu.


**Rekonstrukcja Rozmiaru (UpSampling):** 

UpSampling2D(size=(8, 8), interpolation='bilinear') powiększa mapę z 64x64 do 512x512. Interpolacja Bilinearna: 
Nie jest to zwykłe powielenie pikseli, ale inteligentne wygładzanie. Dzięki temu otrzymujemy estetyczną, płynną 
mapę gęstości, która idealnie pokrywa się z oryginalnym zdjęciem. Finalny cel: Dzięki przywróceniu wymiaru 512x512 
możemy bezpośrednio porównać wynik modelu z naszymi etykietami (Ground Truth) i łatwo policzyć ludzi poprzez 
zsumowanie wszystkich wartości na mapie: total_count = np.sum(output_map).

### 1.4 Wnioski Inżynierskie: Dlaczego Transfer Learning był niezbędny?

Częstym pytaniem badawczym jest możliwość wytrenowania architektury CSRNet od zera (z losowymi wagami). W przypadku 
tego projektu byłoby to **niemożliwe do zrealizowania** z zachowaniem obecnej precyzji z trzech powodów:

1. **Deficyt danych:** Zbiór ShanghaiTech (zaledwie kilkaset zdjęć na set w partach A oraz B) jest zbyt mały, by 
model mógł samodzielnie wykształcić filtry rozpoznające skomplikowane cechy anatomiczne człowieka.

2. **Generalizacja:** Model trenowany od zera na tak małym zbiorze uległby **overfittingowi** (nauczyłby się 
zdjęć na pamięć, tracąc zdolność liczenia ludzi na nowych fotografiach).

3. **Stabilność wag:** Dzięki VGG16 startujemy z poziomu, w którym model "umie już widzieć". Nasz trening był 
jedynie procesem **Fine-tuningu**, czyli ukierunkowania istniejącej inteligencji modelu na 
specyficzny problem map gęstości tłumu. Bez wag z ImageNet, błąd na poziomie **0.17%** dla wybranych zdjęć testowych 
byłby matematycznie nieosiągalny przy obecnej wielkości datasetu.
---
## 2. Eksperymenty i Porównanie Metod
Zgodnie z wymaganiami, przeprowadzono porównanie różnych podejść do architektury i preprocessingu.

### 2.1 Wpływ Batch Normalization
Wstępne testy modelu bez warstw BN wykazały dużą niestabilność. Model przy gęstym tłumie (part A) drastycznie 
przeszacowywał wyniki (np. 890 osób zamiast rzeczywistych 496 - przykład z wybranego zdjęcia testowego, którego 
model nie widział ani w secie treningowym, ani w walidacyjnym).

**Wniosek:** 

Dodanie `BatchNormalization` po każdej warstwie dylatacyjnej pozwoliło ustabilizować aktywacje i "wyczyścić" 
tło, redukując błąd do poziomu błędu pomijalnego.

### 2.2 Precyzyjna Normalizacja Barw (ImageNet Stats)
W projekcie zrezygnowano z uproszczonej normalizacji (dzielenie przez 255). Zamiast tego 
zastosowano **Mean Subtraction** przy użyciu konkretnych wartości: `[123.68, 116.779, 103.939]`.

**Dlaczego to ważne?** 

Ponieważ VGG16 był oryginalnie trenowany na obrazach znormalizowanych w ten właśnie sposób. Dzięki odjęciu 
tych średnich wartości, "podajemy" modelowi dane w formacie, który on najlepiej rozumie. Pozwala to na pełne 
wykorzystanie pretrenowanych wag i zapobiega sytuacji, w której model musiałby marnować pierwsze epoki na 
re-adaptację do innej jasności i kontrastu zdjęć.

### 2.3 Optymalizacja Sigmy Gaussa
Po przebadaniu szerokości jądra Gaussa ($\sigma$) na precyzję modelu:

* **Sigma 4.0 (part B):** Optymalna dla rzadszego tłumu, pozwalająca na lepszą separację osób.

* **Sigma 2.1 (part A):** Wynik optymalizacji dla ekstremalnie gęstych scen. Pozwala na zachowanie wysokiej 
rozdzielczości punktów, co zapobiega zlewaniu się estymowanej gęstości.

### 2.4 Wnioski z Augmentacji
Zastosowanie autorskiego potoku przetwarzania danych (512x512 crops, flips, density multiplier) miało kluczowy 
wpływ na wynik:

* **Mnożnik wiedzy:** Poprzez generowanie 5 losowych cropów z każdego zdjęcia, sztucznie zwiększono zbiór 
treningowy, co zapobiegło przeuczeniu (**overfitting**) na małym zbiorze ShanghaiTech.

* **Selekcja danych:** Mechanizm odrzucania fragmentów bez tłumu (sum < 0.05) sprawił, że model stał się specjalistą 
od detekcji głów, nie tracąc uwagi na nieistotne elementy tła.

* **Stabilność obliczeniowa:** Zastosowanie mnożnika gęstości (x1000) pozwoliło na uzyskanie stabilnego gradientu, co 
bezpośrednio przełożyło się na szybkość i jakość konwergencji modelu.

---

## 3. Dokumentacja Techniczna
### Architektura
* **Model:** FCN (Fully Convolutional Network) oparty na VGG16 (backbone).

* **Backend:** 6 warstw splotowych z batch normalization oraz dylatacją (dilation rate 2 i 4), co pozwala na 
analizę kontekstu przy zachowaniu wysokiej rozdzielczości mapy wyjściowej.

* **Input Shape:** `(None, None, 3)` – model jest uniwersalny i przyjmuje obrazy o dowolnych 
wymiarach (zalecana wielokrotność 16).

### Metryki i Ewaluacja
Zamiast macierzy pomyłek (niewłaściwej dla regresji), zastosowano:
**MAE (Mean Absolute Error):** Średnia różnica między predykcją a prawdą.

**Weryfikacja empiryczna:**

Poniżej pokazano przykładowe zdjęcia tłumu zbitego oraz tłumu bardziej rozrzuconego z strony [unsplash.com](https://unsplash.com/photos/a-large-crowd-of-people-in-a-tennis-court-mhifnbM-leE), wyszukując frazę "a large crowd of people":

Przykład ewaluacji na wybranym zdjęciu tłumu zbitego(part_A) przy użyciu model_for_part_A.keras:
![img_2.png](scripts/img_2.png)

Przykład ewaluacji na wybranym zdjęciu tłumu rozrzuconego(part_B) przy użyciu model_for_part_B.keras:
![img_1.png](scripts/img_1.png)

---

## 4. API REST
Projekt zawiera implementację API (FastAPI), która umożliwia odpytywanie modelu przez sieć:
* **Endpoint:** `POST /predict`
* **Funkcjonalność:** Przyjmuje obraz, wykonuje preprocessing (normalizacja ImageNet) i zwraca liczbę osób w formacie JSON.

---

## 5. Jak skonfigurować projekt
### Pobieranie wag modelu:
Ze względu na duży rozmiar plików (ok. 200MB każdy), wagi modelu są przechowywane zewnętrznie.
1. Pobierz pliki `.keras` z folderu: [Google Drive - Crowd Counting Models](https://drive.google.com/drive/folders/18sNE3c7HuuoI4mOSdWD0SE8KPD7_dWBj?usp=sharing)
2. Umieść pobrane pliki w katalogu: `src/api/models/`

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

---

## 6. Jak uruchomić aplikacje
Przejdź do katalogu z API i uruchom serwer uvicorn:
```bash
cd src/api
```
```bash
uvicorn app:app --reload
```
Po wykonaniu tych kroków aplikacja będzie dostępna w przeglądarce pod adresem: http://127.0.0.1:8000