import torch.nn as nn

"""
Dziedziczymy po nn.Module w klasie LoRALinear ponieważ chcemy móc trenować tą 
warstwę tak jak każdą warstwę PyTorch
"""

class LoRALinear(nn.Module):

    """
    base_layer -> orginalna warstwa, do której doklejamy LoRA i której zachowanie adaptujemy
    r ----------> mówi, w ilu niezależnych kierunkach LoRA może zmienić zachowanie warstwy. Im większe
                  r, tym większa pojemność adaptacji, ale większy koszt i ryzyko overfittingu.
    alpha ------> kontroluje, jak mocno LoRA może zmienić wyjście warstwy liniowej. Większe alpha = większy
                  wpływ LoRA na zachowanie warstwy, ale większe ryzyko niestabilności treningu.
    dropout ----> regularizacja LoRA. Określa prawdopodobieństwo, z jakim wejścia do ścieżki LoRA
                  są losowo „wyłączane” podczas treningu. Wyższy dropout zmniejsza ryzyko overfittingu,
                  zmuszając LoRA do uczenia się bardziej uogólnionych korekt wyjścia warstwy.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
    ):

        """

        inicjalizujemy  nn.Module
        """
        super().__init__()

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        """
        tutaj scaling mówi o tym jak mocno sygnał LoRA (ΔW) jest dodawany do wyjścia warstwy bazowej
        
        formalnie scaling jest mnożony przez x(B*A)^T, działa podobnie jak „volume knob” w głośnikach
        """
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        """
        Wymiary warstwy, LoRA musi je znać by zagwarantować kompatybilność
        """
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        """
        lora_A -> macierz, która wybiera kierunki, w których LoRA może zmienić zachowanie warstwy. 
                  Na starcie wartości są losowe w rozsądnym zakresie, żeby LoRA mogła się uczyć, ale 
                  gradienty były stabilne.  

        lora_B -> macierz, która mówi, jak wybrane kierunki wpływają na wyjście warstwy. 
                  Na starcie wszystkie wartości są zerowe, dzięki czemu LoRA nie zmienia outputu 
                  warstwy bazowej od razu, tylko zaczyna dodawać korekty stopniowo w trakcie treningu.
        
        póki co jedynie je tutaj inicjalizujemy nie wypełniamy niczym!
        """
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        """
        Inicjalizacja wag:

        - nn.init.kaiming_uniform_(self.lora_A.weight)  
          Wartości w macierzy lora_A są losowo ustawione w rozsądnym zakresie. Dzięki temu LoRA ma sensowne
          kierunki, w których może wprowadzać zmiany, a gradienty są stabilne od samego początku treningu.

        - nn.init.zeros_(self.lora_B.weight)  
          Wartości w macierzy lora_B są ustawione na zero. Dzięki temu LoRA na starcie **nie zmienia
          wyjścia warstwy bazowej**, a korekty zaczynają się stopniowo w trakcie treningu.
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        """
        forward(x) -> oblicza wynik warstwy LoRA

        - base_out = self.base_layer(x)
          Wyjście z oryginalnej, zamrożonej warstwy. To jest „nauczyciel”, którego nie zmieniamy.

        - lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
          Ścieżka LoRA:
            1. self.dropout(x) -> losowo wyłącza część wejść, żeby LoRA się nie przeuczyła
            2. self.lora_A(...) -> wybiera kierunki, w których LoRA może wprowadzać korekty
            3. self.lora_B(...) -> decyduje, jak te kierunki wpływają na wyjście warstwy
            4. * self.scaling -> kontroluje siłę wpływu LoRA na wynik warstwy

        - return base_out + lora_out
          Wynik końcowy = output warstwy bazowej + korekta LoRA. Dzięki temu LoRA **nie zastępuje** 
          warstwy bazowej, tylko delikatnie zmienia jej zachowanie.
        """
    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out