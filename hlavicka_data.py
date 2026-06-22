!head -n 2 vzorove_odevzdani.csv

import pandas as pd
tabulka = pd.read_csv("vzorove_odevzdani.csv")
print(tabulka.head(2))


import csv
import torch

# Předpokládejme, že víš, kolik pixelů přesně odevzdáváš (např. 9000)
POCET_PIXELU = 9000

with open("submission_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    
    # --- 1. FÁZE: HLAVIČKA ---
    # Vytvoří: ['id', 'pixel_0', 'pixel_1', ..., 'pixel_8999']
    hlavicka = ["id"] + [f"pixel_{i}" for i in range(POCET_PIXELU)]
    writer.writerow(hlavicka)
    
    # --- 2. FÁZE: OSTRÁ PALBA ---
    # Tohle se děje uvnitř tvé smyčky, kde procházíš testovací data!
    # for batch_x, batch_id in test_loader:
    #     with torch.no_grad():
    #         predikce_tensor = model(batch_x)
            
            # Dejme tomu, že řešíme jeden konkrétní záznam:
            # id_zaznamu = batch_id[0].item()
            # jeden_vysledek = predikce_tensor[0] # Tohle je Tensor o 9000 číslech
            
            # A TADY JE TO KOUZLO:
            # 1. Převedeš 9000 čísel z Tensoru na normální Python list
            predikce_list = jeden_vysledek.tolist()
            
            # 2. Spojíš ID a těch 9000 čísel do jedné dlouhé fronty
            radek_k_zapisu = [id_zaznamu] + predikce_list
            
            # 3. Odpálíš to do souboru
            writer.writerow(radek_k_zapisu)
