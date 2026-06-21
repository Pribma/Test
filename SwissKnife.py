import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import csv

# ==========================================
# 1. ZÁKLADNÍ NASTAVENÍ (Nesahej na to, jen to spusť)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Bojujeme na: {device}")

# Nastavení seedu - abys měl při každém spuštění stejné výsledky a nelítalo ti to náhodně
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 2. LOGISTIKA DAT (Tohle zítra budeš muset upravit podle zadání)
# ==========================================
class UniversalDataset(Dataset):
    def __init__(self, file_path, is_test=False):
        self.is_test = is_test
        # TODO ZÍTRA: Tady načteš data (pd.read_csv, os.listdir, atd.)
        # Příklad pro CSV:
        # self.data = pd.read_csv(file_path)
        pass 

    def __len__(self):
        # TODO ZÍTRA: Vrať celkový počet řádků/obrázků
        return 100 # Příklad

    def __getitem__(self, idx):
        # TODO ZÍTRA: Vytáhni 1 řádek/obrázek a převeď ho na Tensor
        # x = torch.tensor(vstupy, dtype=torch.float32)
        # Pokud self.is_test == False: y = torch.tensor(cil, dtype=torch.long)
        
        # Příklad dummy dat, aby to teď nespadlo:
        x = torch.rand(10) # 10 náhodných čísel na vstupu
        if self.is_test:
            return x, idx # Vracím id pro submission
        y = torch.tensor(1, dtype=torch.long) # Cílová třída
        return x, y

# ==========================================
# 3. MOZEK / MODEL (Tohle zítra ukradneš z GitHubu nebo zadání)
# ==========================================
class UniversalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Zítra to bude buď MLP (nn.Linear), CNN (nn.Conv2d), nebo něco z torchvision
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Výstup: např. 2 třídy
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. HLAVNÍ TRÉNINKOVÁ SMYČKA (Zlatý grál - toto se NIKDY nemění)
# ==========================================
def train_model(model, train_loader, epochs=5):
    # Pro klasifikaci (třídy): nn.CrossEntropyLoss()
    # Pro regresi (čísla/teplota): nn.MSELoss()
    criterion = nn.CrossEntropyLoss() # TODO ZÍTRA: Zkontroluj, co po tobě chtějí!
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train() # Přepnutí do trénovacího módu
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # KROK 1: Vyčisti hlaveň
            optimizer.zero_grad()
            
            # KROK 2: Vystřel (Predikce)
            outputs = model(batch_x)
            
            # KROK 3: Spočítej, jak moc jsi minul
            loss = criterion(outputs, batch_y)
            
            # KROK 4: Zpětný ráz (Gradienty)
            loss.backward()
            
            # KROK 5: Uprav mířidla (Změna vah)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# ==========================================
# 5. ODEVZDÁNÍ / SUBMISSION (Tohle tě dělí od 0 bodů)
# ==========================================
def generate_submission(model, test_loader, filename="submission.csv"):
    model.eval() # ZASTAVÍ UČENÍ! Kritické!
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prediction"]) # TODO ZÍTRA: Koukni se, jak se mají jmenovat sloupce!
        
        with torch.no_grad(): # VYPNE SLEDOVÁNÍ PAMĚTI! Kritické!
            for batch_x, ids in test_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                
                # Zítra: U klasifikace většinou děláš argmax (nejvyšší pravděpodobnost)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Zápis do CSV
                for i, p in zip(ids, preds):
                    writer.writerow([i.item(), p])
    print(f"Mise splněna. Uloženo do {filename}")

# ==========================================
# SPUŠTĚNÍ
# ==========================================
if __name__ == "__main__":
    # 1. Nahraj data
    train_ds = UniversalDataset("cesta/k/train", is_test=False)
    test_ds = UniversalDataset("cesta/k/test", is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 2. Vytvoř zbraň
    model = UniversalModel().to(device)
    
    # 3. Trénink
    print("Začínám výcvik...")
    train_model(model, train_loader, epochs=5)
    
    # 4. Generování výsledků
    print("Jdu do ostré akce...")
    generate_submission(model, test_loader)