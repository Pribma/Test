import torch

# ==============================================================================
# 1. PŘÍPRAVA PŘED BITVOU (Tohle dáš TĚSNĚ PŘED svůj hlavní for cyklus)
# ==============================================================================
best_val_loss = float('inf')

# Cesta k trezoru - ZMĚŇ SI NÁZEV SLOŽKY PODLE REALITY!
cesta_k_trezoru = "/content/drive/MyDrive/MojeSoutezniSlozka/best_model.pth" 

try:
    # Tvůj hlavní cyklus začíná zde
    for epoch in range(1, epochs + 1):
        
        # ----------------------------------------------------------------------
        # TADY BUDE TVŮJ KÓD PRO TRÉNINK (model.train(), zero_grad, backward...)
        # ----------------------------------------------------------------------
        
        # ----------------------------------------------------------------------
        # TADY BUDE TVŮJ KÓD PRO VALIDACI (model.eval(), with torch.no_grad()...)
        # (Na konci této části ti musí vzniknout proměnná val_loss)
        # ----------------------------------------------------------------------

        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

        # ======================================================================
        # 2. KONTROLA REKORDU (Tohle dáš na ÚPLNÝ KONEC těla for cyklu)
        # ======================================================================
        if val_loss < best_val_loss:
            print(f"--> Zlepšení z {best_val_loss:.4f} na {val_loss:.4f}. Ukládám mozek na Disk!")
            best_val_loss = val_loss
            # Uloží to bezpečně k tobě na Google Disk
            torch.save(model.state_dict(), cesta_k_trezoru)

# Tohle tě zachrání, když trénink v panice zastavíš ručně tlačítkem Stop!
except KeyboardInterrupt:
    print("\n[!] Trénink přerušen velitelem! Jdeme zachránit nejlepší verzi.")

print("Výcviková fáze ukončena.")

# ==============================================================================
# 3. NAHRÁNÍ ZÁLOHY (Tohle dáš ÚPLNĚ MIMO for cyklus, až na samotný konec buňky)
# ==============================================================================
print("Nahrávám do sítě nejlepší zálohu z Disku...")
model.load_state_dict(torch.load(cesta_k_trezoru))

# Okamžitě zajistíš zbraň, aby byla připravená na generování submission.csv
model.eval() 
print("Model je připraven k ostré palbě!")







# ==========================================
# 4. HLAVNÍ TRÉNINKOVÁ SMYČKA (Nyní s validací a ukládáním!)
# ==========================================
def train_model(model, train_loader, val_loader, epochs=5): # PŘIDÁN val_loader!
    criterion = nn.CrossEntropyLoss() # Cíl: Kategorie
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 1. PŘÍPRAVA TREZORU
    best_val_loss = float('inf')
    # TADY SI ZÍTRA ZMĚŇ CESTU PODLE SVÉHO DISKU!
    cesta_k_trezoru = "/content/best_model.pth" 

    try:
        for epoch in range(1, epochs + 1):
            # --- A) BITEVNÍ DRIL (TRÉNINK) ---
            model.train() 
            total_train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)

            # --- B) PŘEPADOVÁ INSPEKCE (VALIDACE) ---
            model.eval() # Zastavit učení!
            total_val_loss = 0
            
            with torch.no_grad(): # Vypnout autograd!
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Hlášení veliteli
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # --- C) KONTROLA REKORDU A ULOŽENÍ ---
            if avg_val_loss < best_val_loss:
                print(f"   [!] Nový rekord! Ukládám mozek na disk: {avg_val_loss:.4f}")
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), cesta_k_trezoru)

    except KeyboardInterrupt:
        print("\n[!] Trénink přerušen velitelem (Early Stopping)!")

    # --- D) NÁVRAT DO NEJLEPŠÍHO STAVU PŘED OSTRÝM TESTEM ---
    print("\nNahrávám do zbraně nejlepší uloženou zálohu...")
    model.load_state_dict(torch.load(cesta_k_trezoru))
    return model # Vracíme vylepšeného vojáka!

# ==========================================
# SPUŠTĚNÍ
# ==========================================
if __name__ == "__main__":
    # 1. Nahraj data (ZÍTRA ZDE MUSÍŠ MÍT 3 DATASETY!)
    train_ds = UniversalDataset("cesta/k/train", is_test=False)
    val_ds = UniversalDataset("cesta/k/val", is_test=False) # NOVÉ!
    test_ds = UniversalDataset("cesta/k/test", is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False) # NOVÉ! Nelosovat (shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 2. Vytvoř zbraň
    model = UniversalModel().to(device)
    
    # 3. Trénink
    print("Začínám výcvik...")
    # Funkce nám vrátí nejlepší možnou verzi modelu
    best_model = train_model(model, train_loader, val_loader, epochs=50) 
    
    # 4. Generování výsledků
    print("Jdu do ostré akce...")
    # Do ostré mise už posíláme best_model!
    generate_submission(best_model, test_loader)
