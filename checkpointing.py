# 1. Před začátkem tréninku nastavíš laťku na nekonečno
best_val_loss = float('inf') 

for epoch in range(1, epochs + 1):
    # ... TADY PROBÍHÁ TRÉNINK (model.train) ...
    # ... TADY PROBÍHÁ VALIDACE (model.eval) a spočítá se val_loss ...
    
    print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

    # 2. Kontrola rekordu
    if val_loss < best_val_loss:
        print("Nový rekord! Ukládám stav mysli vojáka...")
        best_val_loss = val_loss
        
        # 3. Uložení mozku na disk
        torch.save(model.state_dict(), "best_model.pth")

print("Výcvik ukončen.")

# 4. Nahrání nejlepší verze před ostrou misí
print("Nahrávám zálohu s nejlepším výkonem...")
model.load_state_dict(torch.load("best_model.pth"))

# Nyní je model ve stavu, v jakém byl v době nejlepší epochy, 
# a můžeš s ním generovat submission.csv!