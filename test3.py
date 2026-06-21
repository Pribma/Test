import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import matplotlib.pyplot as plt

BASE_PATH = r"C:\Users\Creator\Desktop\radar_data"

class RadarDataset(Dataset):
    def __init__(self, folder: str, is_test: bool = False):
        self.folder = Path(folder)
        self.files = sorted(list(self.folder.glob("*.mat.pt")))
        self.is_test = is_test
        if len(self.files) == 0:
            raise RuntimeError(f"Zádné soubory nenalezeny v {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        tensor = torch.load(file_path, map_location="cpu")
        x = tensor[:6].float()
        if self.is_test:
            return x, file_path.name
        y = tensor[6].long() + 1 
        return x, y


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 5, kernel_size=3, padding=1) 
        )

    def forward(self, x):
        return self.net(x)


def ioai_score(preds, labels):
    
    bg_mask = (labels == 0)
    obj_mask = (labels > 0)
    
    correct_bg = ((preds == labels) & bg_mask).sum().item()
    correct_obj = ((preds == labels) & obj_mask).sum().item()
    
    total_bg = bg_mask.sum().item()
    total_obj = obj_mask.sum().item()
    
    num = correct_bg * 1 + correct_obj * 50
    den = total_bg * 1 + total_obj * 50
    return num / den if den > 0 else 0.0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zařízení: {device}")


    train_ds = RadarDataset(os.path.join(BASE_PATH, "training_set"))
    val_ds = RadarDataset(os.path.join(BASE_PATH, "validation_set"))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")


        import csv

def generate_submission(model, loader, device, output_csv):
    model.eval()
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
    
        header = ["filename"] + [f"pixel_{i}" for i in range(50 * 181)]
        writer.writerow(header)

        with torch.no_grad():
            for x, filenames in loader:
                x = x.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1) 
                
                
                preds = (preds - 1).cpu().numpy()
                
                for i, filename in enumerate(filenames):
                    row = [filename] + preds[i].flatten().tolist()
                    writer.writerow(row)
    print(f"Soubor {output_csv} byl vytvořen.")

val_ds_submission = RadarDataset(os.path.join(BASE_PATH, "validation_set"), is_test=True)
val_loader_sub = DataLoader(val_ds_submission, batch_size=1, shuffle=False)

test_ds_submission = RadarDataset(os.path.join(BASE_PATH, "test_set"), is_test=True)
test_loader_sub = DataLoader(test_ds_submission, batch_size=1, shuffle=False)

generate_submission(model, val_loader_sub, device, "submission_val.csv")
generate_submission(model, test_loader_sub, device, "submission_test.csv")

print("Nyní oba soubory (.csv) zabalte do submission.zip a můžete odevzdat!")