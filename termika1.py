import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# --- KONFIGURACE CEST ---
BASE_PATH = r'C:\Users\Creator\Desktop\termika_data'

def get_path(filename):
    return os.path.join(BASE_PATH, filename)

# --- KROK 1: NAČTENÍ DAT ---
def load_data():
    train_df = pd.read_csv(get_path('train_measurements.csv'))
    boundary_df = pd.read_csv(get_path('boundary_partial.csv'))
    initial_df = pd.read_csv(get_path('initial_sparse.csv'))
    test_df = pd.read_csv(get_path('test_points.csv'))
    
    # Přidáno encoding='utf-8' pro opravu UnicodeDecodeError na Windows
    with open(get_path('constants.json'), 'r', encoding='utf-8') as f:
        constants = json.load(f)
        alpha = constants.get('alpha', 0.01)

    def to_t(df_col, grad=False):
        return torch.tensor(df_col.values, dtype=torch.float32).view(-1, 1).requires_grad_(grad)

    return {
        'train': (to_t(train_df['x']), to_t(train_df['t']), to_t(train_df['u'])),
        'bc': (to_t(boundary_df['x']), to_t(boundary_df['t']), to_t(boundary_df['u'])),
        'ic': (to_t(initial_df['x']), to_t(initial_df['t']), to_t(initial_df['u'])),
        'test_x_t': (to_t(test_df['x']), to_t(test_df['t'])),
        'test_df': test_df,
        'alpha': alpha
    }

# --- KROK 2: ARCHITEKTURA PINN ---
class HeatNet(nn.Module):
    def __init__(self, hidden_size=64, num_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        return self.net(X)

# --- KROK 3: FYZIKÁLNÍ ZTRÁTA ---
def loss_physics(model, x, t, alpha):
    u = model(x, t)
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    return torch.mean((du_dt - alpha * d2u_dx2)**2)

# --- KROK 4: TRÉNINK ---
def train():
    data = load_data()
    alpha = data['alpha']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HeatNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    x_train, t_train, u_train = [t.to(device) for t in data['train']]
    x_bc, t_bc, u_bc = [t.to(device) for t in data['bc']]
    x_ic, t_ic, u_ic = [t.to(device) for t in data['ic']]

    for epoch in range(5001):
        optimizer.zero_grad()
        
        # Loss z měření
        l_data = torch.mean((model(x_train, t_train) - u_train)**2)
        l_bc = torch.mean((model(x_bc, t_bc) - u_bc)**2)
        l_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
        
        # Physics loss (kolokační body)
        x_col = torch.rand(1000, 1, device=device).requires_grad_(True)
        t_col = torch.rand(1000, 1, device=device).requires_grad_(True)
        l_phys = loss_physics(model, x_col, t_col, alpha)
        
        loss = l_data + l_bc + l_ic + 0.1 * l_phys
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f} (Phys: {l_phys.item():.6f})")

    # Generování submission
    model.eval()
    with torch.no_grad():
        x_test, t_test = [t.to(device) for t in data['test_x_t']]
        u_pred = model(x_test, t_test).cpu().numpy().flatten()
        submission = pd.DataFrame({'id': data['test_df']['id'], 'u': u_pred})
        submission.to_csv(get_path('submission.csv'), index=False)
        print(f"Hotovo! Soubor uložen v {get_path('submission.csv')}")

if __name__ == "__main__":
    train()