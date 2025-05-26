import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Indykator trójkąta.
def TriangleIndicator(X):
    return ((X[:,0] >= 0) & (X[:,1] >= 0) & (X[:,0] + X[:,1] <= 1)).float()

# Generowanie danych treningowych.
N = 1000
X = torch.rand(N, 2)
y = TriangleIndicator(X).unsqueeze(1)

# Funkcja treningowa.
def train(model, X, y, epochs, lr, verbose=True):
    criterion = nn.MSELoss()
    #optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    LossHistory = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        LossHistory.append(loss.item())
        if verbose and epoch % 1000 == 0:
            print(f"Epoka {epoch}, błąd: {loss.item():.6f}")
        if loss.item() < 1e-3:
            print(f"Osiągnięto zbieżność w epoce {epoch}, błąd: {loss.item():.6f}")
            break
    return LossHistory

# Sieć jednowarstwowa.
class OneLayerNN(nn.Module):
    def __init__(self, activation):
        super(OneLayerNN, self).__init__()
        self.fc = nn.Linear(2, 1, bias=True)
        self.activation = activation

    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        return out

# Sieć dwuwarstwowa.
class TwoLayerNN(nn.Module):
    def __init__(self, hidden_size, activation):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size, bias=True)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        #out = self.activation(out)
        return out

# Sieć trzywarstwowa.
class ThreeLayerNN(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, activation):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size1, bias=True)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, bias=True)
        self.fc3 = nn.Linear(hidden_size2, 1, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        #out = self.activation(out)
        return out

# Lista funkcji aktywacji do testowania.
ActivationFunctions = [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]

# Przygotowujemy strukturę zawierającą wszystkie architektury.
architectures = {
    "OneLayer": {
        "constructor": lambda act: OneLayerNN(activation=act) # 3 parametry
    },
    "TwoLayer": {
        "constructor": lambda act: TwoLayerNN(hidden_size=4, activation=act) # 17 parametrów
    },
    "ThreeLayer": {
        "constructor": lambda act: ThreeLayerNN(hidden_size1=2, hidden_size2=2, activation=act) # 15 parametrów
    }
}

# Słowniki do przechowywania wyników.
LossHistories = {}
Models = {}

# Pętla treningowa dla wszystkich architektur i funkcji aktywacji.
for arch_name, arch_info in architectures.items():
    LossHistories[arch_name] = {}
    Models[arch_name] = {}
    print(f"\nTrening architektury: {arch_name}")
    for act in ActivationFunctions:
        act_name = act.__class__.__name__
        print(f"\nFunkcja aktywacji: {act_name}")
        model = arch_info["constructor"](act)
        loss_history = train(model, X, y, epochs=15000, lr=0.01)
        LossHistories[arch_name][act_name] = loss_history
        Models[arch_name][act_name] = model

# Wizualizacja historii błędów.
def plot_loss_histories(loss_histories, arch_name):
    plt.figure(figsize=(8, 6))
    for act_name, history in loss_histories.items():
        plt.plot(history, label=act_name)
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE")
    plt.title(f"Historia błędu L2 - {arch_name}")
    plt.legend()
    plt.show()

for arch_name in LossHistories:
    plot_loss_histories(LossHistories[arch_name], arch_name)

# Testowanie modeli na nowych danych.
N_test = 1000
X_test = torch.rand(N_test, 2)
y_test = TriangleIndicator(X_test).unsqueeze(1)
criterion = nn.MSELoss()

def test_models(models, arch_name):
    print(f"\nTestowanie modeli - {arch_name}:")
    for act_name, model in models.items():
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            test_loss = criterion(predictions, y_test)
        print(f"Model {act_name} - testowy błąd MSE: {test_loss.item():.6f}")

for arch_name in Models:
    test_models(Models[arch_name], arch_name)

# Wizualizacja.
grid_points = 100
xx, yy = np.meshgrid(np.linspace(0, 1, grid_points), np.linspace(0, 1, grid_points))
grid = np.column_stack([xx.ravel(), yy.ravel()])
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Prawdziwa funkcja indykatorowa trójkąta.
true_values = TriangleIndicator(grid_tensor).reshape(xx.shape)
plt.figure(figsize=(8, 6))
cp_true = plt.contourf(xx, yy, true_values, levels=50, cmap="viridis")
plt.colorbar(cp_true)
plt.title("Prawdziwa funkcja indykatorowa trójkąta")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Nasze aproksymacje.
def plot_decision(models, title_prefix):
    plt.figure(figsize=(12, 8))
    for i, (act_name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            predictions = model(grid_tensor)
        predictions = predictions.reshape(xx.shape)
        plt.subplot(2, 2, i + 1)
        cp = plt.contourf(xx, yy, predictions, levels=50, cmap="viridis")
        plt.colorbar(cp)
        plt.title(f"{title_prefix} - {act_name}")
    plt.tight_layout()
    plt.show()

for arch_name in Models:
    plot_decision(Models[arch_name], arch_name)