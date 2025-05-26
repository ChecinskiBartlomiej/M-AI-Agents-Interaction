import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import data_generator

# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

# Read data.
data = np.loadtxt('/home/bc/Desktop/Documents/muchizm/Symbolic_regression/data.csv', delimiter=',') 
X = data[:,:3]
Y = data[:,3]

# Create new features.
xx = X[:,0]**2
yy = X[:,1]**2
zz = X[:,2]**2
xy = X[:,0] * X[:,1]
xz = X[:,0] * X[:,2]
yz = X[:,1] * X[:,2]
x = X[:,0]
y = X[:,1]
z = X[:,2]

X_transformed = np.stack([xx, yy, zz, xy, xz, yz, x, y, z], axis=1)

# Convert data to tensors.
tensor_X_transformed = torch.from_numpy(X_transformed).float().to(device)  
tensor_Y = torch.from_numpy(Y).float().to(device)
dataset = TensorDataset(tensor_X_transformed, tensor_Y)

# Split data.
total = len(dataset)
train_size = int(0.8 * len(dataset))
test_size = total - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create dataloaders.
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create neual network.
class PolyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(9, device=device))
        self.b = nn.Parameter(torch.randn(1, device=device))
    
    def forward(self, X):
        return (X @ self.w) + self.b

model = PolyNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 1000

# Create train function.
def train():
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 
            running_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss: {running_loss:.6f}")

def evaluate():
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():  
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() 
    return total_loss

train()

# Retrive parameters.
coefs = model.w.detach().cpu().numpy()    
bias  = model.b.item()                    

# Print found polynomial.
monomials = [f"{c:.6f}*{name}" for name, c in zip(data_generator.feature_names, coefs)]
polynomial = " + ".join(monomials) + f" + {bias:.6f}"
print("Found polynomial:")
print(f"y = {polynomial}")

# Print real polynomial.
print("Real polynomial:")
print("y =", data_generator.formula)

# Evaluate on test set.
print("Loss on test set: ", evaluate())