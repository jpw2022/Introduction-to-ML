import torch
import torch.nn as nn
import torch.optim as optim

from data import ModuloDataGenerator
from transformer import SimpleTransformer
from MLP import CustomMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Problem setup
p = 13
num_summands = 2

# Transformer hyper-parameters
input_dim = p + 2
d_model = 128
n_heads = 8
n_layers = 4
output_dim = p + 2
seq_length = 2 * num_summands

#model = SimpleTransformer(input_dim, d_model, n_heads, n_layers, output_dim).to(device)
model = CustomMLP(input_dim, [d_model * n_heads] * n_layers, output_dim,
                  seq_length).to(device)

# Data
batch_size = 4

data_generator = ModuloDataGenerator(p)
train_loader, val_loader = data_generator.get_dataloader(alpha=0.8, batch_size=batch_size)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)  # shape (batch_size, output_dim)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

num_epochs = 1000
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

