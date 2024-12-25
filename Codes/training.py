import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import ModuloDataGenerator
from transformer_pe import SimpleTransformer
from MLP import CustomMLP
import visualize as vis


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        #inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)  # shape (batch_size, output_dim)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy
        with torch.no_grad():
            predicted = torch.argmax(outputs, dim=1)
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
        #inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion,
          num_epochs: int,
          device) -> tuple[list]:
    """
    Train the model for num_epochs, and return the train/validation loss/accuracy
    """
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        train_loss_, train_accuracy_ = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss_, val_accuracy_ = validate(model, val_loader, criterion, device)

        train_loss.append(train_loss_)
        train_acc.append(train_accuracy_)
        val_loss.append(val_loss_)
        val_acc.append(val_accuracy_)

    return train_loss, train_acc, val_loss, val_acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Problem setup
p = 97
num_summands = 2

# Model hyper-parameters
input_dim = p + 2
d_model = 128
n_heads = 4
n_layers = 2
output_dim = p + 2
seq_length = 2 * num_summands

model = SimpleTransformer(input_dim, d_model, n_heads, n_layers, output_dim).to(device)
#model = CustomMLP(input_dim, [d_model * n_heads] * n_layers, output_dim, seq_length).to(device)

# Data
num_epochs = 1000
batch_size = 256

data_generator = ModuloDataGenerator(p)
train_loader, val_loader = data_generator.get_dataloader(alpha=0.4, batch_size=batch_size)

# Training
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

result = train(model, train_loader, val_loader,
              optimizer, criterion, num_epochs, device)
train_loss, train_acc, val_loss, val_acc = result

# plot and save the accuracy
vis.plot_acc(train_acc, val_acc)
vis.save_fig("transformer_test.pdf")
