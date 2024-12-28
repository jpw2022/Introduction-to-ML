import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import ModuloDataGenerator
from transformer_pe import SimpleTransformer
from MLP import CustomMLP
from LSTM import LSTMClassifier


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

def train_by_args(args):
    """
    Training using arguments from command line
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Problem setup
    p = args.p
    num_summands = args.k

    # Model hyper-parameters
    input_dim = p + 2
    d_model = args.d_model
    n_heads = args.n_heads
    n_layers = args.n_layers
    output_dim = p + 2
    seq_length = 2 * num_summands

    if args.model == "transformer":
        model = SimpleTransformer(input_dim, d_model, n_heads,
                                  n_layers, output_dim).to(device)
    elif args.model == "MLP":
        model = CustomMLP(input_dim, [d_model * 4] * n_layers,
                          output_dim, seq_length).to(device)
    elif args.model == "LSTM":
        model = LSTMClassifier(input_dim, d_model * 4, n_layers,
                               output_dim).to(device)
    else:
        raise ValueError(f"""Unkown model type: {args.model}, only supports
                         transformer, MLP, LSTM.""")

    # Data
    num_epochs = args.n_epochs
    batch_size = args.batch_size
    alpha = args.alpha
    n_samples = args.n_samples

    data_generator = ModuloDataGenerator(p)
    train_loader, val_loader = data_generator.get_dataloader(alpha,
                                                             batch_size,
                                                             num_summands,
                                                             n_samples)

    # Training
    lr = args.lr
    criterion = nn.CrossEntropyLoss().to(device)
    optim_str = args.optimizer
    if optim_str == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98),
                                weight_decay=0.05)
    elif optim_str == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif optim_str == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optim_str == 'SGD_nesternov':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optim_str}")

    result = train(model, train_loader, val_loader,
                  optimizer, criterion, num_epochs, device)
    return result

