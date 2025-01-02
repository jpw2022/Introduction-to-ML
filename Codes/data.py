import torch
import torch.nn.functional as F
from torch.utils.data import (
        TensorDataset,
        DataLoader,
        random_split,
)

from torch import Tensor

def MODULO_ADD(*args: Tensor, mod: int=-1) -> Tensor:
    if mod < 0:
        raise ValueError("Invalid modulo number in MODULO_ADD")
    return torch.stack(args, dim=0).sum(dim=0) % mod

OPERATION_DICT = {
        "+": MODULO_ADD,
        }

class ModuloDataGenerator:
    def __init__(self,
                 modulo_number: int,
                 operation_str: str='+'):
        self.op_str = operation_str
        self.operation = OPERATION_DICT[operation_str]
        self.p = modulo_number

    def generate_data(self, num_summands: int=2,
                      num_samples: int=0) -> tuple[Tensor]:
        """
        Generate inputs as one-hot vectors, labels are not one-hot encoded
        'op' and 'eq' are treated the same as numbers

        returns shape:
        data: (num_samples, 2*num_summands, p+2), labels: (num_samples)
        if num_samples is not specified, it is by default p ** num_summands
        """
        # generate all p * p x-y pairs
        x = torch.arange(0, self.p).repeat(num_summands, 1)
        # x has shape (num_summands, p)
        if num_samples:
            idx = torch.randint(0, self.p ** num_summands, size=(num_samples,))
            components = [(idx / self.p**k).long() % self.p for k in range(num_summands)]
            x_combination = torch.stack(components)
        else:
            x_combination = torch.cartesian_prod(*x).T
        # shape (num_summands, p^num_summands), this may cause memory issue

        # compute the label of 'x op y'
        # use p and p+1 to represent 'op' and 'eq'
        labels = self.operation(*x_combination, mod=self.p)
        op = torch.zeros_like(x_combination[0]) + self.p
        eq = torch.zeros_like(x_combination[0]) + self.p + 1
        expression = [x_combination[k//2] if k % 2 == 0 else op
                      for k in range(2 * num_summands - 1)]
        expression.append(eq)
        data = torch.stack(expression, dim=1)

        # label them as one-hot vectors
        num_classes = self.p + 2
        data = F.one_hot(data, num_classes=num_classes)

        # cast data to float type
        data = data.float()

        return data, labels

    def get_dataloader(self, alpha: float,
                       batch_size: int=1,
                       num_summands: int=2,
                       num_samples: int=0) -> tuple[DataLoader]:
        """
        returns train loader and validation loader with total length num_samples,
        split train and validation according to the ratio of alpha
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")
        data, labels = self.generate_data(num_summands, num_samples)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data, labels = data.to(device), labels.to(device)
        dataset = TensorDataset(data, labels)

        train_length = int(alpha * len(dataset))
        val_length = len(dataset) - train_length
        train_set, val_set = random_split(dataset, [train_length, val_length])

        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size, shuffle=True)

        return train_loader, val_loader

if __name__ == '__main__':
    test_generator = ModuloDataGenerator(7)
    train_loader, val_loader = test_generator.get_dataloader(0.9, 1, 4)
    print("Training/validation size", len(train_loader), len(val_loader))
    print("The first training sample is:")
    for sample in train_loader:
        print(sample)
        break

