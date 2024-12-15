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

    # TODO: write general code for num_summands > 2
    def generate_data(self, num_summands: int=2) -> Tensor:
        """
        Generate inputs as one-hot vectors, labels are not one-hot encoded
        'op' and 'eq' are treated the same as numbers
        returns shape:
        data: (p^2, 4, p+2), labels: (p^2)
        """
        # generate all p * p x-y pairs
        x = torch.arange(0, self.p)
        y = torch.arange(0, self.p)
        x, y = torch.cartesian_prod(x, y).T

        # compute the label of 'x op y'
        # use p and p+1 to represent 'op' and 'eq'
        labels = self.operation(x, y, mod=self.p)
        op = torch.zeros_like(x) + self.p
        eq = torch.zeros_like(x) + self.p + 1
        data = torch.stack([x, op, y, eq], dim=1)

        # label them as one-hot vectors
        num_classes = self.p + 2
        data = F.one_hot(data, num_classes=num_classes)

        # cast data to float type
        data = data.float()

        return data, labels

    def get_dataloader(self, alpha: float,
                       batch_size: int=1,
                       num_summands: int=2):
        """
        returns train loader and validation loader,
        split train and validation according to the ratio of alpha
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")
        data, labels = self.generate_data(num_summands)
        dataset = TensorDataset(data, labels)

        train_size = int(alpha * len(dataset))
        validation_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, validation_size])

        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size, shuffle=True)

        return train_loader, val_loader

if __name__ == '__main__':
    test_generator = ModuloDataGenerator(10)
    train_loader, val_loader = test_generator.get_dataloader(0.9)
    print("Training dataloader has length", len(train_loader))
    print("The first training sample is:")
    for sample in train_loader:
        print(sample)
        break

