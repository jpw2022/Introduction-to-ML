import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, seq_length, activation=F.relu):
        """
        
        parameters:
        input_size (int): imput
        hidden_sizes (list of int): 隐藏层大小列表
        output_size (int): 输出层大小
        activation (callable, optional): 激活函数，默认为ReLU
        """
        super(CustomMLP, self).__init__()
        self.real_input_size = input_size * seq_length
        
        # 输入层到第一个隐藏层的权重和偏置
        self.layers = nn.ModuleList([nn.Linear(self.real_input_size, hidden_sizes[0])])
        
        # 添加所有隐藏层
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # 最后一个隐藏层到输出层的权重和偏置
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.activation = activation

    def forward(self, x):
        """
        前向传播
        
        参数:
        x (torch.Tensor): 输入数据
        
        返回:
        torch.Tensor: 网络输出
        """
        x = x.reshape(-1, self.real_input_size)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 最后一层不应用激活函数
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
