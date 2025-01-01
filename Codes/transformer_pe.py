import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        
        # create the position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(10.0 / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd: cos
        
        pe = pe.unsqueeze(0)  # the first dimension: batch size 
        self.register_buffer('pe', pe)

    def forward(self, x):
        # shape of x: (batch_size, seq_length, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# 定义 Transformer 模型
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(SimpleTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dim = model_dim
        
        # 线性层将输入从 input_dim 映射到 model_dim
        self.input_fc = nn.Linear(input_dim, model_dim)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(model_dim)
        
        # Transformer 编码器部分
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,  # 模型的维度
            nhead=num_heads,    # 自注意力机制中的头数
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers  # 编码器层数
        )

        # 线性层用于最终输出
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # 输入 x 的形状是 (batch_size, seq_length, input_dim)
        # 先通过线性层将输入从 input_dim 映射到 model_dim
        x = self.input_fc(x)  # 现在的 x 形状是 (batch_size, seq_length, model_dim)

        # 加入 Positional Encoding
        x = self.positional_encoding(x)  # x 形状仍然是 (batch_size, seq_length, model_dim)

        # 使用 Transformer 编码器处理输入
        transformer_out = self.transformer_encoder(x)

        # 从 Transformer 输出中提取最后一个时刻的隐藏状态
        output = transformer_out[:, -1, :]  # 取最后一个时间步，形状 (batch_size, model_dim)

        # 通过线性层输出最终的预测结果
        output = self.fc_out(output)  # 形状为 (batch_size, output_dim)
        
        return output
