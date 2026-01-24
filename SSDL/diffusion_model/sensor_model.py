import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, l = x.shape
        x = x.permute(0, 2, 1)  # (b, l, c)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, l, self.heads, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, l, -1)
        out = self.to_out(out)
        return out.permute(0, 2, 1)  # (b, c, l)

class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = TemporalAttention(dim)
        self.norm = nn.LayerNorm(dim)  # Normalize along the feature dimension

    def forward(self, x):
        x = x + self.attention(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)  # Apply LayerNorm across the channels dimension
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, channels, sequence_length)
        return x

class SensorConvLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, conv_channels, kernel_size, dropout=0.5):
        super(SensorConvLSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(conv_channels)  # Batch Normalization after Conv1D
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Temporal attention before LSTM
        self.temporal_attention = TemporalAttentionBlock(conv_channels)

        self.lstm = nn.LSTM(conv_channels, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)  # Keep it consistent with the cross-attention expectation

        # Linear layer to project the residual to match the LSTM output
        self.residual_projection = nn.Linear(90, hidden_size * 2)

    def forward(self, x):
        residual = x

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Apply temporal attention before LSTM
        x = self.temporal_attention(x)

        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        # Project the residual to match the LSTM output size
        residual = residual[:, :, -1]  # Get the last timestep from the input
        residual = self.residual_projection(residual)  # Project to match hidden_size * 2

        out += residual
        return out

class CombinedLSTMClassifier(nn.Module):
    def __init__(self, sensor_input_size, hidden_size, num_layers, num_classes, conv_channels, kernel_size, dropout=0.5, num_heads=8):
        super(CombinedLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.sensor1_conv_lstm = SensorConvLSTMClassifier(sensor_input_size, hidden_size, num_layers, conv_channels, kernel_size, dropout)
        self.sensor2_conv_lstm = SensorConvLSTMClassifier(sensor_input_size, hidden_size, num_layers, conv_channels, kernel_size, dropout)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sensor1_data, sensor2_data, return_attn_output=False):
        # print(sensor1_data.shape)
        sensor1_out = self.sensor1_conv_lstm(sensor1_data)
        sensor2_out = self.sensor2_conv_lstm(sensor2_data)

        sensor1_out = sensor1_out.unsqueeze(1)
        sensor2_out = sensor2_out.unsqueeze(1)

        attn_output, _ = self.attention(sensor1_out, sensor2_out, sensor2_out)
        attn_output = attn_output.squeeze(1)

        out = self.dropout(attn_output)
        class_output = self.fc(out)

        return class_output, out