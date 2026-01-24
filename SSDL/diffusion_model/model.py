import torch
from torch import nn
import torch.nn.functional as F

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, context_dim, sequence_length, num_heads=4):
        super().__init__()
        # latent_dim here is the channel dimension at the bottleneck (hidden_dim * 16)
        self.latent_projection = nn.Linear(latent_dim, latent_dim // 2) 
        self.context_projection = nn.Linear(context_dim, (latent_dim // 2) * sequence_length)
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim // 2, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(latent_dim // 2)
        self.feedforward = nn.Sequential(
            nn.Linear(latent_dim // 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim // 2)
        )

    def forward(self, latent, context):
        # latent shape: (Batch, Seq_Len, Channels)
        latent = self.latent_projection(latent)
        
        # context shape: (Batch, Context_Dim) -> Project to match latent capacity
        context = self.context_projection(context)
        
        # Reshape context to treat it as a sequence for attention
        # We force it to match the latent sequence length
        context = context.view(latent.size(0), latent.size(1), -1)

        attention_output, _ = self.attention(latent, context, context)
        latent = self.layernorm(attention_output + latent)

        latent = latent + self.feedforward(latent)
        return latent

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(Conv1DBlock, self).__init__()
        self.time_projection = nn.Linear(time_emb_dim, out_channels)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, time_emb):
        # Project time embedding and add it to the feature space
        time_emb = self.time_projection(time_emb).unsqueeze(-1)
        x = self.conv1d(x)
        x = x + time_emb
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        # Bidirectional = Output dim will be hidden_dim * 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (Batch, Channels, Length) -> (Batch, Length, Channels)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)  # Back to (Batch, Channels, Length)
        return x

class Downsample1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.transposed_conv(x)

class UNet1D(nn.Module):
    def __init__(self, num_joints=16, window_size=48, time_emb_dim=12, context_dim=512, hidden_dim=128, num_classes=14, dropout_prob=0.5):
        super().__init__()
        
        # DYNAMIC DIMENSIONS
        self.latent_dim = num_joints * 3 # 16 * 3 = 48
        
        # CALCULATE BOTTLENECK SIZE
        # 2 Downsamples: Window 48 -> 24 -> 12
        self.bottleneck_seq_len = window_size // 4 
        
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.input_conv = nn.Conv1d(self.latent_dim, hidden_dim, kernel_size=1, padding=0)

        # Encoder 1
        self.lstm1 = LSTMBlock(hidden_dim, hidden_dim)
        self.conv1 = Conv1DBlock(hidden_dim * 2, hidden_dim * 2, time_emb_dim)
        self.downsample1 = Downsample1D(hidden_dim * 2, hidden_dim * 4) # 48 -> 24
        
        # Encoder 2
        self.lstm2 = LSTMBlock(hidden_dim * 4, hidden_dim * 4)
        self.conv2 = Conv1DBlock(hidden_dim * 8, hidden_dim * 4, time_emb_dim)
        self.downsample2 = Downsample1D(hidden_dim * 4, hidden_dim * 8) # 24 -> 12

        # Encoder 3
        self.lstm3 = LSTMBlock(hidden_dim * 8, hidden_dim * 8)
        self.conv3 = Conv1DBlock(hidden_dim * 16, hidden_dim * 16, time_emb_dim)

        # Cross-Attention
        self.cross_attn = CrossAttention(
            latent_dim=hidden_dim * 16, 
            context_dim=context_dim, 
            sequence_length=self.bottleneck_seq_len # Dynamic!
        )

        # Decoder 1
        self.conv4 = Conv1DBlock(hidden_dim * 8, hidden_dim * 4, time_emb_dim)
        self.lstm4 = LSTMBlock(hidden_dim * 4, hidden_dim * 4)
        self.upsample1 = Upsample1D(hidden_dim * 8, hidden_dim * 4)

        # Decoder 2
        self.conv5 = Conv1DBlock(hidden_dim * 4, hidden_dim * 2, time_emb_dim)
        self.lstm5 = LSTMBlock(hidden_dim * 2, hidden_dim * 2)
        self.upsample2 = Upsample1D(hidden_dim * 4, hidden_dim * 2)

        # Decoder 3
        self.conv6 = Conv1DBlock(hidden_dim * 2, hidden_dim, time_emb_dim)
        self.lstm6 = LSTMBlock(hidden_dim, self.latent_dim) 
        
        # FINAL CONV FIX: Input is latent_dim * 2 (from bidirectional LSTM6)
        self.final_conv = nn.Conv1d(self.latent_dim * 2, self.latent_dim, kernel_size=1, padding=0)

    def forward(self, x, context, time, sensor_pred):
        time_emb = self.time_mlp(time)
        class_emb = self.class_emb(sensor_pred)
        
        time_emb = time_emb + class_emb
        x = self.input_conv(x)

        # Encoder Path
        x = self.lstm1(x)
        x = self.conv1(x, time_emb)
        x = self.downsample1(x)

        x = self.lstm2(x)
        x = self.conv2(x, time_emb)
        x = self.downsample2(x)

        x = self.lstm3(x)
        x = self.conv3(x, time_emb)

        # Bottleneck (Cross-Attention)
        x = x.permute(0, 2, 1) # (B, C, L) -> (B, L, C)
        x = self.cross_attn(x, context)
        x = x.permute(0, 2, 1) # Back to (B, C, L)

        # Decoder Path
        x = self.conv4(x, time_emb)
        x = self.lstm4(x)
        x = self.upsample1(x)

        x = self.conv5(x, time_emb)
        x = self.lstm5(x)
        x = self.upsample2(x)

        x = self.conv6(x, time_emb)
        x = self.lstm6(x)
        x = self.final_conv(x)

        return x

class Diffusion1D(nn.Module):
    def __init__(self, num_joints=16, num_classes=14, window_size=48):
        super().__init__()
        self.latent_dim = num_joints * 3
        self.unet = UNet1D(
            num_joints=num_joints, 
            num_classes=num_classes, 
            window_size=window_size
        )
        self.final = nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=1, padding=0)
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, latent, context, time, sensor_pred):
        latent = self.unet(latent, context, time, sensor_pred)
        return self.final(latent)