import torch.nn as nn


class MultiAxisAttention(nn.Module):
    def __init__(self, input_channels, embed_dim, num_heads, depth, dim_feedforward, reduction_factor, dropout=0.1):
        super(MultiAxisAttention, self).__init__()
        self.embed_dim = embed_dim
        
        self.expand_channels = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        
        # Attention per Channel
        self.channel_attention = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Attention per Frequency
        self.freq_attention = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Reduce subcarriers
        self.reduce_frequency = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((embed_dim // reduction_factor, None))
        )
        
        # Fully connected layer 
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Input shape: [B, C, F, T]
        Output shape: [B, embed_dim, F_reduced, T]
        """
        B, C, F, T = x.shape

        # **(Expand Channels)**
        x = self.expand_channels(x)  # [B, embed_dim, F, T]

        # **Channel Attention**
        # Reshape to fit Transformer
        x_c = x.permute(0, 2, 3, 1).reshape(B * F * T, self.embed_dim)  # [B * F * T, embed_dim]
        x_c = x_c.view(F * T, B, self.embed_dim)  # [seq_len, batch_size, embed_dim]
        
        for layer in self.channel_attention:
            x_c = layer(x_c)  # Transformer per Channel

        # Recover size after Channel Attention
        x_c = x_c.permute(1, 2, 0).view(B, self.embed_dim, F, T)  # [B, embed_dim, F, T]

        # **Frequency Attention**
        # Reshape input to fit Transformer
        x_f = x.permute(0, 1, 3, 2).reshape(B * self.embed_dim * T, F)  # [B * embed_dim * T, F]
        x_f = x_f.view(F, B * T, self.embed_dim)  # [seq_len, batch_size, embed_dim]
        
        for layer in self.freq_attention:
            x_f = layer(x_f)  # Transformer xử lý trên Frequency

        # Recover size after Frequency Attention
        x_f = x_f.permute(1, 2, 0).view(B, self.embed_dim, T, F).permute(0, 1, 3, 2)  # [B, embed_dim, F, T]

        # **Features extraction**
        out = x_c + x_f  # Extract per Channel and Frequency

        # **Reduce subcarriers (Reduce Frequency)**
        out = self.reduce_frequency(out)  # [B, embed_dim, F_reduced, T]

        # **Projection Layer**
        out = self.fc(out.permute(0, 2, 3, 1))  # [B, F_reduced, T, embed_dim]
        out = out.permute(0, 3, 1, 2)  # [B, embed_dim, F_reduced, T]

        return out
