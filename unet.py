import torch
import torch.nn as nn

# This ConvBlock is the elementary block of UNet (for both encoder and decoder)
# ConvBlock preserves dimensionality
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),  # similar to BatchNormalilatentation
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
    
  
# Elementary encoder block
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, downscale=2):
        super(UnetDown, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels, out_channels), 
            nn.MaxPool2d(downscale)
        )

    def forward(self, x):

        return self.model(x)
    
# Encoder network
class Encoder(nn.Module):
    def __init__(self, n_features):
        super(Encoder, self).__init__()

        self.initial_features = nn.Sequential(
            nn.Conv2d(1, n_features, 3, 1, 1),
            nn.GroupNorm(8, n_features),
            nn.ReLU(),
        )
        self.down1 = UnetDown(n_features, n_features, downscale=2)
        self.down2 = UnetDown(n_features, 2*n_features, downscale=2)
        self.down3 = UnetDown(2*n_features, 2*n_features, downscale=2)
        self.down4 = nn.Sequential(
            nn.AvgPool2d(3), 
            nn.ReLU())

    def forward(self, x):

        x_f = self.initial_features(x)
        d1 = self.down1(x_f)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        latent = self.down4(d3)
        return x_f, d1, d2, d3, latent
    
# UnetUp aims at reverting UnetDown, but also uses skip connections
# In some cases, output padding is needed (e.g., on the second layer,
# with upscale 2 we get Cx3x3--> C'x6x6, so we need extra_dim 1 to get C'x7x7)
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, upscale=2, extra_dim=0):
        super(UnetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, upscale, upscale, output_padding=extra_dim),
            ConvBlock(out_channels, out_channels),
            # ResBlock(out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)

        return x
    
# Decoder network
class Decoder(nn.Module):
    def __init__(self, n_features):
        super(Decoder, self).__init__()
        
        # first decoder layer does not have any skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(2*n_features, 2*n_features, 3, 3),
            nn.GroupNorm(8, 2*n_features),
            nn.ReLU(),
            
        )
        
        self.up2 = UnetUp(4*n_features, 2*n_features, upscale=2, extra_dim=1) 
        self.up3 = UnetUp(4*n_features, n_features, upscale=2)
        self.up4 = UnetUp(2*n_features, n_features, upscale=2)
        self.out = nn.Conv2d(2*n_features, 1, 3, 1, 1)

    def forward(self, latent, d3, d2, d1, x_f):

        u1 = self.up1(latent)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        eps_hat = self.out(torch.cat([u4, x_f], dim=1))
        return eps_hat
    
# Time embedding layer
class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.lin1 = nn.Linear(1, embedding_dim, bias=False)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, ts):
        ts = ts.view(-1, 1)
        temb = torch.sin(self.lin1(ts))  # sine activation is common to encode time information
        temb = self.lin2(temb)
        temb = temb.view(-1, self.embedding_dim, 1, 1)  # add dummy channels to sum with latent
        return temb
    
# Overall Unet model
class Unet(nn.Module):

    def __init__(self, n_features=64):
        super(Unet, self).__init__()
        self.encoder = Encoder(n_features)
        self.decoder = Decoder(n_features)
        self.time_embedding_layer = TimeEmbedding(2*n_features)
    
    def forward(self, x, t):

        x_f, d1, d2, d3, latent = self.encoder(x)
        temb = self.time_embedding_layer(t)
        latent_temb = latent + temb
        eps_hat = self.decoder(latent_temb, d3, d2, d1, x_f)  # intermediate decoder steps are not needed
        
        return eps_hat