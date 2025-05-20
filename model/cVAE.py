import torch
import torch.nn as nn

# ========= Conditional VAE 정의 =========
# class Encoder(nn.Module):
#     def __init__(self, input_dim: int, cond_dim: int, hidden_dims: list[int], latent_dim: int):
#         super().__init__()
#         dims = [input_dim + cond_dim] + hidden_dims

#         layers = []
#         for i in range(len(dims) - 1):
#             layers.append(nn.Linear(dims[i], dims[i + 1]))
#             layers.append(nn.ELU())
#         self.fc = nn.Sequential(*layers)

#         self.mu = nn.Linear(hidden_dims[-1], latent_dim)
#         self.logvar = nn.Linear(hidden_dims[-1], latent_dim)

#     def forward(self, x, y):
#         h = self.fc(torch.cat([x, y], dim=1))
#         return self.mu(h), self.logvar(h)

# class Decoder(nn.Module):
#     def __init__(self, output_dim: int, cond_dim: int, hidden_dims: list[int], latent_dim: int):
#         super().__init__()
#         dims = [latent_dim + cond_dim] + hidden_dims + [output_dim]

#         layers = []
#         for i in range(len(dims) - 2):
#             layers.append(nn.Linear(dims[i], dims[i + 1]))
#             layers.append(nn.ELU())
#         layers.append(nn.Linear(dims[-2], dims[-1]))  # 마지막 출력층 (ReLU 없음)

#         self.fc = nn.Sequential(*layers)

#     def forward(self, z, y):
#         return self.fc(torch.cat([z, y], dim=1))
class Encoder(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        self.layers = nn.ModuleList()
        in_dim = input_dim + cond_dim  # 초기 입력은 x + y

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim + input_dim, h_dim))  # h + x
            in_dim = h_dim

        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x, y):
        h = torch.cat([x, y], dim=1)  # 초기 입력
        for layer in self.layers:
            h = torch.cat([h, x], dim=1)  # 매 layer마다 x 추가
            h = torch.relu(layer(h))
        return self.mu(h), self.logvar(h)
    
class Decoder(nn.Module):
    def __init__(self, output_dim: int, cond_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.layers = nn.ModuleList()
        in_dim = latent_dim + cond_dim  # 초기 입력은 z + y

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim + latent_dim, h_dim))  # h + z
            in_dim = h_dim

        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)  # 초기 입력
        for layer in self.layers:
            h = torch.cat([h, z], dim=1)  # 매 layer마다 z 추가
            h = torch.relu(layer(h))
        return self.output_layer(h)

class cVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, cond_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(input_dim, cond_dim, hidden_dims[::-1], latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, y)
        return x_hat, mu, logvar

# ========= 손실 함수 =========
def vae_loss(x_hat, x, mu, logvar, beta):
    recon_loss = nn.MSELoss()(x_hat, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_div , recon_loss, kl_div