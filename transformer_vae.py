import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from config import CONFIG


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class ForecastTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, dim_feedforward, pe_max_len):
        super(ForecastTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=pe_max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, memory):
        memory = self.input_proj(memory)
        memory = self.pos_encoder(memory)
        tgt = torch.zeros_like(memory)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory)
        return self.output_proj(output)


class TransformerVAE(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, nhead, num_layers, latent_dim, dropout, forecast_layers, dim_feedforward, forecast_dim_feedforward, pe_max_len):
        super(TransformerVAE, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_mu = nn.Linear(d_model * seq_len, latent_dim)
        self.fc_logvar = nn.Linear(d_model * seq_len, latent_dim)
        self.fc_latent = nn.Linear(latent_dim, d_model * seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_dim)

        self.forecaster = ForecastTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=forecast_layers,
            dropout=dropout,
            dim_feedforward=forecast_dim_feedforward,
            pe_max_len=pe_max_len,
        )

    def encode(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.contiguous().view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_latent(z).view(-1, self.seq_len, self.d_model)
        tgt = torch.zeros_like(x)
        x = self.transformer_decoder(tgt, x)
        return self.output_layer(x)

    def forecast(self, recon_x):
        return self.forecaster(recon_x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        pred_y = self.forecast(recon_x)
        return recon_x, pred_y, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    reduction = CONFIG["training"]["loss"].get("reduction", "sum")
    recon_loss = F.mse_loss(recon_x, x, reduction=reduction)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta = float(CONFIG["vae"].get("beta", 1.0))
    return recon_loss + beta * kl_loss


def _get_device():
    dev = CONFIG.get("device")
    if dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _maybe_seed():
    s = CONFIG.get("seed", {})
    import random, numpy as np
    if s.get("python") is not None:
        random.seed(s["python"])
    if s.get("numpy") is not None:
        np.random.seed(s["numpy"])
    if s.get("torch") is not None:
        torch.manual_seed(s["torch"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s["torch"])
    torch.backends.cudnn.deterministic = bool(s.get("deterministic", False))
    torch.backends.cudnn.benchmark = bool(s.get("benchmark", True))


def _make_optimizer(model_params, lr):
    name = CONFIG["training"].get("optimizer", "Adam")
    params = CONFIG["training"].get("optimizer_params", {}) or {}
    opt_cls = getattr(torch.optim, name)
    return opt_cls(model_params, lr=lr, **params)


def _maybe_scheduler(optimizer):
    name = CONFIG["training"].get("scheduler")
    if not name:
        return None
    params = CONFIG["training"].get("scheduler_params", {}) or {}
    sched_cls = getattr(torch.optim.lr_scheduler, name)
    return sched_cls(optimizer, **params)


def train_and_predict(train_data, test_data):
    _maybe_seed()

    input_dim = CONFIG["dataset"]["input_dim"]
    seq_len = CONFIG["dataset"]["seq_len"]
    d_model = CONFIG["transformer"]["d_model"]
    nhead = CONFIG["transformer"]["nhead"]
    num_layers = CONFIG["transformer"]["num_layers"]
    dropout = CONFIG["transformer"]["dropout"]
    dim_feedforward = CONFIG["transformer"].get("dim_feedforward", 512)
    forecast_layers = CONFIG["transformer"].get("forecast_num_layers")
    forecast_dim_feedforward = CONFIG["transformer"].get("forecast_dim_feedforward", 512)
    pe_max_len = CONFIG["transformer"].get("pe_max_len", 500)
    latent_dim = CONFIG["vae"]["latent_dim"]

    epochs = CONFIG["training"]["epochs"]
    lr = CONFIG["training"]["lr"]
    grad_clip_norm = CONFIG["training"].get("grad_clip_norm")

    dl_cfg = CONFIG.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size")
    shuffle = dl_cfg.get("shuffle", True)
    num_workers = dl_cfg.get("num_workers", 0)
    pin_memory = dl_cfg.get("pin_memory", False)
    drop_last = dl_cfg.get("drop_last", False)
    persistent_workers = dl_cfg.get("persistent_workers", False)

    for k, v in {
        "dataset.input_dim": input_dim,
        "dataset.seq_len": seq_len,
        "transformer.d_model": d_model,
        "transformer.nhead": nhead,
        "transformer.num_layers": num_layers,
        "transformer.dropout": dropout,
        "transformer.dim_feedforward": dim_feedforward,
        "transformer.forecast_num_layers": forecast_layers,
        "vae.latent_dim": latent_dim,
        "training.epochs": epochs,
        "training.lr": lr,
        "dataloader.batch_size": batch_size,
    }.items():
        if v is None:
            raise ValueError(f"CONFIG value '{k}' must be set.")

    device = _get_device()

    model = TransformerVAE(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        latent_dim=latent_dim,
        dropout=dropout,
        forecast_layers=forecast_layers,
        dim_feedforward=dim_feedforward,
        forecast_dim_feedforward=forecast_dim_feedforward,
        pe_max_len=pe_max_len,
    ).to(device)

    optimizer = _make_optimizer(model.parameters(), lr=lr)
    scheduler = _maybe_scheduler(optimizer)

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    model.train()
    log_interval = int(CONFIG["training"].get("log_interval", 1))
    for epoch in range(int(epochs)):
        total_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, pred_y, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            total_loss += float(loss.item())
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        if scheduler is not None:
            scheduler.step()

    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(test_data, dtype=torch.float32).to(device)
        _, y_pred, _, _ = model(x_test)
    return y_pred.cpu().numpy()
