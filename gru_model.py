import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG


class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0, bidirectional=False):
        super(GRUAutoencoder, self).__init__()
        self.bidirectional = bidirectional
        gru_hidden = hidden_dim * (2 if bidirectional else 1)

        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.decoder = nn.GRU(
            gru_hidden,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.output_layer = nn.Linear(gru_hidden, input_dim)

    def forward(self, x):
        _, hidden = self.encoder(x) 
        last = hidden[-1]           
        if self.bidirectional:
            forward_last = hidden[-2]
            backward_last = hidden[-1]
            last = torch.cat([forward_last, backward_last], dim=-1)  # (B, 2*hidden_dim)
        hidden_repeated = last.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(hidden_repeated)
        out = self.output_layer(decoded)
        return out


def gru_loss(recon_x, x):
    reduction = CONFIG["training"]["loss"].get("reduction", "sum")
    return F.mse_loss(recon_x, x, reduction=reduction)


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
        np.random.seed(s["numpy"])  # type: ignore
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
    hidden_dim = CONFIG["gru"]["hidden_dim"]
    num_layers = CONFIG["gru"]["num_layers"]
    dropout = CONFIG["gru"].get("dropout", 0.0)
    bidirectional = CONFIG["gru"].get("bidirectional", False)

    epochs = CONFIG["training"]["epochs"]
    lr = CONFIG["training"]["lr"]
    grad_clip_norm = CONFIG["training"].get("grad_clip_norm")

    # dataloader block
    dl_cfg = CONFIG.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size")
    shuffle = dl_cfg.get("shuffle", True)
    num_workers = dl_cfg.get("num_workers", 0)
    pin_memory = dl_cfg.get("pin_memory", False)
    drop_last = dl_cfg.get("drop_last", False)
    persistent_workers = dl_cfg.get("persistent_workers", False)

    for k, v in {
        "dataset.input_dim": input_dim,
        "gru.hidden_dim": hidden_dim,
        "gru.num_layers": num_layers,
        "training.epochs": epochs,
        "training.lr": lr,
        "dataloader.batch_size": batch_size,
    }.items():
        if v is None:
            raise ValueError(f"CONFIG value '{k}' must be set.")

    device = _get_device()

    model = GRUAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
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
            recon_x = model(x)
            loss = gru_loss(recon_x, x)
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
        pred = model(x_test)
    return pred.cpu().numpy()
