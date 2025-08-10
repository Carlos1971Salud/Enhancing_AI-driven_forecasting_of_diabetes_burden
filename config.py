CONFIG = {
    "device": None,
    "seed": {
        "python": None,
        "numpy": None,
        "torch": None,
        "deterministic": None, 
        "benchmark": None,
    },

    "dataset": {
        "input_dim": None,
        "seq_len": None,
    },
    "dataloader": {
        "batch_size": None,   
        "shuffle": None,        
        "num_workers": 0,
        "pin_memory": None,
        "drop_last": None,
        "persistent_workers": None,
    },

    "training": {
        "epochs": None,
        "lr": None,
        "optimizer": None, 
        "optimizer_params":
        },
        "scheduler": None, 
        "scheduler_params": {},
        "grad_clip_norm": None,
        "loss": {
            "reduction": None
        },
        "log_interval": 1, 
    },

    "gru": {
        "hidden_dim": None,
        "num_layers": None,
        "dropout": None,
        "bidirectional": None,
    },

    "lstm": {
        "hidden_dim": None,
        "num_layers": None,
        "dropout": 0.0,
        "bidirectional": None,
    },

    "transformer": {
        "d_model": None,
        "nhead": None,
        "num_layers": None,
        "dropout": None,
        "dim_feedforward": None,          
        "forecast_num_layers": None,
        "forecast_dim_feedforward": None, 
        "pe_max_len": None,  
    },
    "vae": {
        "latent_dim": None,
        "beta": 1.0, 
    },
}
