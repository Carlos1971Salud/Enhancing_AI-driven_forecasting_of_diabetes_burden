Code for Forecasting Disease Burden Using Deep Learning Models
=========================================================================

This repository contains Python implementations of the forecasting models used in our study on the burden of Type 2 Diabetes.

--------------------------------------------------------------------------------
Files Included:
--------------------------------------------------------------------------------

1. transformer_vae.py
   - Implements a Transformer-based Variational Autoencoder (VAE).
   - Contains encoder, decoder, and a forecasting transformer head.
   - Includes training and prediction routines.

2. lstm_model.py
   - Implements an LSTM Autoencoder.
   - Suitable for sequence-to-sequence reconstruction and forecasting.

3. gru_model.py
   - Implements a GRU Autoencoder.
   - Similar architecture to LSTM, but using GRU units.


--------------------------------------------------------------------------------
Dependencies:
--------------------------------------------------------------------------------
- Python >= 3.8
- torch >= 1.10
- numpy
- pandas
- scikit-learn
- statsmodels

You can install required packages using:

    pip install torch numpy pandas scikit-learn statsmodels

--------------------------------------------------------------------------------
Usage Instructions:
--------------------------------------------------------------------------------
Each script contains a function called `train_and_predict()` which accepts:
- Set the model parameters in config file.
- `train_data`: Training time series data (as NumPy array)
- `test_data`: Data for prediction
- `input_dim` / `seq_len`: Input dimensionality and sequence length

Example (for Transformer-VAE):

    from transformer_vae import train_and_predict
    predictions = train_and_predict(train_data, test_data, input_dim=6, seq_len=15)

--------------------------------------------------------------------------------
License:
--------------------------------------------------------------------------------
This repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to use, share, and adapt the contents of this repository for non-commercial research and educational purposes, provided that appropriate credit is given.

Commercial use is not permitted without prior written permission from the authors.

To request permission for commercial use, please contact us via the corresponding author's email provided in the paper.


