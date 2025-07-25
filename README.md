# Anomaly Detection in High-Frequency Limit Order Book Data (LOBSTER)

This project focuses on detecting anomalies in high-frequency limit order book (LOB) data using an unsupervised deep learning approach. We use an **LSTM Autoencoder** trained to reconstruct normal LOB behavior and identify anomalies based on **reconstruction error**.



## Overview

- **Data Source**: [LOBSTER](https://lobsterdata.com/) - limit order book data at nanosecond precision.
- **Goal**: Identify rare or unusual market behaviors in order book dynamics using unsupervised learning.
- **Approach**:
  - Train an LSTM Autoencoder on normal LOB sequences.
  - Compute reconstruction error for each sample.
  - Flag samples in the top 1% of reconstruction errors as anomalies.



## Model Details

- **Architecture**: LSTM Autoencoder
  - Encoder: 2 stacked LSTM layers (128 and 64 units)
  - Decoder: Fully connected dense layer + reshape
- **Loss Function**: Mean Squared Error (MSE) between original and reconstructed sequences
- **Thresholding**: Anomalies are defined as samples with reconstruction error in the 99th percentile.


## References

- [LOBSTER Data](https://lobsterdata.com/)
- Poutré et al. (2024) "Deep unsupervised anomaly detection in high-frequency markets"
- Siering et al. (2017) "A taxonomy of financial manipulations"
