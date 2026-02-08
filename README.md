# EnergyMamba

**EnergyMamba: A Graph-Enhanced State Space Model for Uncertainty-Aware Energy Consumption Prediction**

EnergyMamba addresses the challenge of accurate and reliable energy consumption forecasting by combining spatiotemporal modeling with uncertainty quantification. The framework consists of two key components:

1. **Graph-Enhanced Selective State Space Model (GE-Mamba)**: Injects spatial context from the power grid topology into Mamba's selective state transitions via GCN-based neighborhood aggregation, organized within a U-Net encoder-decoder architecture for multi-scale temporal pattern extraction.

2. **Adaptive Sequential Conformalized Quantile Regression (AS-CQR)**: A distribution-free post-hoc calibration framework that produces reliable prediction intervals through locally adaptive nonconformity scores and an online feedback mechanism for adapting to distribution shifts.

## Project Structure

```
code/
├── main.py                    # Entry point
├── base/
│   ├── model.py               # Base model class
│   ├── engine.py              # Training and evaluation engine
│   └── metrics.py             # Evaluation metrics
├── model/
│   ├── energy_mamba.py        # GE-Mamba model (GCN + Bidirectional Mamba + U-Net)
│   └── ascqr.py               # AS-CQR calibration module
└── utils/
    ├── args.py                # Argument configuration
    ├── dataloader.py          # Data loading utilities
    ├── generate.py            # Data scalers
    ├── graph_algo.py          # GCN normalization
    └── log.py                 # Logging
```

## Model Architecture

- **Input Embedding**: Projects each node's time series into a latent space with learnable positional encoding
- **GE-Mamba Block**: GCN spatial context extraction followed by bidirectional Mamba with spatial-conditioned selectivity, connected via residual paths with RMSNorm
- **U-Net Backbone**: Encoder stages with temporal downsampling, bottleneck, and decoder stages with skip connections
- **Output Heads**: Three separate linear projections (lower quantile, median, upper quantile) from the last time step representation
- **Training Objective**: Composite pinball loss across quantile levels with monotonicity penalty

## Dependencies

- PyTorch
- mamba-ssm
- numpy
- scipy
