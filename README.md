# Generative Design Engine

> **Inverse design for bluff-body aerodynamics** — given a target drag coefficient (Cd), generate valid cylinder geometries using a Conditional GAN and a Denoising Diffusion model, both trained on OpenFOAM simulation data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-v2206-green)](https://www.openfoam.com/)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel)](https://vercel.com/)

---

## Overview

Traditional CFD-based design loops are expensive: each geometry evaluation requires a full simulation. This project inverts that loop. Two generative models learn the mapping **Cd → (diameter, x\_center)** from a parametric OpenFOAM dataset, enabling real-time inverse design without running a single new simulation.

The pipeline has three stages:

1. **Data generation** — `automate_pitzdaily.py` / `parametric_cylinder.py` run parametric OpenFOAM sweeps and export drag coefficient data to CSV.
2. **Model training** — `cgan.py` trains a Conditional GAN; `train_diffusion.py` trains a latent diffusion model; `surrogate_model.py` fits a Gaussian-Process surrogate for fast evaluation.
3. **Inference & validation** — `inference.py` generates candidate geometries from a target Cd; `validate*.py` scripts verify predictions against held-out simulation data.

A static Vercel dashboard (`index.html` / `app.js` / `styles.css`) wraps everything in a browser UI, and a serverless API route (`/api/generate`) mirrors the CLI interface.

---

## Architecture

```
Target Cd (scalar)
       │
       ▼
┌─────────────────────────┐
│   cGAN Generator        │  latent_dim=10  · 2× Dense(64, ReLU)
│   (cgan.py)             │  → (diameter, x_center)  [2D design space]
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│   Diffusion Model       │  T=1000 timesteps · UNet-style MLP
│   (train_diffusion.py)  │  → refined / diverse samples
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│   GP Surrogate          │  Gaussian Process regression
│   (surrogate_model.py)  │  → predicted Cd + uncertainty
└─────────────────────────┘
       │
       ▼
  Validate vs OpenFOAM
  (validate*.py)
```

Both the cGAN and diffusion model operate in a normalised latent space (StandardScaler) and invert back to physical units at inference time. The GP surrogate provides an uncertainty estimate that ranks candidates before any expensive CFD check.

---

## Dataset

Datasets are produced by parametric OpenFOAM runs over two regimes:

| File | Description | Regime |
|---|---|---|
| `cylinder_dataset_large.csv` | Primary cylinder sweep | Laminar |
| `turbulent_dataset.csv` | High-Re sweep | Turbulent (k-ε) |
| `turbulent_4param.csv` | 4-parameter turbulent sweep | Turbulent |
| `combined_all.csv` | Merged laminar + turbulent | Mixed |
| `combined_final.csv` | Cleaned final training set | Mixed |

Each row contains geometry parameters (`diameter`, `x_center`, …) and the resulting drag coefficient (`Cd`). The `regime` column encodes `0` (laminar) or `1` (turbulent).

---

## Quick Start

### 1 · Install dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

> **Note:** OpenFOAM is only required to regenerate datasets. Pre-computed CSVs are already included.

### 2 · Train the cGAN

```bash
python cgan.py
# Trains for 6000 epochs; saves cgan_generator.keras + scalers
```

### 3 · Train the diffusion model

```bash
python train_diffusion.py
# Outputs diffusion_loss.png after training
```

### 4 · Run inverse design inference

```bash
# Generate 5 candidate geometries for Cd = 1.10, turbulent regime
python inference.py --cd 1.10 --regime 1 --count 5
```

Sample output:

```
Target Cd = 1.10 | regime = turbulent
  Design 1: diameter=0.0842  x_center=0.3201
  Design 2: diameter=0.0911  x_center=0.2987
  ...
```

### 5 · Validate predictions

```bash
python validate.py          # laminar validation
python validate_turbulent.py  # turbulent validation
```

---

## Web Dashboard & API

Deploy the static site to Vercel directly from the repo root (`vercel.json` is already configured):

```bash
vercel --prod
```

The dashboard calls a serverless API endpoint that wraps `inference.py` logic:

```
GET /api/generate?cd=1.10&regime=1&count=5
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cd` | float | required | Target drag coefficient |
| `regime` | int | `0` | `0` = laminar, `1` = turbulent |
| `count` | int | `5` | Number of designs to return |

---

## Results

Training plots and validation outputs are committed alongside the code:

| Artifact | Description |
|---|---|
| `cgan_designs.png` | Generated design cloud vs real data |
| `cgan_improved_designs.png` | Improved generation after anti-mode-collapse tuning |
| `cgan_training.png` | Generator & discriminator loss curves |
| `diffusion_loss.png` | Diffusion model training loss |
| `gp_surrogate.png` | GP surrogate fit and uncertainty bands |
| `validation_final.png` | Predicted vs simulated Cd scatter |
| `validation_error.png` | Absolute prediction error distribution |

---

## Repository Structure

```
generative-design-engine/
├── cgan.py                   # Conditional GAN (phase 3, anti-collapse)
├── train_diffusion.py        # Denoising diffusion model
├── surrogate_model.py        # Gaussian-Process surrogate
├── inference.py              # CLI inference entry point
├── inverse_design.py         # Core inverse-design logic
├── parametric_cylinder.py    # OpenFOAM parametric sweep (laminar)
├── parametric_step_duct.py   # Step-duct geometry sweep
├── automate_pitzdaily.py     # OpenFOAM pitzDaily automation
├── turbulent_*.py            # Turbulent-regime scripts
├── validate*.py              # Validation suites
├── scaling.py                # Data scaling utilities
├── combo*.py                 # Combined-dataset utilities
├── *.csv                     # Pre-computed training / validation data
├── *.png                     # Training & validation plots
├── index.html                # Vercel dashboard
├── styles.css                # Dashboard styles
├── app.js                    # Dashboard JS
├── api/                      # Vercel serverless functions
└── vercel.json               # Vercel deployment config
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| CFD simulation | OpenFOAM v2206 |
| Generative model | TensorFlow / Keras (cGAN, Diffusion) |
| Surrogate model | scikit-learn (GaussianProcessRegressor) |
| Data processing | pandas, NumPy, scikit-learn |
| Visualisation | Matplotlib |
| Web frontend | Vanilla JS, HTML/CSS |
| Deployment | Vercel (static + serverless) |

---

## Author

**Naval Singh** · [github.com/Flacko-07](https://github.com/Flacko-07)  
*ML & Scientific Computing | GNN · PINN · Open Source*
