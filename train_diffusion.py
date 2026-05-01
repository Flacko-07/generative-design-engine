#!/usr/bin/env python3
"""
Conditional Diffusion Model for Cylinder Inverse Design
Predicts (diameter, x_center, blockage_ratio) from (target_Cd, flow_regime)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model

# ─────────────────────────────────────────────────────────
# 1. Load and normalize data
# ─────────────────────────────────────────────────────────
df = pd.read_csv("combined_dataset.csv")
print("Columns:", df.columns.tolist())
print(f"Dataset size: {len(df)}")

# Inputs (what we want to control)
conditions = df[["Cd", "flow_regime"]].values.astype(np.float32)

# Outputs (design parameters)
designs = df[["diameter", "x_center", "blockage_ratio"]].values.astype(np.float32)

# Normalize
scaler_c = StandardScaler()
scaler_d = StandardScaler()
cond_norm = scaler_c.fit_transform(conditions)
design_norm = scaler_d.fit_transform(designs)

# Save scalers for inference
joblib.dump(scaler_c, "diffusion_scaler_c.gz")
joblib.dump(scaler_d, "diffusion_scaler_d.gz")

# ─────────────────────────────────────────────────────────
# 2. Diffusion hyperparameters
# ─────────────────────────────────────────────────────────
T = 1000                # number of diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

def q_sample(x0, t, noise):
    """Forward diffusion: x_t = sqrt(α_bar_t) * x0 + sqrt(1-α_bar_t) * noise"""
    alpha_bar_t = tf.gather(alpha_bars, t)
    alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1])
    return tf.sqrt(alpha_bar_t) * x0 + tf.sqrt(1.0 - alpha_bar_t) * noise

# ─────────────────────────────────────────────────────────
# 3. Denoising model (MLP)
# ─────────────────────────────────────────────────────────
def build_denoiser(design_dim, cond_dim):
    design_input = layers.Input(shape=(design_dim,))
    time_input = layers.Input(shape=(), dtype=tf.int32)
    cond_input = layers.Input(shape=(cond_dim,))

    # Time embedding (sinusoidal)
    time_embed = layers.Embedding(T, 64)(time_input)
    time_embed = layers.Flatten()(time_embed)

    # Concatenate everything
    x = layers.Concatenate()([design_input, cond_input, time_embed])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(design_dim)(x)   # predict noise

    return Model([design_input, time_input, cond_input], output)

denoiser = build_denoiser(3, 2)
denoiser.summary()

# ─────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 1000
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Compile with a dummy loss (we'll use custom training loop)
@tf.function
def train_step(x0_batch, cond_batch):
    batch_size = tf.shape(x0_batch)[0]
    t = tf.random.uniform([batch_size], minval=0, maxval=T, dtype=tf.int32)
    noise = tf.random.normal(shape=tf.shape(x0_batch))
    x_t = q_sample(x0_batch, t, noise)

    with tf.GradientTape() as tape:
        noise_pred = denoiser([x_t, t, cond_batch])
        loss = tf.reduce_mean(tf.square(noise - noise_pred))
    grads = tape.gradient(loss, denoiser.trainable_variables)
    optimizer.apply_gradients(zip(grads, denoiser.trainable_variables))
    return loss

# Training loop
dataset = tf.data.Dataset.from_tensor_slices((design_norm, cond_norm))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x0_batch, cond_batch in dataset:
        loss = train_step(x0_batch, cond_batch)
        epoch_loss += loss.numpy()
    epoch_loss /= len(dataset)
    losses.append(epoch_loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}/{EPOCHS}  loss: {epoch_loss:.6f}")

# Save model
denoiser.save("diffusion_denoiser.keras")
print("💾 Denoiser saved as diffusion_denoiser.keras")

# Plot loss
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.title("Diffusion training loss")
plt.grid(True)
plt.savefig("diffusion_loss.png", dpi=150)
print("📈 Loss plot saved.")

# ─────────────────────────────────────────────────────────
# 5. Sampling (reverse process)
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# 5. Sampling (corrected type handling)
# ─────────────────────────────────────────────────────────

@tf.function
def p_sample(x_t, t, cond_batch):
    """One reverse step – all inputs are tensors."""
    noise_pred = denoiser([x_t, t, cond_batch])
    beta_t = tf.gather(betas, t)
    alpha_t = tf.gather(alphas, t)
    alpha_bar_t = tf.gather(alpha_bars, t)

    coef1 = 1.0 / tf.sqrt(alpha_t)
    coef2 = beta_t / tf.sqrt(1.0 - alpha_bar_t)

    mean = coef1 * (x_t - coef2 * noise_pred)

    # At t > 0 we add noise; at t=0 we return the mean
    noise = tf.random.normal(shape=tf.shape(x_t))
    sigma = tf.sqrt(beta_t)
    return tf.where(tf.reshape(t, [-1, 1]) > 0, mean + sigma * noise, mean)

def generate_design(target_Cd, regime, num_samples=5):
    """Generate designs for given conditions."""
    # Prepare conditions as tensor
    cond_raw = np.array([[target_Cd, regime]] * num_samples, dtype=np.float32)
    cond_norm = scaler_c.transform(cond_raw)
    cond_tensor = tf.constant(cond_norm)

    # Start from pure noise
    x = tf.random.normal((num_samples, 3))

    # Reverse diffusion (for loop outside @tf.function is fine)
    for t in reversed(range(T)):
        t_batch = tf.fill([num_samples], t)
        x = p_sample(x, t_batch, cond_tensor)

    # Inverse transform
    design = scaler_d.inverse_transform(x.numpy())
    return design

print("\n🔧 Inverse design examples:")
for cd in [0.9, 1.0, 1.1]:
    designs = generate_design(cd, regime=1, num_samples=3)   # turbulent regime
    print(f"\nTarget Cd = {cd} (turbulent):")
    for i, (d, xc, br) in enumerate(designs, 1):
        print(f"  Design {i}: D={d:.4f}  x={xc:.4f}  blockage={br:.4f}")