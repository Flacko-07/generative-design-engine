#!/usr/bin/env python3
"""Train diffusion model on combined_all.csv for 3-parameter inverse design."""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. Load and normalise
# ------------------------------------------------------------
df = pd.read_csv("combined_all.csv")
print(f"Total designs: {len(df)}")

conditions = df[["Cd", "flow_regime"]].values.astype(np.float32)
designs = df[["diameter", "x_center", "channel_height"]].values.astype(np.float32)

scaler_c = StandardScaler()
scaler_d = StandardScaler()
cond_norm = scaler_c.fit_transform(conditions)
design_norm = scaler_d.fit_transform(designs)

joblib.dump(scaler_c, "scaler_c.pkl")
joblib.dump(scaler_d, "scaler_d.pkl")

# ------------------------------------------------------------
# 2. Diffusion parameters
# ------------------------------------------------------------
T = 1000
betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

@tf.function
def q_sample(x0, t, noise):
    alpha_bar_t = tf.reshape(tf.gather(alpha_bars, t), [-1, 1])
    return tf.sqrt(alpha_bar_t) * x0 + tf.sqrt(1.0 - alpha_bar_t) * noise

# ------------------------------------------------------------
# 3. Denoising model
# ------------------------------------------------------------
def build_denoiser(design_dim, cond_dim):
    design_input = tf.keras.layers.Input(shape=(design_dim,))
    time_input = tf.keras.layers.Input(shape=(), dtype=tf.int32)
    cond_input = tf.keras.layers.Input(shape=(cond_dim,))

    time_embed = tf.keras.layers.Embedding(T, 64)(time_input)
    time_embed = tf.keras.layers.Flatten()(time_embed)

    x = tf.keras.layers.Concatenate()([design_input, cond_input, time_embed])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(design_dim)(x)

    model = tf.keras.Model([design_input, time_input, cond_input], output)
    return model

denoiser = build_denoiser(3, 2)
denoiser.summary()

# ------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 1500
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

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

dataset = tf.data.Dataset.from_tensor_slices((design_norm, cond_norm))
dataset = dataset.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x0_batch, cond_batch in dataset:
        loss = train_step(x0_batch, cond_batch)
        epoch_loss += loss.numpy()
    epoch_loss /= len(dataset)
    losses.append(epoch_loss)
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}/{EPOCHS}  loss: {epoch_loss:.6f}")

# Save model
denoiser.save("diffusion_denoiser.keras")
print("💾 Denoiser saved.")

# ------------------------------------------------------------
# 5. Sampling functions
# ------------------------------------------------------------
@tf.function
def p_sample(x_t, t, cond_batch):
    noise_pred = denoiser([x_t, t, cond_batch])
    beta_t = tf.reshape(tf.gather(betas, t), [-1, 1])
    alpha_t = tf.reshape(tf.gather(alphas, t), [-1, 1])
    alpha_bar_t = tf.reshape(tf.gather(alpha_bars, t), [-1, 1])
    coef1 = 1.0 / tf.sqrt(alpha_t)
    coef2 = beta_t / tf.sqrt(1.0 - alpha_bar_t)
    mean = coef1 * (x_t - coef2 * noise_pred)
    noise = tf.random.normal(shape=tf.shape(x_t))
    sigma = tf.sqrt(beta_t)
    return tf.where(tf.reshape(t, [-1, 1]) > 0, mean + sigma * noise, mean)

def generate_design(target_Cd, regime, num_samples=5):
    cond_raw = np.array([[target_Cd, regime]] * num_samples, dtype=np.float32)
    cond_tensor = tf.constant(scaler_c.transform(cond_raw))
    x = tf.random.normal((num_samples, 3))
    for t in reversed(range(T)):
        t_batch = tf.fill([num_samples], t)
        x = p_sample(x, t_batch, cond_tensor)
    return scaler_d.inverse_transform(x.numpy())

# Quick test
print("\n🔧 Sample generation (turbulent):")
for cd in [0.9, 1.0, 1.1]:
    designs = generate_design(cd, 1, 3)
    for d, xc, ch in designs:
        print(f"  Cd={cd:.2f} → D={d:.4f} x={xc:.4f} H={ch:.4f}")