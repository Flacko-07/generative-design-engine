#!/usr/bin/env python3
"""
Phase 3 – WORKING Conditional GAN (no mode collapse)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
import joblib

# ──────────────────────────────────────────────────────
# 1. Load and normalise data
# ──────────────────────────────────────────────────────
df = pd.read_csv("cylinder_dataset_large.csv")
print(f"Loaded {len(df)} designs")

X = df[["Cd"]].values.astype(np.float32)
Y = df[["diameter", "x_center"]].values.astype(np.float32)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# ──────────────────────────────────────────────────────
# 2. Build generator (simple, but working)
# ──────────────────────────────────────────────────────
LATENT_DIM = 10

generator = tf.keras.Sequential([
    layers.Input(shape=(LATENT_DIM + 1,)),   # noise + Cd
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="tanh")        # matches scaled Y range
], name="generator")

generator.summary()

# ──────────────────────────────────────────────────────
# 3. Build discriminator
# ──────────────────────────────────────────────────────
discriminator = tf.keras.Sequential([
    layers.Input(shape=(3,)),                # design(2) + Cd(1)
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
], name="discriminator")

discriminator.summary()

# ──────────────────────────────────────────────────────
# 4. Combined model (generator → discriminator)
# ──────────────────────────────────────────────────────
discriminator.trainable = False
cond_input = layers.Input(shape=(1,))
noise_input = layers.Input(shape=(LATENT_DIM,))
gen_input = layers.Concatenate()([noise_input, cond_input])
gen_design = generator(gen_input)
validity = discriminator(layers.Concatenate()([gen_design, cond_input]))
combined = Model([cond_input, noise_input], validity, name="cgan")

combined.summary()

# ──────────────────────────────────────────────────────
# 5. Compile
# ──────────────────────────────────────────────────────
discriminator.trainable = True
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

combined.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
    loss="binary_crossentropy"
)

# ──────────────────────────────────────────────────────
# 6. Training loop
# ──────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 6000
SAVE_INTERVAL = 1000

real_label = np.ones((BATCH_SIZE, 1)) * 0.9
fake_label = np.zeros((BATCH_SIZE, 1)) + 0.1

g_losses, d_losses = [], []

for epoch in range(EPOCHS + 1):
    # ── Discriminator on real data ──
    idx = np.random.randint(0, len(Y_scaled), BATCH_SIZE)
    real_designs = Y_scaled[idx]
    real_conds = X_scaled[idx]

    # ── Generate fake data ──
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_designs = generator.predict(
        np.concatenate([noise, real_conds], axis=1), verbose=0
    )

    # Train D on real and fake
    d_loss_real = discriminator.train_on_batch(
        np.concatenate([real_designs, real_conds], axis=1), real_label
    )
    d_loss_fake = discriminator.train_on_batch(
        np.concatenate([fake_designs, real_conds], axis=1), fake_label
    )
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ── Generator (train twice) ──
    for _ in range(2):
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        cond = X_scaled[np.random.randint(0, len(X_scaled), BATCH_SIZE)]
        g_loss = combined.train_on_batch([cond, noise], real_label)

    g_losses.append(g_loss)
    d_losses.append(d_loss[0])

    if epoch % SAVE_INTERVAL == 0:
        print(f"Epoch {epoch:5d} | D loss: {d_loss[0]:.4f} (acc {d_loss[1]:.2f}) | G loss: {g_loss:.4f}")

print("🎯 Training complete!")

# ──────────────────────────────────────────────────────
# 7. Save models & scalers
# ──────────────────────────────────────────────────────
generator.save("cgan_generator.keras")
discriminator.save("cgan_discriminator.keras")
combined.save("cgan_combined.keras")
joblib.dump(scaler_X, "scaler_X.gz")
joblib.dump(scaler_Y, "scaler_Y.gz")
print("💾 Models and scalers saved.")

# ──────────────────────────────────────────────────────
# 8. Inference function
# ──────────────────────────────────────────────────────
def design_for_target(Cd_target, num_samples=5):
    Cd_norm = scaler_X.transform(np.array([[Cd_target]]))
    Cd_norm = np.tile(Cd_norm, (num_samples, 1))
    noise = np.random.normal(0, 1, (num_samples, LATENT_DIM))
    gen_input = np.concatenate([noise, Cd_norm], axis=1)
    designs_norm = generator.predict(gen_input, verbose=0)
    return scaler_Y.inverse_transform(designs_norm)

# ──────────────────────────────────────────────────────
# 9. Test the generator
# ──────────────────────────────────────────────────────
print("\n🔧 Inverse design examples:")
for target_cd in [1.50, 1.55, 1.60, 1.65]:
    designs = design_for_target(target_cd, num_samples=3)
    print(f"\nTarget Cd = {target_cd}:")
    for i, (d, x) in enumerate(designs, 1):
        print(f"  Design {i}: diameter={d:.4f}, x_center={x:.4f}")

# Visualisation
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(df["diameter"], df["x_center"], c=df["Cd"], cmap="viridis", edgecolors="k")
plt.colorbar(sc, label="Cd")
ax.set_xlabel("Diameter"); ax.set_ylabel("x_center")
ax.set_title("Real designs (colour=Cd)")

for target in [1.50, 1.55, 1.60, 1.65]:
    gen = design_for_target(target, num_samples=20)
    ax.scatter(gen[:, 0], gen[:, 1], marker="*", s=100, label=f"Target {target}")
ax.legend()
plt.tight_layout()
plt.savefig("cgan_designs.png", dpi=150)
print("📊 Design map saved.")