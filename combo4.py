import pandas as pd

# Load existing datasets
lam = pd.read_csv("cylinder_dataset_large.csv")      # 400 laminar, fixed H=2.0, U=0.001
turb_old = pd.read_csv("turbulent_dataset.csv")       # 392 turbulent, fixed H=2.0
turb_new = pd.read_csv("turbulent_4param.csv")        # 500 turbulent, variable H

# Add channel_height and U_inlet to laminar (if not present)
lam["channel_height"] = 2.0
lam["U_inlet"] = 0.001

# Add channel_height to old turbulent (if missing)
if "channel_height" not in turb_old.columns:
    turb_old["channel_height"] = 2.0

# Compute blockage ratio and flow regime
lam["blockage_ratio"] = lam["diameter"] / lam["channel_height"]
lam["flow_regime"] = 0

turb_old["blockage_ratio"] = turb_old["diameter"] / turb_old["channel_height"]
turb_old["flow_regime"] = 1

turb_new["blockage_ratio"] = turb_new["diameter"] / turb_new["channel_height"]
turb_new["flow_regime"] = 1

# Combine
df = pd.concat([lam, turb_old, turb_new], ignore_index=True)
df.to_csv("combined_all.csv", index=False)
print(f"Combined dataset: {len(df)} designs (lam:{len(lam)}, turb_old:{len(turb_old)}, turb_new:{len(turb_new)})")