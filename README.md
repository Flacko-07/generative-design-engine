# Generative Design Engine

Lightweight inverse-design workspace for cylinder-flow geometry generation and validation.

This repository intentionally keeps only the final source scripts, CSV datasets, plots, and the static Vercel showcase. OpenFOAM case folders, virtual environments, trained binary model files, and cache artifacts are ignored so GitHub stays small and readable.

## Contents

- `*.py` - training, validation, surrogate, and case-generation scripts.
- `*.csv` - final datasets and validation summaries.
- `*.png` - training, generation, and validation plots.
- `index.html`, `styles.css`, `app.js` - static Vercel dashboard for the project.

## Deploy

Vercel can deploy this as a static site from the repository root.

## Fast Inference

Use the lightweight inference script locally:

```bash
python3 inference.py --cd 1.10 --regime 1 --count 5
```

The deployed Vercel app exposes the same idea at:

```text
/api/generate?cd=1.10&regime=1&count=5
```

`regime=0` is laminar and `regime=1` is turbulent.
