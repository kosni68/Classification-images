# Classification-images

.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
python classify_images.py --config config.json

Notes:
- Set `two_stage` in `config.json` (or use `--no-two-stage`) to toggle the coarse/fine pass.
- Set `top_k` in `config.json` (or use `--top-k`) to add extra columns in `predictions.csv`.
