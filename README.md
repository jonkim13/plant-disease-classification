# Plant Disease Classification

Comparing classical ML baselines and a basic CNN on five tomato-leaf disease classes from the PlantVillage dataset.

**Group 18:** Jonathan Kim and Alex Loya

## Final Results

| Model               | Test Accuracy |
|---------------------|---------------|
| KNN (k=5)           | 61.6%         |
| Logistic Regression | 72.6%         |
| SVM (Linear)        | 73.6%         |
| MLP (1×64)          | 78.6%         |
| SVM (RBF)           | 80.6%         |
| Basic CNN           | 80.8%         |

## Key Findings

The CNN and SVM-RBF essentially tie at ~80% — a small CNN trained on 1,500 images doesn't dominate a tuned classical model the way the headline narrative around deep learning would suggest. Most errors come from Early-blight ↔ Late-blight confusion; these diseases produce similar dark lesions and every model conflates them.

## Project Structure

```
plant-disease-classification/
├── main.py                 # End-to-end orchestrator (runnable)
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py             # Dataset discovery, feature extraction, splits
│   ├── models.py           # Model factory functions
│   ├── train.py            # Training routines (classical + CNN)
│   └── evaluate.py         # Metrics and plotting helpers
└── notebooks/
    └── Plant_Disease_Training_ML.ipynb   # Original exploratory notebook
```

## Setup

```bash
git clone <your-fork-url>
cd plant-disease-classification
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

This project uses the [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
You have two options for getting it:

### Option A — Kaggle API (recommended, fully automatic)

1. Sign in (or sign up) at [kaggle.com](https://www.kaggle.com/).
2. Go to your account settings: click your profile picture (top-right) → **Settings**.
3. Scroll to the **API** section and click **Create New Token**.
   This downloads a file called `kaggle.json` containing your username and key.
4. Move `kaggle.json` to the location Kaggle expects:
   - **macOS / Linux:** `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   - **Windows (PowerShell):** `mkdir $env:USERPROFILE\.kaggle; Move-Item $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\`
5. Run `python main.py` — the dataset will download automatically on first run (~1 GB).

> ⚠️ Never commit `kaggle.json` to a repository. Treat it like a password.
> If you ever accidentally expose it, go back to Kaggle Settings → API → **Expire Token**, then create a new one.

### Option B — Manual download (no API setup)

1. Go to the [PlantVillage dataset page](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
2. Click **Download** (you'll need a free Kaggle account).
3. Unzip the archive.
4. Place the unzipped folder so the structure looks like:

```
      project-root/
      ├── data/
      │   └── plantvillage_dataset/
      │       └── color/
      │           ├── Tomato___Bacterial_spot/
      │           ├── Tomato___Early_blight/
      │           └── ... (38 class folders total)
      └── main.py
```

5. Run `python main.py`. The code will detect the local folder and skip the download.

## Usage

Run the full pipeline with defaults:

```bash
python main.py
```

All flags:

```bash
python main.py \
  --data-dir data/plantvillage_dataset/color \
  --output-dir results \
  --epochs 8 \
  --batch-size 32 \
  --img-size 32 \
  --max-per-class 400
```

| Flag              | Default                              | Description                                       |
|-------------------|--------------------------------------|---------------------------------------------------|
| `--data-dir`      | (autoresolved)                       | Local dataset path; falls back to kagglehub.      |
| `--output-dir`    | `results`                            | Where plots are written.                          |
| `--epochs`        | `8`                                  | CNN training epochs.                              |
| `--batch-size`    | `32`                                 | CNN batch size.                                   |
| `--img-size`      | `32`                                 | Pixel size for the classical feature extractor.   |
| `--max-per-class` | `400`                                | Cap on images sampled from each class.            |

The script prints per-model accuracy, per-class classification reports, the best model's name, and writes three plots to `results/`:

- `cnn_training_curves.png`
- `accuracy_comparison.png`
- `confusion_matrix.png`

## License

[MIT](LICENSE) © 2025 Jonathan Kim and Alex Loya.
