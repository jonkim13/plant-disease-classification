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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Setup

This project uses the [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

### Option A — Kaggle API

1. Sign in at [kaggle.com](https://www.kaggle.com/).
2. Go to your account settings
3. Scroll to the **API** section and click **Create New Token**.
4. Move `kaggle.json` to the location Kaggle expects:
   - **macOS / Linux:** `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   - **Windows:** `mkdir $env:USERPROFILE\.kaggle; Move-Item $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\`
5. Run `python main.py` — the dataset will download automatically on first run.

## Usage

Run the full pipeline with defaults:

```bash
python main.py
```

The script prints per-model accuracy, per-class classification reports, the best model's name, and writes three plots to `results/`:

- `cnn_training_curves.png`
- `accuracy_comparison.png`
- `confusion_matrix.png`
