# Predicting Bulldozer Sale Prices

An end-to-end machine learning project that predicts the **sale price of a bulldozer** from its characteristics (year made, model, hours of use, state, etc.) using Random Forest regression.

Data and the evaluation metric (Root Mean Squared Log Error — RMSLE) come from the [Kaggle Bluebook for Bulldozers competition](https://www.kaggle.com/c/bluebook-for-bulldozers/overview).

## What's in the notebook

1. **Load and explore** the raw data (~400k bulldozer sales 1989–2011)
2. **Preprocessing** — parse `saledate` into year/month/day features, handle missing values, ordinal-encode categorical columns
3. **Train/validation split** by date (we predict the future, so the validation set must come from later sales than the training set)
4. **Train** a `RandomForestRegressor` baseline and evaluate it with MAE, RMSLE, and R²
5. **Tune** hyperparameters with `RandomizedSearchCV`
6. **Save** the best model with `joblib`
7. **Predict** on the test set and on a custom hand-built sample
8. **Inspect feature importance** to see what actually drives the price

## Project Structure

```
bluebook-bulldozer-project/
├── end-to-end-bluebook-bulldozer-price-regression-v2.ipynb   # the analysis
├── data/
│   └── bluebook-for-bulldozers.zip                           # dataset (unzip before running)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> **Note:** The raw dataset (`bluebook-for-bulldozers/`, ~600 MB unzipped) and the trained model (`randomforest_regressor_best_RMSLE.pkl`, ~1 GB) are **not** committed to the repo because of GitHub's per-file size limit. The zip is included; unzip it before running the notebook (see below). The model is regenerated when the notebook runs.

## Getting Started

### Clone the repo

```bash
git clone https://github.com/SeyedMani/bluebook-bulldozer-project.git
cd bluebook-bulldozer-project
```

### Set up a virtual environment

Using `conda`:

```bash
conda create -n bluebook python=3.11 -y
conda activate bluebook
pip install -r requirements.txt
```

Or using `venv`:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Unzip the dataset

```bash
cd data
unzip bluebook-for-bulldozers.zip
cd ..
```

This creates `data/bluebook-for-bulldozers/` with all the CSVs.

### Run the notebook

```bash
jupyter notebook end-to-end-bluebook-bulldozer-price-regression-v2.ipynb
```

Run the cells top-to-bottom. The full run takes a few minutes because the dataset is large and we train multiple Random Forests.

## Notes

- A few cells in the notebook are tagged `raises-exception` on purpose — they demonstrate what happens when you try to fit a model on un-preprocessed data, or call methods that fail before preprocessing. The notebook still executes end-to-end with `nbconvert --execute`.
- The project is inspired by the [`mrdbourke/zero-to-mastery-ml`](https://github.com/mrdbourke/zero-to-mastery-ml) course, but the notebook has been rewritten with my own explanations and bug-fixed for newer versions of pandas and scikit-learn.

## License

MIT — see [LICENSE](LICENSE).
