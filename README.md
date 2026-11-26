
# Machine Learning Classification Project

Master's-level Machine Learning course — Université Paris Dauphine–PSL, January 2026.

**Authors:** El Ghali Debbagh and Adrien Bouisson

## Objective

Predict a binary target (B/G) using socioeconomic, demographic, employment, retirement and geographical data.

The dataset comprises 100,092 individuals:
- 50,046 labelled observations.
- 50,046 unlabelled observations.

## Data preparation

- Integration of multiple individual and geographical datasets.
- Validation of joins, unique identifiers and data consistency.
- Feature engineering using occupational classifications.
- Handling of missing numerical values through median imputation and missingness indicators.
- Dedicated categories for missing categorical values.
- One-hot encoding of categorical variables for Random Forest.
- Native handling of numerical missing values with CatBoost.

## Methodology

- Stratified 80/20 training-validation split.
- Three-fold stratified cross-validation.
- Comparison of Dummy Classifier, Random Forest and CatBoost.
- Hyperparameter selection based on ROC-AUC.
- Evaluation on a held-out validation sample.

## Results

| Model | Accuracy | ROC-AUC | F1-score |
|---|---:|---:|---:|
| Dummy Classifier | 53.21% | 0.5000 | 0.0000 |
| Random Forest | 70.82% | 0.7772 | 0.6628 |
| CatBoost | 74.00% | 0.8105 | 0.7017 |

CatBoost achieves the best validation performance and is selected for final predictions.

## Repository structure

- `01_prepare.py`: data integration and preparation.
- `02_train.py`: model training, selection and evaluation.
- `03_predict.py`: final predictions.
- `pipeline.py`: preprocessing and model configuration.
- `results/`: cross-validation and validation results.
- `requirements.txt`: Python dependencies.

## Reproducibility

With the original course datasets placed in `data/`:

```bash
pip install -r requirements.txt
python 01_prepare.py
python 02_train.py
python 03_predict.py
```

Individual-level datasets are not publicly distributed. Results refer to the current implementation and its validation protocol.

## Tools

Python, Pandas, NumPy, Scikit-learn, CatBoost and Git.
