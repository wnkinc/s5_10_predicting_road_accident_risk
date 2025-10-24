Overview
This repository organizes notebooks and artifacts for Kaggle Playground Series Season 5 Episode 10, which predicts the continuous accident_risk target from tabular features such as road type, lane count, curvature, speed limit, lighting, weather, and reported accidents.

Data assets
The data/ directory includes the official competition train.csv/test.csv files alongside several synthetic accident datasets that expand the feature distribution for experimentation.

Guidance in s5e10-how-to-improve-your-cv-score.ipynb explains loading both competition and “original” external data to enrich training folds when searching for better cross-validation performance.

Modeling notebooks
baseline/s5e10-lgbm-origcol-20seeds.ipynb configures a LightGBM regressor with deep trees, regularization, and 100k boosting rounds for repeated cross-validation seeding experiments.

baseline/s5e10-xgb-origcol-20seeds.ipynb trains a GPU-accelerated XGBRegressor with tuned subsampling, depth, and early stopping inside a five-fold loop, logging fold-wise RMSE.

baseline/s5e10-realmlp-tuned.ipynb relies on pytabkit’s RealMLP_TD_Regressor, specifying extensive architectural and optimization hyperparameters before running five-fold training and exporting both OOF and test predictions.

baseline/s5e10-tabm-over-residuals.ipynb reimplements Chris Deotte’s residual boosting idea but swaps XGBoost for the full TabM model, comparing it with the lighter tabm-mini variant.

Ensembling workflow
s5e10-nn-stacking-baseline.ipynb documents a stacking approach that feeds stored OOF/test predictions from Level-1 models (LightGBM, CatBoost, XGBoost, TabM, etc.) into a neural-network meta-learner, emphasizing the need for consistent five-fold CV splits to avoid leakage.

The oof_test/ directory holds the exported OOF and test prediction files that serve as Level-1 inputs for the stacking notebook.

Outputs
submission.csv provides the repository’s ready-to-upload prediction file pairing each test id with its inferred accident risk.
