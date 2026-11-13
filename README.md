# Mental Health Prediction System

A local educational classification project with a FastAPI interface, model comparison, held-out evaluation, and downloadable PDF reports.

## What the prediction means

The available target is `treatment`: whether a historical survey respondent reported seeking mental health treatment. `Yes` and `No` refer to that survey answer. This dataset does not provide a clinical diagnosis or a validated future mental-health outcome. The project cannot determine whether someone needs care. Probabilities are uncalibrated model scores, not clinical confidence. Do not use it for screening or consequential decisions.

## Run on Windows PowerShell

Tested with Python 3.14. Use the same environment for training and serving.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 (API documentation: `/docs`). A trained artifact is included. If dependencies or data change, retrain before serving:

```powershell
python train_model.py
python -m unittest test_application -v
```

## Classification experiment

- Strip surrounding whitespace and normalize blank cells to missing values.
- Remove exact duplicate source records and exclude missing/invalid target labels.
- Use the 11 survey fields listed in `train_model.py`. Exclude timestamp, country, gender, interview, and care-options fields from predictors.
- Reserve approximately 20% of records using a stratified grouped split. All records with a shared timestamp stay together. Timestamps may identify repeated source survey records, but are only a proxy; verified respondent IDs are unavailable. This prevents timestamp overlap, not every possible source of leakage. Identical categorical feature combinations can legitimately occur across groups.
- Compare a prior baseline, logistic regression, Bernoulli naive Bayes, random forest, and extra trees using three-fold grouped cross-validation on training data only. All imputation and encoding are fitted within each fold.
- Tree ensembles use 60 trees, depth at most 10, and a 10% bootstrap sample per tree to bound training cost on the repeated survey data.
- Select the highest mean macro F1. Evaluate the selected model once on the reserved test set. The saved pipeline is fitted only on training rows, so its test results match the deployed artifact.
- Save per-class precision/recall/F1, confusion matrix, accuracy, weighted/macro F1, ROC AUC, and Brier score. The dashboard also displays CV comparisons and split limitations.

This is a bounded comparison of fixed configurations, not an exhaustive algorithm or hyperparameter search. A stronger result requires verified respondent provenance and an independent dataset. Do not interpret a high internal score as clinical validity.

## Saved outputs

`models/mental_health_model.joblib` contains the pipeline, schema, category options, metrics, and package versions. `models/model_metrics.json` contains the same metrics. `models/environment.json` records numerical-library versions. Training updates all three. Requirements pin the tested runtime; serving rejects a scikit-learn mismatch before loading the artifact. Load only trusted Joblib files.

## API and data handling

- `GET /health`, `/api/options`, `/api/stats`, `/api/data-preview?limit=8`
- `GET /dataset` downloads the included source CSV.
- `POST /api/predict` accepts exactly the 11 required category strings and returns class scores.
- `POST /api/report` accepts the same inputs and streams a generated PDF. There are no report identifiers or report-download routes.

Unknown categories, empty answers, non-string values, missing fields, and extra fields are rejected with HTTP 422. Choose `Not specified` to use the training-fitted most-frequent imputation. Reports are generated in memory only when requested and responses use `Cache-Control: no-store`. The application does not save submitted survey answers. Any PDFs left in `reports/` from the previous version remain local legacy files and are no longer served.

Run on localhost with demo responses. Public deployment would require deployment-specific authentication, TLS, request limits, and operational review. Frontend form, statistics, and preview load independently; dataset cells use text rendering. Changing a form response invalidates the displayed result.

## Checks

`test_application.py` uses the standard library to start a local test server and exercise page/static loading, valid prediction, invalid input, actual missing-value imputation, PDF download without disk storage, removed report URLs, metrics consistency, and dataset endpoints. No additional test dependency is needed.
