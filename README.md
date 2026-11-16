# Mental Health Prediction System

A local educational classification project with a FastAPI interface, model comparison, held-out evaluation, and downloadable PDF reports.

## What the prediction means

The available target is `treatment`: whether a historical survey respondent reported seeking mental health treatment. `Yes` and `No` refer to that survey answer. This dataset does not provide a clinical diagnosis or a validated future mental-health outcome. The project cannot determine whether someone needs care. Probabilities are uncalibrated model scores, not clinical confidence. Do not use it for screening or consequential decisions.

## Run on Windows PowerShell

Tested with Python 3.14. Use the same environment for training and serving.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 (API documentation: `/docs`). A trained artifact is included. If dependencies or data change, retrain before serving:

```powershell
python backend/train_model.py
python -m unittest backend.test_application -v
```

## Classification experiment

- Strip surrounding whitespace and normalize blank cells to missing values.
- Remove exact duplicate source records and exclude missing/invalid target labels.
- Use the 11 survey fields listed in `backend/train_model.py`. Exclude timestamp, country, gender, interview, and care-options fields from predictors.
- Reserve approximately 20% of records using a stratified grouped split. All records with a shared timestamp stay together. Timestamps may identify repeated source survey records, but are only a proxy; verified respondent IDs are unavailable. This prevents timestamp overlap, not every possible source of leakage. Identical categorical feature combinations can legitimately occur across groups.
- Compare a prior baseline, logistic regression, Bernoulli naive Bayes, random forest, and extra trees using three-fold grouped cross-validation on training data only. All imputation and encoding are fitted within each fold.
- Tree ensembles use 60 trees, depth at most 10, and a 10% bootstrap sample per tree to bound training cost on the repeated survey data.
- Select the highest mean macro F1. Evaluate the selected model once on the reserved test set. The saved pipeline is fitted only on training rows, so its test results match the deployed artifact.
- Save per-class precision/recall/F1, confusion matrix, accuracy, weighted/macro F1, ROC AUC, and Brier score. The dashboard also displays CV comparisons and split limitations.

This is a bounded comparison of fixed configurations, not an exhaustive algorithm or hyperparameter search. A stronger result requires verified respondent provenance and an independent dataset. Do not interpret a high internal score as clinical validity.

## Saved outputs

`backend/models/mental_health_model.joblib` contains the pipeline, schema, category options, metrics, and package versions. `backend/models/model_metrics.json` contains the same metrics. `backend/models/environment.json` records numerical-library versions. Training updates all three. Requirements pin the tested runtime; serving rejects a scikit-learn mismatch before loading the artifact. Load only trusted Joblib files.

## API and data handling

### Knowledge Assistant

Gemini timeouts, API errors, and missing API keys fall back to local TF-IDF search of the bundled guide. These responses are explicitly labelled **Guide search mode** and quote a matching passage with a page/entry citation; they are not generated answers. Out-of-scope questions return no passage. After a provider error, cloud calls pause for 60 seconds. Set `RAG_MODE=guide_search` in the backend environment to use this mode immediately without provider calls. Restart after changing environment variables.

Use the floating **AI** button on the dashboard to open the assistant in a panel along the right edge. The panel fills the screen height and uses the available width on small screens. Close it with the **×** button, Escape, or a click outside the panel. Closing and reopening preserves the current chat while the dashboard remains open. The standalone `/rag` page remains available.

The supplied eight-page `Stress_and_Mental_Health_RAG_Knowledge_Base.pdf` is bundled in `backend/knowledge_sources/`. Once Gemini is configured, opening `/rag` or asking a question indexes its 24 Q&A entries automatically. A content/model hash prevents re-indexing unchanged files across restarts. Answers can cite the PDF page and KB entry. The introductory setup page and source directory are excluded from answer retrieval. Visitors can use the shared guide without uploading files, and it cannot be deleted from the public document list. The source is supplied educational material, not independently clinically validated content.

Click **Knowledge Assistant** in the sidebar, or visit `/rag` directly. The page is separate from the dashboard. Install `backend/requirements.txt`, copy `backend/.env.example` to `backend/.env`, supply `GEMINI_API_KEY`, and restart the backend. Without a key the page remains usable through labelled local PDF passages. For EC2, use the server-side environment file described in `deploy/README.md`; a local `.env` is deliberately excluded from Docker images. Document uploads and retrieved context are sent to Gemini for embedding and generation. Use educational demo documents, not personal records.

### Dashboard and batch features

- **Fill sample responses** populates the form with synthetic allowed-category values, which you can edit before submitting.
- Dashboard charts show CV macro F1, source treatment-response distribution, and the held-out confusion matrix. Labels and values remain available as text.
- Download the sample CSV from the batch panel, enter up to 1,000 survey rows, then upload it to download predictions and class probabilities. UTF-8 files only; maximum 1 MB. Use exactly the 11 template columns. Blank cells use training-fitted imputation; invalid categories or malformed rows reject the entire upload with row-specific errors. CSV files and results are processed in memory and are not stored.
- `GET /api/batch-template` downloads the template; `POST /api/predict-batch` accepts raw CSV with `Content-Type: text/csv` and returns a CSV result. The scores describe treatment-seeking responses and have no clinical interpretation.

- `GET /health`, `/api/options`, `/api/stats`, `/api/data-preview?limit=8`
- `GET /dataset` downloads the included source CSV.
- `POST /api/predict` accepts exactly the 11 required category strings and returns class scores.
- `POST /api/report` accepts the same inputs and streams a generated PDF. There are no report identifiers or report-download routes.

Unknown categories, empty answers, non-string values, missing fields, and extra fields are rejected with HTTP 422. Choose `Not specified` to use the training-fitted most-frequent imputation. Reports are generated in memory only when requested and responses use `Cache-Control: no-store`. The application does not save submitted survey answers. Any PDFs left in `backend/reports/` from the previous version remain local legacy files and are no longer served.

Run on localhost with demo responses. Public deployment would require deployment-specific authentication, TLS, request limits, and operational review. Frontend form, statistics, and preview load independently; dataset cells use text rendering. Changing a form response invalidates the displayed result.

## Checks

`backend/test_application.py` uses the standard library to start a local test server and exercise page/static loading, valid prediction, invalid input, actual missing-value imputation, PDF download without disk storage, removed report URLs, metrics consistency, and dataset endpoints. No additional test dependency is needed.

### GitHub Actions CI/CD

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`, and can also be started manually from the Actions tab. It uses Python 3.14 on Ubuntu and Windows, caches pip downloads, installs `backend/requirements.txt`, checks dependency compatibility and Python syntax, and runs the integration tests against the committed model. Tests serve the files in `frontend/` through FastAPI. Full model training is run separately with `python backend/train_model.py`. After both test jobs pass on `main`, a deployment job uploads the tested commit to EC2 using the SSH secrets and starts a persistent Docker container on HTTP port 80. See [EC2 deployment setup](deploy/README.md) for required server preparation, security-group rules, and troubleshooting.

## Project layout

Run the commands above from the project root. FastAPI serves the frontend and API together at http://127.0.0.1:8000; no separate frontend server or build step is needed.

```text
backend/
  app/main.py             # FastAPI routes and PDF generation
  train_model.py          # Classification training workflow
  requirements.txt        # Python dependencies
  test_application.py     # Integration tests
  data/                   # Source dataset
  models/                 # Trained pipeline and evaluation
frontend/
  index.html              # Web interface
  static/app.js           # Form, API requests, and dashboard
  static/style.css        # Interface styles
```
