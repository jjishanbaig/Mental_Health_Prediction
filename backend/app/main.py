from pathlib import Path
from uuid import uuid4
from io import BytesIO, StringIO
import csv
from fastapi.responses import Response
from pydantic import ConfigDict, field_validator
import numpy as np
import sklearn
import logging
from functools import lru_cache
from datetime import datetime, timezone
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, ValidationError
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT.parent / "frontend"
DATA_PATH = ROOT / "data" / "Mental Health Dataset.csv"
MODEL_PATH = ROOT / "models" / "mental_health_model.joblib"
environment_path = ROOT / "models" / "environment.json"
if not MODEL_PATH.exists() or not environment_path.exists():
    raise RuntimeError("Model missing. Run python backend/train_model.py first.")
environment = json.loads(environment_path.read_text(encoding="utf-8"))
if environment.get("scikit-learn") != sklearn.__version__:
    raise RuntimeError("Model version mismatch. Install backend/requirements.txt or retrain the model.")
artifact = joblib.load(MODEL_PATH)
if artifact.get("versions", {}).get("scikit-learn") != sklearn.__version__:
    raise RuntimeError("Model version mismatch. Run python backend/train_model.py in this environment.")
model = artifact["pipeline"]
FEATURES = artifact["features"]
METRICS = artifact["metrics"]
data = pd.read_csv(DATA_PATH)

app = FastAPI(title="Mental Health Prediction System", version="2.0.0",
              description="Educational ML demonstration for survey response patterns.")
app.mount("/static", StaticFiles(directory=str(FRONTEND / "static")), name="static")
templates = Jinja2Templates(directory=str(FRONTEND))

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def validate_category(cls, value, info):
        if not isinstance(value, str):
            raise ValueError("Select a survey response")
        value = value.strip()
        if value != "Not specified" and value not in artifact["options"].get(info.field_name, []):
            raise ValueError("Unknown survey response")
        return value

    Occupation: str
    self_employed: str
    family_history: str
    Days_Indoors: str
    Growing_Stress: str
    Changes_Habits: str
    Mental_Health_History: str
    Mood_Swings: str
    Coping_Struggles: str
    Work_Interest: str
    Social_Weakness: str

def make_pdf(report_id: str, payload: dict, prediction: str, confidence: float, probabilities: dict):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8,
                               leading=11, textColor=colors.HexColor("#555555")))
    story = [
        Paragraph("Mental Health Survey Prediction Report", styles["Title"]),
        Spacer(1, 5*mm),
        Paragraph(f"Report ID: {report_id}", styles["Normal"]),
        Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]),
        Spacer(1, 7*mm),
        Paragraph("Model output", styles["Heading2"]),
        Paragraph("Target: treatment. Yes means the respondent reported seeking treatment; "
                  "No means they reported not seeking it. Neither answer determines need for care.", styles["BodyText"]),
        Paragraph(f"Classifier: {METRICS['selected_model']}", styles["BodyText"]),
        Paragraph(f"<b>Predicted survey response:</b> {prediction}", styles["BodyText"]),
        Paragraph(f"<b>Model probability for this predicted class:</b> {confidence:.1%}", styles["BodyText"]),
        Spacer(1, 3*mm),
        Paragraph("Class probabilities", styles["Heading3"])
    ]
    prob_rows = [["Class", "Model probability"]] + [[k, f"{v:.1%}"] for k, v in probabilities.items()]
    t = Table(prob_rows, colWidths=[75*mm, 75*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24324a")),
                           ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                           ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d8dee9")),
                           ("PADDING",(0,0),(-1,-1),7)]))
    story += [t, Spacer(1, 7*mm), Paragraph("Submitted survey responses", styles["Heading2"])]
    rows = [["Field", "Value"]] + [[k, str(v)] for k, v in payload.items()]
    t2 = Table(rows, colWidths=[75*mm, 75*mm], repeatRows=1)
    t2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24324a")),
                            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#d8dee9")),
                            ("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),6)]))
    story += [t2, Spacer(1, 8*mm),
              Paragraph("<b>Important limitation:</b> This is an educational machine-learning demonstration "
                        "trained on survey data. It predicts a historical survey response category, not a "
                        "mental health condition, diagnosis, risk level, or need for care. The probability is "
                        "a model score and is not a clinical confidence measure. Do not use this output for "
                        "screening, treatment, employment, insurance, or other consequential decisions.",
                        styles["SmallNote"])]
    doc.build(story)
    return buffer.getvalue()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.get("/api/options")
def options():
    return {key: values + ["Not specified"] for key, values in artifact["options"].items()}

@app.get("/api/stats")
@lru_cache(maxsize=1)
def stats():
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "target_distribution": data["treatment"].value_counts(dropna=False).to_dict(),
        "metrics": METRICS
    }

@app.get("/api/data-preview")
def data_preview(limit: int = 12):
    limit = max(1, min(limit, 100))
    preview = data.head(limit).fillna("").to_dict(orient="records")
    return {"columns": list(data.columns), "rows": preview, "total_rows": int(len(data))}

@app.get("/dataset")
def download_dataset():
    return FileResponse(DATA_PATH, filename="Mental_Health_Dataset.csv", media_type="text/csv")

def infer(payload):
    values = payload.model_dump()
    frame = pd.DataFrame([{k: np.nan if v == "Not specified" else v for k,v in values.items()}], columns=FEATURES)
    raw = model.predict_proba(frame)[0]
    probabilities = {str(c):float(p) for c,p in zip(model.classes_,raw)}
    prediction = max(probabilities, key=probabilities.get)
    return values, prediction, probabilities[prediction], probabilities

@app.post("/api/predict")
def predict(payload: PredictionInput):
    values, prediction, confidence, probabilities = infer(payload)
    return {"prediction":prediction, "confidence":confidence, "probabilities":probabilities,
            "target":"treatment", "model":METRICS["selected_model"]}

@app.post("/api/report")
def report(payload: PredictionInput):
    values, prediction, confidence, probabilities = infer(payload)
    try:
        content = make_pdf(uuid4().hex, values, prediction, confidence, probabilities)
    except Exception:
        logging.exception("PDF generation failed")
        raise HTTPException(status_code=500, detail="Report generation failed")
    return Response(content, media_type="application/pdf", headers={
        "Content-Disposition":"attachment; filename=mental_health_survey_report.pdf",
        "Cache-Control":"no-store"})


@app.get("/api/batch-template")
def batch_template():
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(FEATURES)
    # Synthetic example, assembled from allowed categories rather than a person's record.
    writer.writerow([artifact["options"][key][0] for key in FEATURES])
    return Response(output.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=survey_template.csv"})


@app.post("/api/predict-batch")
async def predict_batch(request: Request):
    if request.headers.get("content-type", "").split(";")[0].strip() != "text/csv":
        raise HTTPException(415, "Send the CSV file with Content-Type: text/csv")
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > 1024 * 1024:
            raise HTTPException(413, "CSV must be no larger than 1 MB")
    try:
        reader = csv.DictReader(StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        if not reader.fieldnames or len(reader.fieldnames) != len(FEATURES) or set(reader.fieldnames) != set(FEATURES):
            raise HTTPException(422, "CSV headers must match the 11 template columns exactly")
        rows, errors = [], []
        for number, row in enumerate(reader, start=2):
            if number > 1001:
                raise HTTPException(413, "CSV supports at most 1,000 survey rows")
            if None in row or any(value is None for value in row.values()):
                errors.append(f"Row {number}: wrong number of columns")
                continue
            try:
                cleaned = {key: value.strip() or "Not specified" for key, value in row.items()}
                rows.append(PredictionInput(**cleaned).model_dump())
            except ValidationError as exc:
                fields = ", ".join(str(error["loc"][0]) for error in exc.errors())
                errors.append(f"Row {number}: invalid response in {fields}")
        if errors:
            raise HTTPException(422, "; ".join(errors[:20]))
        if not rows:
            raise HTTPException(422, "CSV must contain at least one survey row")
    except (UnicodeDecodeError, csv.Error):
        raise HTTPException(422, "Upload a valid UTF-8 CSV file")
    frame = pd.DataFrame(rows, columns=FEATURES).replace("Not specified", np.nan)
    probabilities = model.predict_proba(frame)
    classes = model.classes_.tolist()
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["row_number", *FEATURES, "predicted_treatment", "predicted_probability", *[f"probability_{c}" for c in classes]])
    for number, (row, scores) in enumerate(zip(rows, probabilities), start=2):
        selected = int(np.argmax(scores))
        writer.writerow([number, *[row[k] for k in FEATURES], classes[selected],
                         float(scores[selected]), *[float(score) for score in scores]])
    return Response(output.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=survey_predictions.csv",
        "X-Prediction-Count": str(len(rows))})

@app.middleware("http")
async def privacy_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


from .rag_routes import router as rag_router

app.include_router(rag_router)
@app.get("/rag", response_class=HTMLResponse)
async def rag_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="rag.html",
        context={"embedded": request.query_params.get("embedded") == "1"}
    )
