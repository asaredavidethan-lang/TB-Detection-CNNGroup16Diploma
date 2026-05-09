"""
app/main.py  —  PulmoScan FastAPI backend
"""
import os, io, uuid, time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model import predict

BASE       = Path(__file__).parent.parent
app        = FastAPI(title="PulmoScan TB Detection", version="2.0.0")
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static",  StaticFiles(directory=BASE / "static"),  name="static")
app.mount("/uploads", StaticFiles(directory=BASE / "uploads"), name="uploads")
templates  = Jinja2Templates(directory=str(BASE / "templates"))

ALLOWED = {".png", ".jpg", ".jpeg", ".dcm"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "model": "EfficientNet-B3", "version": "2.0.0"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, "Unsupported file. Upload PNG, JPG, or DICOM (.dcm).")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large. Max 20 MB.")

    # DICOM → PNG conversion
    if ext == ".dcm":
        try:
            import pydicom
            import numpy as np
            from PIL import Image
            ds  = pydicom.dcmread(io.BytesIO(contents))
            arr = ds.pixel_array.astype(float)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).convert("RGB").save(buf, format="PNG")
            contents = buf.getvalue()
        except Exception as e:
            raise HTTPException(400, f"DICOM read error: {e}")

    fname = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / fname).write_bytes(contents)

    t0     = time.perf_counter()
    result = predict(contents)
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000)
    result["image_url"]    = f"/uploads/{fname}"
    return JSONResponse(result)


@app.get("/dataset-stats")
async def dataset_stats():
    import pandas as pd
    csv = BASE / "data" / "shenzhen_metadata.csv"
    if not csv.exists():
        return {"total": 662, "tb": 336, "normal": 326, "avg_age": 35.4}
    df      = pd.read_csv(csv)
    total   = len(df)
    normal  = int((df["findings"] == "normal").sum())
    return {
        "total":   total,
        "tb":      total - normal,
        "normal":  normal,
        "avg_age": round(float(df["age"].mean()), 1),
        "male":    int((df["sex"] == "Male").sum()),
        "female":  int((df["sex"] == "Female").sum()),
    }
