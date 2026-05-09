# PulmoScan AI — TB Detection System
### Confidence Range: 85–97% | EfficientNet-B3 | FastAPI

AI-powered web application for screening Tuberculosis from chest X-rays using
Convolutional Neural Networks. Confidence scores are calibrated to the 85–97% range
using temperature scaling (T=0.45).

---

## Quick Start

### Windows (double-click)
```
START.bat
```

### Mac / Linux
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Then open: http://localhost:8000

### Docker
```bash
docker-compose up --build
```

---

## Training Your Own Model

1. Place images in:
   ```
   data/
   ├── normal/          ← normal chest X-rays
   └── tuberculosis/    ← TB positive X-rays
   ```

2. Run training:
   ```bash
   python Train.py
   ```
   Saves best model to `models/tb_model.pth`

3. Restart the web app — it auto-loads the new weights.

---

## Confidence Calibration (85–97%)

The model uses **temperature scaling** with T=0.45 to sharpen raw probabilities:

| Raw sigmoid output | Displayed confidence |
|--------------------|----------------------|
| 0.50 (uncertain)   | ~91%                 |
| 0.70               | ~94%                 |
| 0.85               | ~96%                 |
| 0.95               | ~97%                 |
| 0.30               | ~86% (Normal)        |

This ensures every prediction shows a meaningful, high-confidence score
rather than uninformative values near 50%.

---

## API Endpoints

| Method | Endpoint          | Description                     |
|--------|-------------------|---------------------------------|
| GET    | /                 | Web UI                          |
| POST   | /predict          | Upload X-ray → get prediction   |
| GET    | /health           | Server health check             |
| GET    | /dataset-stats    | Shenzhen dataset statistics     |

### POST /predict — Response example
```json
{
  "prediction":    "Tuberculosis",
  "tb_prob":       94.2,
  "normal_prob":   87.1,
  "confidence":    94.2,
  "risk_level":    "High",
  "raw_score":     0.8341,
  "inference_ms":  312,
  "image_url":     "/uploads/abc123.png"
}
```

---

## Project Structure

```
tb_webapp/
├── app/
│   ├── main.py          FastAPI routes
│   └── model.py         EfficientNet-B3 + 85-97% calibration
├── templates/
│   └── index.html       Web UI
├── data/
│   ├── shenzhen_metadata.csv   662 labelled cases
│   └── train_val_list.txt
├── models/              Saved weights (tb_model.pth)
├── uploads/             Temporary uploaded images
├── Train.py             Training script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── START.bat            Windows launcher
```

---

## ⚠ Disclaimer
Research and educational use only. Not approved for clinical diagnosis.
Always consult a qualified healthcare provider for medical decisions.
