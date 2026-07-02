---
title: MedPredict AI
emoji: 🏥
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# MedPredict AI — Multi-Disease Risk Assessment

An AI-powered clinical decision support system that predicts patient risk for **Diabetes**, **Heart Disease**, and **Liver Disease** using 19 clinical biomarkers.

## Model Performance

| Disease | AUC | Recall | F1 |
|---|---|---|---|
| Diabetes | 0.9695 | 89.6% | 0.925 |
| Heart Disease | 0.9419 | 93.0% | 0.903 |
| Liver Disease | 0.9124 | 80.3% | 0.820 |

## Tech Stack
- **Frontend**: HTML5, CSS3, Vanilla JS
- **Backend**: Python 3.10, FastAPI
- **ML Models**: scikit-learn ensemble (RF + GBC + LR, calibrated)
- **Deployment**: Hugging Face Spaces (Docker)

> ⚠️ For research and educational purposes only. Not medical advice.
