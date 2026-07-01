# MedPredict AI — Multi-Disease Risk Assessment

![MedPredict AI](https://img.shields.io/badge/MedPredict-AI-6366f1?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-f7931e?style=flat-square&logo=scikit-learn)

An advanced AI-powered clinical decision support system that predicts patient risk for **Diabetes**, **Heart Disease**, and **Liver Disease** using 19 clinical biomarkers.

## Live Demo

> Deployed on Vercel: `https://your-project.vercel.app`

---

## Features

- **3 Disease Models**: Diabetes, Heart Disease, Liver Disease
- **19 Clinical Inputs**: Demographics, vitals, lipid panel, liver enzymes, lifestyle factors
- **Real-time Risk Bars**: Animated probability bars with clinical color-coding
- **Feature Explanations**: Top 3 contributing biomarkers per disease
- **PDF Report**: Downloadable patient assessment report
- **AI Model**: Calibrated Ensemble (Random Forest + Gradient Boosting + Logistic Regression)

## Model Performance

| Disease | AUC | Recall | F1 |
|---|---|---|---|
| Diabetes | 0.9695 | 89.6% | 0.925 |
| Heart Disease | 0.9419 | 93.0% | 0.903 |
| Liver Disease | 0.9124 | 80.3% | 0.820 |

---

## Project Structure

```
detecting-system/
├── api/
│   └── main.py              # FastAPI backend (Vercel serverless)
├── src/
│   ├── predict.py           # Prediction pipeline
│   ├── train.py             # Model training script
│   ├── preprocessing.py     # Data preprocessing (RobustScaler)
│   └── feature_engineering.py
├── models/
│   ├── model_Diabetes.joblib
│   ├── model_Heart_Disease.joblib
│   ├── model_Liver_Disease.joblib
│   ├── preprocessor.joblib
│   └── feature_engineer.joblib
├── data/
│   └── generate_mock_data.py   # Synthetic data generator (v6)
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── vercel.json                  # Vercel deployment config
├── requirements.txt             # Python dependencies
└── run_project.bat              # Windows local launcher
```

---

## Running Locally

### Prerequisites
- Python 3.10+
- pip

### Setup & Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate training data and retrain models
python data/generate_mock_data.py
python src/train.py

# 3. Start both servers (Windows)
run_project.bat

# Or manually:
# Terminal 1 — API Backend
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Frontend
python -m http.server 8080 --directory frontend
```

Open: http://localhost:8080

---

## Deploying to Vercel

### Method 1: Vercel CLI
```bash
npm install -g vercel
vercel login
vercel --prod
```

### Method 2: Vercel Dashboard
1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → New Project
3. Import your GitHub repository
4. No extra configuration needed (vercel.json handles everything)
5. Click Deploy

---

## Deploying to GitHub

```bash
git init
git add .
git commit -m "Initial commit: MedPredict AI v3.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/medpredict-ai.git
git push -u origin main
```

> **Note**: The trained model files (`models/*.joblib`) are included in the repository and are required for inference. They use maximum joblib compression (level 9) to minimize file size.

---

## Clinical Disclaimer

> ⚠️ MedPredict AI is for **research and educational purposes only**. It does not constitute medical advice or diagnosis. Always consult a qualified healthcare professional for medical decisions.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JS |
| Backend | Python 3.10, FastAPI |
| ML Models | scikit-learn (RF + GBC + LR ensemble) |
| Calibration | Isotonic Regression (CalibratedClassifierCV) |
| Deployment | Vercel (frontend + API) |
| Version Control | GitHub |
