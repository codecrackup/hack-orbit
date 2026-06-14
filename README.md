# 🚀 Hack Orbit

AI Mission Intelligence Copilot for Satellite Operations.

Hack Orbit combines satellite health monitoring, AI-based failure prediction, space weather analysis, orbital debris monitoring, and mission forecasting into a unified operations dashboard.

---

# 🛠️ How to Run

## Backend

```bash
cd backend

pip install -r requirements.txt

uvicorn app.main:app --reload
```

Backend:

```text
http://localhost:8000
```

API Docs:

```text
http://localhost:8000/docs
```

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

Frontend:

```text
http://localhost:3000
```

---

# 📌 Features

### 🛰️ Satellite Health Monitoring
- Health Score Calculation
- Telemetry Analysis
- Operational Status Tracking

### 🤖 AI Failure Prediction
- Machine Learning Based Risk Assessment
- Early Failure Detection
- Risk Categorization

### ☀️ Space Weather Intelligence
- Kp Index Monitoring
- Solar Activity Analysis
- Environmental Risk Assessment

### ☄️ Orbital Debris Monitoring
- Nearby Object Tracking
- Threat Assessment
- Conjunction Awareness

### 📈 Mission Forecasting
- Health Trend Forecasts
- Risk Predictions
- Mission Outlook

### 🧠 AI Mission Copilot
- Mission Insights
- Operational Recommendations
- Decision Support

---

# 🏗️ Architecture

```text
Datasets
    │
    ▼
Data Processing
    │
    ▼
ML Prediction Engine
    │
    ▼
FastAPI Backend
    │
    ▼
Next.js Dashboard
```

---

# 💻 Tech Stack

## Frontend
- Next.js 14
- React
- TypeScript
- Tailwind CSS
- Recharts

## Backend
- FastAPI
- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib

---

# 📊 Datasets

### health_features.csv
Satellite health and telemetry features used for risk prediction.

### space_weather.csv
Space weather indicators including geomagnetic activity and solar conditions.

### satellite_catalog.csv
Satellite and orbital object information used for debris awareness.

---

# 🔌 API Endpoints

```http
GET /api/health
GET /api/weather
GET /api/debris
GET /api/failure-risk
GET /api/forecast
GET /api/copilot
```

---

# 📂 Project Structure

```text
hack-orbit/
│
├── frontend/
├── backend/
├── datasets/
│   ├── health_features.csv
│   ├── space_weather.csv
│   └── satellite_catalog.csv
│
└── README.md
```
