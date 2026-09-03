# 🛡️ SentinelKey AI

**Continuous behavioral authentication** — instead of logging in once and
trusting a session forever, SentinelKey AI silently watches *how* you type
and move your mouse, and continuously computes a **Trust Score (0–100)**
that reflects how closely your current behavior matches your own baseline.
If someone else takes over your keyboard mid-session, the score drops and
the system can flag or lock the session — no extra login step required.

This repo contains the **software-only MVP**: keystroke + mouse capture,
an Isolation-Forest anomaly model, a FastAPI backend, and a live React
dashboard. Hardware sensor fusion (ESP32 + motion data) and an
explainability layer (SHAP-based per-signal attribution) are the next
planned milestones — see [Roadmap](#-roadmap).


---

## 🧩 How it works

```
 Keyboard/Mouse ──▶  Desktop Agent  ──▶  Feature Extraction  ──▶  FastAPI Backend
  (pynput)             (collector.py)      (typing rhythm,          │
                                             mouse dynamics)          ▼
                                                              Isolation Forest
                                                              (Trust Score 0-100)
                                                                       │
                                                                       ▼
                                                          WebSocket ──▶ React Dashboard
```

1. **Collector** listens to raw keyboard/mouse events in the background.
2. Every few seconds it extracts a **behavioral feature vector**
   (typing speed, key hold/flight time, mouse speed, click rate, etc.).
3. An **Isolation Forest** model — trained on *your own* baseline
   behavior — scores how normal or anomalous that vector is.
4. The backend broadcasts the score over **WebSocket** to a live
   dashboard that updates in real time.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Desktop Agent | Python, `pynput` |
| ML Model | scikit-learn (Isolation Forest) |
| Backend API | FastAPI, WebSocket, Uvicorn |
| Dashboard | React, TypeScript, Vite |
| Data | pandas / CSV (baseline data), joblib (model persistence) |

---

## ✨ Key Features

- Real-time keystroke dynamics capture (hold time, flight time, typing speed)
- Real-time mouse dynamics capture (speed, click rate, click interval)
- Personalized anomaly detection — trains on *your* baseline, not a generic model
- Live trust score streamed to a dashboard over WebSocket (auto-reconnects)
- Clear trust bands: Trusted / Normal / Deviation Detected / Anomaly
- Modular design — swap the model, add new features, or plug in new sensors independently

---

## 📦 Dependencies

**Backend (Python)** — see [`requirements.txt`](./requirements.txt):
`fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `pandas`, `numpy`,
`joblib`, `pynput`, `requests`

**Frontend (Node)** — see [`frontend/package.json`](./frontend/package.json):
`react`, `react-dom`, `vite`, `typescript`, `@vitejs/plugin-react`

---

## 🚀 Running it locally

### 1. Clone & install backend dependencies

```bash
git clone https://github.com/jerinnusrat2507/sentinelkey-ai.git
cd sentinelkey-ai
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Collect baseline behavioral data

Run the agent in **record mode** and just use your computer normally for
10–15 minutes (typing, browsing, etc.):

```bash
cd backend/agent
python collector.py --record ../../data/baseline.csv --window 5
```

### 3. Train the trust model

```bash
cd ../ml
python train_model.py --data ../../data/baseline.csv --out ../../models/trust_model.joblib
```

### 4. Start the backend API

```bash
cd ../api
uvicorn main:app --reload --port 8000
```

### 5. Start the dashboard

```bash
cd ../../frontend
npm install
npm run dev
```

Open **http://localhost:5173** — you should see the live Trust Score.

### 6. Start the agent in live mode (separate terminal)

```bash
cd backend/agent
python collector.py --window 5
```

Now use your keyboard/mouse normally and watch the trust score update in
real time. Try typing in an unusual way (much faster/slower, erratic
mouse movement) to see the score drop.

---

## 🗺️ Roadmap

- [x] Software-only MVP (keystroke + mouse → Isolation Forest → trust score)
- [ ] ESP32-S3 + MPU6050 hardware sensor fusion (device motion signal)
- [ ] Webcam-based head-pose / presence signal
- [ ] SHAP-based explainability — *"which signal caused the score to drop?"*
- [ ] PostgreSQL persistence for long-term behavioral history
- [ ] Research paper: *"Explainable Multi-Modal Continuous Authentication —
      Quantifying Per-Signal Contribution to Trust Score Degradation"*

---

## 🔗 Links

- Live demo: _add link once deployed_
- Author: [Jerin Nusrat](https://github.com/jerinnusrat2507)
