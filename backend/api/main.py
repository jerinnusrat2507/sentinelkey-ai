"""
main.py
-------
FastAPI backend for SentinelKey AI.

Endpoints:
  POST /ingest        -> agent posts a feature vector, backend scores it
                          and broadcasts the result to all connected
                          dashboard clients over WebSocket.
  WS   /ws/trust       -> dashboard connects here to receive live trust
                          score updates as JSON: {"trust_score": .., "timestamp": ..}
  GET  /health         -> simple liveness check
  GET  /latest         -> last known trust score (useful for page refresh)

Run with:
    uvicorn main:app --reload --port 8000
"""

import sys
import time
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append("../ml")
sys.path.append("../agent")
from model import TrustModel  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402

app = FastAPI(title="SentinelKey AI API")

# Dashboard runs on a different port during development (Vite default: 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

trust_model = TrustModel()

_latest_state = {"trust_score": 50.0, "timestamp": time.time(), "model_ready": trust_model.is_ready()}


class FeatureVector(BaseModel):
    typing_speed_kps: float
    avg_hold_time: float
    avg_flight_time: float
    hold_time_std: float
    flight_time_std: float
    avg_mouse_speed: float
    mouse_speed_std: float
    click_rate_cps: float
    avg_click_interval: float
    idle_ratio: float


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        stale = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


manager = ConnectionManager()


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": trust_model.is_ready()}


@app.get("/latest")
def latest():
    return _latest_state


@app.post("/ingest")
async def ingest(features: FeatureVector):
    feature_dict = features.dict()
    trust_score = trust_model.score(feature_dict)

    _latest_state["trust_score"] = trust_score
    _latest_state["timestamp"] = time.time()
    _latest_state["model_ready"] = trust_model.is_ready()

    await manager.broadcast(
        {
            "trust_score": round(trust_score, 2),
            "timestamp": _latest_state["timestamp"],
            "model_ready": trust_model.is_ready(),
        }
    )
    return {"trust_score": round(trust_score, 2)}


@app.websocket("/ws/trust")
async def ws_trust(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # send current state immediately on connect
        await websocket.send_json(
            {
                "trust_score": round(_latest_state["trust_score"], 2),
                "timestamp": _latest_state["timestamp"],
                "model_ready": _latest_state["model_ready"],
            }
        )
        while True:
            # keep the connection alive; dashboard doesn't need to send anything
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
