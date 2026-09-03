import { useEffect, useRef, useState } from "react";

const WS_URL = "ws://127.0.0.1:8000/ws/trust";
const MAX_HISTORY = 40;

interface TrustMessage {
  trust_score: number;
  timestamp: number;
  model_ready: boolean;
}

function statusFor(score: number): { label: string; color: string } {
  if (score >= 90) return { label: "Trusted", color: "#22c55e" };
  if (score >= 70) return { label: "Normal", color: "#84cc16" };
  if (score >= 40) return { label: "Deviation Detected", color: "#f59e0b" };
  return { label: "Anomaly — Possible Intrusion", color: "#ef4444" };
}

export default function App() {
  const [connected, setConnected] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [current, setCurrent] = useState<number>(50);
  const [history, setHistory] = useState<number[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 2000); // auto-reconnect
      };
      ws.onerror = () => ws.close();

      ws.onmessage = (event) => {
        const data: TrustMessage = JSON.parse(event.data);
        setCurrent(data.trust_score);
        setModelReady(data.model_ready);
        setHistory((prev) => [...prev.slice(-MAX_HISTORY + 1), data.trust_score]);
      };
    }
    connect();
    return () => wsRef.current?.close();
  }, []);

  const status = statusFor(current);
  const points = history
    .map((v, i) => `${(i / Math.max(history.length - 1, 1)) * 300},${100 - v}`)
    .join(" ");

  return (
    <div className="page">
      <header>
        <h1>🛡️ SentinelKey AI</h1>
        <p className="subtitle">Continuous Behavioral Authentication — Live Trust Monitor</p>
      </header>

      <div className={`conn-badge ${connected ? "on" : "off"}`}>
        {connected ? "● Connected to agent" : "○ Disconnected — retrying..."}
      </div>

      {!modelReady && (
        <div className="warning">
          ⚠️ No trained model detected yet. Showing neutral score (50). Run
          <code> train_model.py</code> after collecting baseline data.
        </div>
      )}

      <div className="score-card" style={{ borderColor: status.color }}>
        <div className="score" style={{ color: status.color }}>
          {current.toFixed(1)}
        </div>
        <div className="score-label" style={{ color: status.color }}>
          {status.label}
        </div>
      </div>

      <div className="chart-card">
        <h3>Trust Score — recent history</h3>
        <svg viewBox="0 0 300 100" preserveAspectRatio="none" className="chart">
          <polyline points={points} fill="none" stroke={status.color} strokeWidth="2" />
        </svg>
      </div>

      <footer>
        <p>SentinelKey AI · Software-only MVP · ESP32 sensor fusion coming next</p>
      </footer>
    </div>
  );
}
