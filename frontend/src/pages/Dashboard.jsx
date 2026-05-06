import React from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
} from "recharts";
import { useSolarData, usePrediction, useLogs, useBackendStatus } from "../hooks/useApi";

// ── Value formatting ──────────────────────────────────────────────────────────
const fmtVal = (val, unit = "", decimals = 1) => {
  if (val === null || val === undefined) return null;
  const n = parseFloat(val);
  if (isNaN(n)) return null;
  return `${n.toFixed(decimals)}${unit}`;
};

const stormColors = {
  LOW: "#22d3a0", MODERATE: "#f59e0b", HIGH: "#f97316", EXTREME: "#ef4444",
};

// ── Backend status banner ─────────────────────────────────────────────────────
function BackendBanner({ status }) {
  if (status === true) return null;
  const isChecking = status === null;
  return (
    <div style={{
      padding: "0.6rem 1rem", borderRadius: 8, marginBottom: "1rem", fontSize: "0.82rem",
      fontWeight: 500, display: "flex", alignItems: "center", gap: "0.5rem",
      background: isChecking ? "rgba(56,189,248,0.08)" : "rgba(239,68,68,0.12)",
      border: `1px solid ${isChecking ? "rgba(56,189,248,0.2)" : "rgba(239,68,68,0.3)"}`,
      color: isChecking ? "var(--accent)" : "var(--red)",
    }}>
      {isChecking ? "⏳ Connecting to backend…" : "❌ Backend offline — run: uvicorn main:app --port 8000"}
    </div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, unit, decimals = 1, color, loading, notReady }) {
  const formatted = fmtVal(value, unit ? ` ${unit}` : "", decimals);

  let display;
  if (loading)         display = <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>Connecting…</span>;
  else if (notReady)   display = <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>Awaiting models…</span>;
  else if (!formatted) display = <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>No data</span>;
  else                 display = formatted;

  return (
    <div className="card">
      <div className="card-title">{label}</div>
      <div className="card-value" style={{ color: color || "var(--text)", fontSize: "1.45rem", marginTop: "0.25rem" }}>
        {display}
      </div>
    </div>
  );
}

// ── Confidence card ───────────────────────────────────────────────────────────
function ConfidenceCard({ confidence, loading, notReady }) {
  const pct   = (!loading && !notReady && confidence != null) ? Math.round(confidence * 100) : 0;
  const color = pct > 70 ? "#22d3a0" : pct > 40 ? "#f59e0b" : "#ef4444";
  const label = pct > 70 ? "High confidence" : pct > 40 ? "Moderate" : "Low confidence";

  return (
    <div className="card">
      <div className="card-title">Model Confidence</div>
      <div className="card-value" style={{ color, fontSize: "1.45rem", marginTop: "0.25rem" }}>
        {loading ? <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>Connecting…</span>
         : notReady ? <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>Awaiting models…</span>
         : `${pct}%`}
      </div>
      <div className="conf-bar-wrap" style={{ marginTop: "0.5rem" }}>
        <div className="conf-bar" style={{ width: `${pct}%`, background: color, transition: "width 0.8s ease" }} />
      </div>
      <div className="card-sub" style={{ marginTop: "0.35rem" }}>{!loading && !notReady ? label : ""}</div>
    </div>
  );
}

// ── Storm card ────────────────────────────────────────────────────────────────
function StormCard({ level, probability, loading, notReady }) {
  const safeLevel = level || "LOW";
  const pct = Math.round((probability ?? 0) * 100);
  const color = stormColors[safeLevel] || "#22d3a0";

  return (
    <div className="card">
      <div className="card-title">Storm Level</div>
      {loading
        ? <div style={{ color: "var(--muted)", fontSize: "0.9rem", marginTop: "0.5rem" }}>Connecting…</div>
        : notReady
        ? <div style={{ color: "var(--muted)", fontSize: "0.9rem", marginTop: "0.5rem" }}>Awaiting models…</div>
        : <>
            <div style={{ marginTop: "0.5rem" }}>
              <span className={`storm-badge storm-${safeLevel}`}>{safeLevel}</span>
            </div>
            <div className="card-sub" style={{ marginTop: "0.5rem", color }}>
              Probability: {pct}%
            </div>
          </>
      }
    </div>
  );
}

// ── Weights card ──────────────────────────────────────────────────────────────
function WeightsCard({ weights, loading }) {
  const w = weights || { rf: 0.3, xgb: 0.35, lstm: 0.35 };
  const wColors = { rf: "#f59e0b", xgb: "#f97316", lstm: "#a78bfa" };

  return (
    <div className="card">
      <div className="card-title">Ensemble Weights</div>
      {["rf", "xgb", "lstm"].map((k) => {
        const pct = Math.round((w[k] ?? 0) * 100);
        return (
          <div key={k} style={{ marginTop: "0.55rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginBottom: "3px" }}>
              <span style={{ color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>{k}</span>
              <span style={{ color: wColors[k], fontWeight: 600 }}>{loading ? "…" : `${pct}%`}</span>
            </div>
            <div className="conf-bar-wrap">
              <div className="conf-bar" style={{ width: loading ? "0%" : `${pct}%`, background: wColors[k], transition: "width 0.8s ease" }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Log panel ─────────────────────────────────────────────────────────────────
function LogPanel({ logs }) {
  return (
    <div className="card">
      <div className="card-title" style={{ marginBottom: "0.75rem" }}>System Logs</div>
      <div className="log-panel">
        {logs.length === 0
          ? <div style={{ color: "var(--muted)" }}>Waiting for backend logs…</div>
          : [...logs].reverse().map((l, i) => (
              <div className="log-entry" key={i}>
                <span className="log-time">{String(l.time || "").slice(11, 19)}</span>
                {l.message}
              </div>
            ))
        }
      </div>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function Dashboard() {
  const backendUp                              = useBackendStatus();
  const { data: solarRows, loading: solLoad }  = useSolarData(5000);
  const { pred, predLoading, predError }       = usePrediction(8000);
  const logs                                   = useLogs(4000);

  const latest   = solarRows.length > 0 ? solarRows[solarRows.length - 1] : null;

  // Models not yet trained = backend responds but predictions are 0/null
  const modelsNotReady = !predLoading && (predError != null ||
    (pred && pred.rf_prediction === null && pred.xgb_prediction === null));

  // Chart data
  const chartData = solarRows.slice(-40).map((row, i) => ({
    t:       String(row.timestamp || "").slice(11, 19) || `${i}`,
    actual:  typeof row.speed === "number" ? Math.round(row.speed) : null,
    bz:      typeof row.bz    === "number" ? Math.round(row.bz * 10) / 10 : null,
    density: typeof row.density === "number" ? Math.round(row.density * 100) / 100 : null,
  }));

  if (chartData.length > 0 && pred && !modelsNotReady) {
    const last = chartData[chartData.length - 1];
    if (pred.rf_prediction       != null) last.rf       = pred.rf_prediction;
    if (pred.xgb_prediction      != null) last.xgb      = pred.xgb_prediction;
    if (pred.lstm_prediction     != null) last.lstm     = pred.lstm_prediction;
    if (pred.ensemble_prediction != null) last.ensemble = pred.ensemble_prediction;
  }

  return (
    <div>
      <div className="page-title">🛰️ Solar Wind Monitor</div>

      <BackendBanner status={backendUp} />

      {modelsNotReady && (
        <div style={{
          padding: "0.7rem 1rem", borderRadius: 8, marginBottom: "1rem", fontSize: "0.82rem",
          background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.3)",
          color: "#f59e0b",
        }}>
          ⚠️ Models not trained yet. Run <code style={{ background: "rgba(0,0,0,0.3)", padding: "1px 6px", borderRadius: 4 }}>python models_ml.py</code> and <code style={{ background: "rgba(0,0,0,0.3)", padding: "1px 6px", borderRadius: 4 }}>python models_dl.py</code> in your backend folder, then restart the server.
        </div>
      )}

      {pred?.alert && (
        <div className="alert-banner">{pred.alert}</div>
      )}

      {/* Row 1 — raw telemetry */}
      <div className="grid-4 section">
        <StatCard label="Solar Wind Speed" value={latest?.speed}       unit="km/s"  decimals={0} color="var(--accent)"  loading={solLoad} />
        <StatCard label="Bz Component"     value={latest?.bz}          unit="nT"    decimals={1} color={latest?.bz < -5 ? "#ef4444" : "#22d3a0"} loading={solLoad} />
        <StatCard label="Proton Density"   value={latest?.density}     unit="/cm³"  decimals={2} loading={solLoad} />
        <StatCard label="Plasma Temp"      value={latest?.temperature ? latest.temperature / 1000 : null} unit="k K" decimals={0} loading={solLoad} />
      </div>

      {/* Row 2 — model predictions */}
      <div className="grid-4 section">
        <StatCard label="RF Prediction"       value={pred?.rf_prediction}       unit="km/s" decimals={1} color="#f59e0b" loading={predLoading} notReady={modelsNotReady} />
        <StatCard label="XGBoost Prediction"  value={pred?.xgb_prediction}      unit="km/s" decimals={1} color="#f97316" loading={predLoading} notReady={modelsNotReady} />
        <StatCard label="LSTM Prediction"     value={pred?.lstm_prediction}     unit="km/s" decimals={1} color="#a78bfa" loading={predLoading} notReady={modelsNotReady} />
        <StatCard label="Ensemble Prediction" value={pred?.ensemble_prediction} unit="km/s" decimals={1} color="var(--accent)" loading={predLoading} notReady={modelsNotReady} />
      </div>

      {/* Row 3 — confidence + storm + weights */}
      <div className="grid-3 section">
        <ConfidenceCard confidence={pred?.confidence}    loading={predLoading} notReady={modelsNotReady} />
        <StormCard level={pred?.storm_level} probability={pred?.storm_probability} loading={predLoading} notReady={modelsNotReady} />
        <WeightsCard weights={pred?.ensemble_weights}   loading={predLoading} />
      </div>

      {/* Live speed chart */}
      <div className="card section">
        <div className="card-title" style={{ marginBottom: "1rem" }}>Live Solar Wind Speed + Model Predictions</div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="t" tick={{ fontSize: 10, fill: "var(--muted)" }} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10, fill: "var(--muted)" }} domain={["auto", "auto"]} />
            <Tooltip
              contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }}
              formatter={(val, name) => val != null ? [`${val} km/s`, name] : [null, name]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line dataKey="actual"   stroke="var(--text)"   strokeWidth={2}   dot={false} name="Actual"   connectNulls />
            <Line dataKey="rf"       stroke="#f59e0b"       strokeWidth={1.5} dot={{ r: 5 }} name="RF"       connectNulls />
            <Line dataKey="xgb"     stroke="#f97316"       strokeWidth={1.5} dot={{ r: 5 }} name="XGBoost" connectNulls />
            <Line dataKey="lstm"    stroke="#a78bfa"        strokeWidth={1.5} dot={{ r: 5 }} name="LSTM"    connectNulls />
            <Line dataKey="ensemble" stroke="var(--accent)" strokeWidth={2.5} dot={{ r: 6 }} name="Ensemble" connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Bz chart */}
      <div className="card section">
        <div className="card-title" style={{ marginBottom: "1rem" }}>Bz Component (nT) — Storm Driver</div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="t" tick={{ fontSize: 10, fill: "var(--muted)" }} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10, fill: "var(--muted)" }} />
            <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }} />
            <ReferenceLine y={-5}  stroke="#f59e0b" strokeDasharray="4 2" label={{ value: "Moderate", fill: "#f59e0b", fontSize: 9, position: "insideTopLeft" }} />
            <ReferenceLine y={-12} stroke="#ef4444" strokeDasharray="4 2" label={{ value: "High",     fill: "#ef4444", fontSize: 9, position: "insideTopLeft" }} />
            <Line dataKey="bz" stroke="var(--accent)" strokeWidth={1.5} dot={false} name="Bz" connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <LogPanel logs={logs} />
    </div>
  );
}