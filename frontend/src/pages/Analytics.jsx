import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { useMetrics, useFeatureImportance } from "../hooks/useApi";

function Spinner({ message }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "2.5rem", gap: 12 }}>
      <div style={{
        width: 32, height: 32, border: "3px solid var(--border)",
        borderTop: "3px solid var(--accent)", borderRadius: "50%",
        animation: "spin 1s linear infinite",
      }} />
      <div style={{ color: "var(--muted)", fontSize: "0.82rem" }}>{message}</div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ── Metrics Table ─────────────────────────────────────────────────────────────
function MetricsTable({ metrics }) {
  const rows = Object.entries(metrics).map(([model, m]) => ({
    model,
    MAE:  parseFloat(m.MAE  ?? m.mae  ?? 0),
    RMSE: parseFloat(m.RMSE ?? m.rmse ?? 0),
    R2:   parseFloat(m.R2   ?? m.r2   ?? 0),
  }));

  const best = {
    MAE:  Math.min(...rows.map((r) => r.MAE)),
    RMSE: Math.min(...rows.map((r) => r.RMSE)),
    R2:   Math.max(...rows.map((r) => r.R2)),
  };

  return (
    <div style={{ overflowX: "auto" }}>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>MAE ↓ <span style={{ color: "var(--muted)", fontWeight: 400 }}>(lower=better)</span></th>
            <th>RMSE ↓</th>
            <th>R² ↑ <span style={{ color: "var(--muted)", fontWeight: 400 }}>(higher=better)</span></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.model}>
              <td style={{ fontWeight: 600 }}>{r.model}</td>
              <td style={{ color: r.MAE  === best.MAE  ? "var(--green)" : "var(--text)" }}>
                {r.MAE.toFixed(4)} {r.MAE === best.MAE && "🏆"}
              </td>
              <td style={{ color: r.RMSE === best.RMSE ? "var(--green)" : "var(--text)" }}>
                {r.RMSE.toFixed(4)} {r.RMSE === best.RMSE && "🏆"}
              </td>
              <td style={{ color: r.R2   === best.R2   ? "var(--green)" : "var(--text)" }}>
                {r.R2.toFixed(4)} {r.R2 === best.R2 && "🏆"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Metrics Bar Charts ────────────────────────────────────────────────────────
function MetricsBars({ metrics }) {
  const data = Object.entries(metrics).map(([model, m]) => ({
    model: model.split(" ")[0],
    MAE:   parseFloat(m.MAE  ?? m.mae  ?? 0),
    RMSE:  parseFloat(m.RMSE ?? m.rmse ?? 0),
    R2:    parseFloat(m.R2   ?? m.r2   ?? 0),
  }));

  const metricDefs = [
    { key: "MAE",  color: "var(--yellow)", label: "MAE" },
    { key: "RMSE", color: "var(--orange)", label: "RMSE" },
    { key: "R2",   color: "var(--green)",  label: "R²" },
  ];

  return (
    <div className="grid-3" style={{ marginTop: "1rem" }}>
      {metricDefs.map(({ key, color, label }) => (
        <div className="card" key={key}>
          <div className="card-title">{label}</div>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="model" tick={{ fontSize: 10, fill: "var(--muted)" }} />
              <YAxis tick={{ fontSize: 10, fill: "var(--muted)" }} />
              <Tooltip
                contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 11 }}
                formatter={(v) => [v.toFixed(4), label]}
              />
              <Bar dataKey={key} fill={color} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}

// ── Feature Importance ────────────────────────────────────────────────────────
function FeatureImportance({ fi }) {
  if (!fi?.random_forest) return (
    <div style={{ color: "var(--muted)", fontSize: "0.85rem", padding: "1rem" }}>
      Feature importance not available. Ensure RF model is trained.
    </div>
  );

  const data = Object.entries(fi.random_forest)
    .slice(0, 12)
    .map(([name, val]) => ({
      name: name.length > 18 ? name.slice(0, 18) + "…" : name,
      importance: parseFloat(val.toFixed(4)),
    }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} layout="vertical" margin={{ left: 10, right: 24, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" tick={{ fontSize: 10, fill: "var(--muted)" }} />
        <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 10, fill: "var(--muted)" }} />
        <Tooltip
          contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 11 }}
          formatter={(v) => [v, "Importance"]}
        />
        <Bar dataKey="importance" fill="var(--yellow)" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Architecture Cards ────────────────────────────────────────────────────────
const ARCH = [
  {
    title: "Random Forest", color: "var(--yellow)",
    rows: [["Estimators","200"],["Max Depth","12"],["Features","sqrt"],["OOB","✓"],["Type","Ensemble/Bag"]],
  },
  {
    title: "XGBoost", color: "var(--orange)",
    rows: [["Estimators","300"],["Max Depth","6"],["LR","0.05"],["Subsample","0.8"],["Type","Grad. Boost"]],
  },
  {
    title: "LSTM", color: "#a78bfa",
    rows: [["Layers","2×LSTM"],["Units","128→64"],["Seq Len","20"],["Uncertainty","MC Dropout"],["Type","Deep RNN"]],
  },
  {
    title: "Ensemble", color: "var(--accent)",
    rows: [["RF","30%"],["XGBoost","35%"],["LSTM","35%"],["Weighting","Dynamic"],["Type","Fusion"]],
  },
];

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function Analytics() {
  const { metrics, loading: metricsLoading, error: metricsError } = useMetrics();
  const { fi, loading: fiLoading }                                  = useFeatureImportance();

  return (
    <div>
      <div className="page-title">🧠 AI Analytics</div>

      {/* Architecture */}
      <div className="grid-4 section">
        {ARCH.map((c) => (
          <div className="card" key={c.title}>
            <div className="card-title" style={{ color: c.color, marginBottom: "0.75rem" }}>{c.title}</div>
            {c.rows.map(([label, val]) => (
              <div key={label} style={{
                display: "flex", justifyContent: "space-between",
                fontSize: "0.78rem", padding: "0.22rem 0",
                borderBottom: "1px solid var(--border)",
              }}>
                <span style={{ color: "var(--muted)" }}>{label}</span>
                <span style={{ fontWeight: 500 }}>{val}</span>
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Metrics */}
      <div className="card section">
        <div className="card-title" style={{ marginBottom: "0.75rem" }}>
          Model Performance Comparison
          {metricsLoading && (
            <span style={{ color: "var(--accent)", fontSize: "0.75rem", marginLeft: "0.75rem", fontWeight: 400 }}>
              ⏳ Computing metrics… retrying automatically
            </span>
          )}
        </div>

        {metricsLoading && <Spinner message="Backend is computing metrics. Retrying every 3s…" />}

        {metricsError && !metricsLoading && (
          <div style={{
            background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)",
            borderRadius: 8, padding: "0.75rem 1rem", color: "var(--red)", fontSize: "0.85rem",
          }}>
            ⚠️ {metricsError}
          </div>
        )}

        {metrics && !metricsLoading && (
          <>
            <MetricsTable metrics={metrics} />
            <MetricsBars  metrics={metrics} />
          </>
        )}
      </div>

      {/* Feature importance */}
      <div className="card section">
        <div className="card-title" style={{ marginBottom: "1rem" }}>
          Top Feature Importances (Random Forest)
        </div>
        {fiLoading
          ? <Spinner message="Loading feature importance…" />
          : <FeatureImportance fi={fi} />
        }
      </div>

      {/* Horizon table */}
      <div className="card section">
        <div className="card-title" style={{ marginBottom: "0.75rem" }}>Prediction Horizons</div>
        <table className="metrics-table">
          <thead>
            <tr><th>Horizon</th><th>Time</th><th>Use Case</th><th>Models</th></tr>
          </thead>
          <tbody>
            {[
              ["t+1",  "5 min",  "Immediate alert",      "RF, XGB, LSTM, Ensemble"],
              ["t+6",  "30 min", "Short-range watch",    "Ensemble (recursive)"],
              ["t+12", "1 hour", "Operational forecast", "Ensemble (recursive)"],
              ["t+24", "2 hours","Extended outlook",     "Ensemble (recursive)"],
            ].map(([h, t, u, m]) => (
              <tr key={h}>
                <td style={{ color: "var(--accent)", fontWeight: 600 }}>{h}</td>
                <td style={{ color: "var(--muted)" }}>{t}</td>
                <td>{u}</td>
                <td style={{ fontSize: "0.8rem", color: "var(--muted)" }}>{m}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}