import React, { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid, AreaChart, Area,
} from "recharts";
import { useMultistep, useBackendStatus } from "../hooks/useApi";

export default function Forecast() {
  const [horizon, setHorizon]                          = useState(12);
  const { forecast, forecastLoading, forecastError, refetch } = useMultistep(horizon);
  const backendUp                                       = useBackendStatus();

  const chartData = forecast
    ? Array.from({ length: forecast.horizon }, (_, i) => ({
        step:     `t+${i + 1}`,
        RF:       forecast.rf_forecast[i]       ?? 0,
        XGBoost:  forecast.xgb_forecast[i]      ?? 0,
        LSTM:     forecast.lstm_forecast[i]      ?? 0,
        Ensemble: forecast.ensemble_forecast[i]  ?? 0,
      }))
    : [];

  const spreadData = chartData.map((row) => {
    const vals = [row.RF, row.XGBoost, row.LSTM].filter((v) => v > 0);
    return {
      step:     row.step,
      Ensemble: row.Ensemble,
      upper:    vals.length ? Math.max(...vals) : row.Ensemble,
      lower:    vals.length ? Math.min(...vals) : row.Ensemble,
    };
  });

  const allZero = forecast && chartData.every((r) => r.Ensemble === 0);

  return (
    <div>
      <div className="page-title">🔮 Multi-Step Forecast</div>

      {/* Horizon selector */}
      <div className="card section" style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
        <span style={{ fontSize: "0.82rem", color: "var(--muted)" }}>Horizon:</span>
        {[6, 12, 18, 24].map((h) => (
          <button key={h} onClick={() => setHorizon(h)} style={{
            padding: "0.3rem 0.9rem", borderRadius: 6,
            border: `1px solid ${horizon === h ? "var(--accent)" : "var(--border)"}`,
            background: horizon === h ? "rgba(56,189,248,0.12)" : "transparent",
            color: horizon === h ? "var(--accent)" : "var(--muted)",
            cursor: "pointer", fontSize: "0.82rem", fontWeight: horizon === h ? 600 : 400,
            transition: "all 0.15s",
          }}>
            t+{h}
          </button>
        ))}
        <button onClick={refetch} style={{
          marginLeft: "auto", padding: "0.3rem 0.9rem", borderRadius: 6,
          border: "1px solid var(--border)", background: "transparent",
          color: "var(--muted)", cursor: "pointer", fontSize: "0.82rem",
        }}>↺ Refresh</button>
      </div>

      {/* Status messages */}
      {backendUp === false && (
        <div style={{ padding: "0.75rem 1rem", borderRadius: 8, marginBottom: "1rem", fontSize: "0.82rem",
          background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#ef4444" }}>
          ❌ Backend is offline. Start it with: <code style={{ background: "rgba(0,0,0,0.3)", padding: "1px 6px", borderRadius: 4 }}>uvicorn main:app --port 8000</code>
        </div>
      )}

      {forecastError && backendUp !== false && (
        <div style={{ padding: "0.75rem 1rem", borderRadius: 8, marginBottom: "1rem", fontSize: "0.82rem",
          background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.3)", color: "#f59e0b" }}>
          ⚠️ {forecastError}
          {forecastError.includes("timeout") && (
            <div style={{ marginTop: "0.4rem", color: "var(--muted)" }}>
              The /multistep endpoint is slow because models aren't trained yet. Run <code style={{ background: "rgba(0,0,0,0.3)", padding: "1px 5px", borderRadius: 4 }}>python models_ml.py</code> first.
            </div>
          )}
        </div>
      )}

      {allZero && (
        <div style={{ padding: "0.75rem 1rem", borderRadius: 8, marginBottom: "1rem", fontSize: "0.82rem",
          background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.3)", color: "#f59e0b" }}>
          ⚠️ All forecasts are 0 — models are not trained. Run <code style={{ background: "rgba(0,0,0,0.3)", padding: "1px 6px", borderRadius: 4 }}>python models_ml.py && python models_dl.py</code> in your backend folder.
        </div>
      )}

      {forecastLoading && !forecast && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "3rem", gap: 12 }}>
          <div style={{ width: 36, height: 36, border: "3px solid var(--border)", borderTop: "3px solid var(--accent)",
            borderRadius: "50%", animation: "spin 1s linear infinite" }} />
          <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>Fetching forecast… (first call may take 30s)</div>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {forecast && !allZero && (
        <>
          <div className="card section">
            <div className="card-title" style={{ marginBottom: "1rem" }}>All Models — {horizon}-Step Forecast (km/s)</div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="step" tick={{ fontSize: 10, fill: "var(--muted)" }} />
                <YAxis tick={{ fontSize: 10, fill: "var(--muted)" }} domain={["auto", "auto"]} />
                <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }}
                  formatter={(val, name) => [`${val} km/s`, name]} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="RF"       stroke="#f59e0b" strokeWidth={1.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                <Line dataKey="XGBoost"  stroke="#f97316" strokeWidth={1.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                <Line dataKey="LSTM"     stroke="#a78bfa" strokeWidth={1.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                <Line dataKey="Ensemble" stroke="var(--accent)" strokeWidth={2.5} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card section">
            <div className="card-title" style={{ marginBottom: "1rem" }}>Ensemble + Uncertainty Band</div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={spreadData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="step" tick={{ fontSize: 10, fill: "var(--muted)" }} />
                <YAxis tick={{ fontSize: 10, fill: "var(--muted)" }} domain={["auto", "auto"]} />
                <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 12 }}
                  formatter={(val, name) => [`${val} km/s`, name]} />
                <Area dataKey="upper" stroke="none" fill="rgba(56,189,248,0.1)" name="Upper" />
                <Area dataKey="lower" stroke="none" fill="var(--bg)" name="Lower" />
                <Line dataKey="Ensemble" stroke="var(--accent)" strokeWidth={2.5} dot={{ r: 4 }} name="Ensemble" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="card section">
            <div className="card-title" style={{ marginBottom: "0.75rem" }}>Forecast Table</div>
            <div style={{ overflowX: "auto" }}>
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Step</th>
                    <th style={{ color: "#f59e0b" }}>RF</th>
                    <th style={{ color: "#f97316" }}>XGBoost</th>
                    <th style={{ color: "#a78bfa" }}>LSTM</th>
                    <th style={{ color: "var(--accent)" }}>Ensemble</th>
                    <th style={{ color: "var(--muted)" }}>±Spread</th>
                  </tr>
                </thead>
                <tbody>
                  {chartData.map((row) => {
                    const vals   = [row.RF, row.XGBoost, row.LSTM];
                    const spread = Math.round(Math.max(...vals) - Math.min(...vals));
                    return (
                      <tr key={row.step}>
                        <td style={{ color: "var(--muted)", fontWeight: 500 }}>{row.step}</td>
                        <td>{row.RF}</td>
                        <td>{row.XGBoost}</td>
                        <td>{row.LSTM}</td>
                        <td style={{ fontWeight: 700, color: "var(--accent)" }}>{row.Ensemble}</td>
                        <td style={{ color: spread > 50 ? "#ef4444" : "#22d3a0" }}>±{spread}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}