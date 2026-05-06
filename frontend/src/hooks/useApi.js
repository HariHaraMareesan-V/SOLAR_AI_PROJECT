/**
 * useApi.js — Fixed version
 * Fixes: backend detection, timeout handling, model-not-trained state
 */

import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";

// ── Axios instance pointing directly to backend ───────────────────────────────
const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL || "http://localhost:8000",
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// ── Safe number parser ────────────────────────────────────────────────────────
export const safeNum = (val, decimals = 1) => {
  const n = parseFloat(val);
  return isNaN(n) ? null : parseFloat(n.toFixed(decimals));
};

// ── Core polling hook ─────────────────────────────────────────────────────────
function usePoll(fetchFn, interval = 5000) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const mountedRef            = useRef(true);
  const timerRef              = useRef(null);
  const fnRef                 = useRef(fetchFn);

  // Always keep ref current so interval uses latest fn without restarting
  useEffect(() => { fnRef.current = fetchFn; }, [fetchFn]);

  const execute = useCallback(async () => {
    try {
      const result = await fnRef.current();
      if (mountedRef.current) {
        setData(result);
        setError(null);
        setLoading(false);
      }
    } catch (err) {
      if (!mountedRef.current) return;
      const isTimeout  = err.code === "ECONNABORTED" || err.message?.includes("timeout");
      const isOffline  = err.code === "ERR_NETWORK"  || err.message?.includes("Network");
      const serverMsg  = err?.response?.data?.detail;

      setError(
        isTimeout  ? "timeout"  :
        isOffline  ? "offline"  :
        serverMsg  ? serverMsg  :
        err.message || "error"
      );
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    execute();
    timerRef.current = setInterval(execute, interval);
    return () => {
      mountedRef.current = false;
      clearInterval(timerRef.current);
    };
  }, [execute, interval]);

  return { data, loading, error, refetch: execute };
}

// ── Retry on HTTP 202 (metrics still computing) ───────────────────────────────
function useRetryOn202(url, retryDelay = 3000, maxRetries = 40) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const mountedRef            = useRef(true);
  const attemptsRef           = useRef(0);
  const timerRef              = useRef(null);

  const attempt = useCallback(async () => {
    if (!mountedRef.current) return;
    if (attemptsRef.current >= maxRetries) {
      if (mountedRef.current) {
        setError("Metrics not ready. Train models first then restart backend.");
        setLoading(false);
      }
      return;
    }
    attemptsRef.current += 1;
    try {
      const res = await API.get(url);
      if (mountedRef.current) { setData(res.data); setLoading(false); setError(null); }
    } catch (err) {
      if (err?.response?.status === 202) {
        timerRef.current = setTimeout(attempt, retryDelay);
      } else if (mountedRef.current) {
        setError(err?.response?.data?.detail || err.message || "Failed");
        setLoading(false);
      }
    }
  }, [url, retryDelay, maxRetries]);

  useEffect(() => {
    mountedRef.current = true;
    attemptsRef.current = 0;
    attempt();
    return () => {
      mountedRef.current = false;
      clearTimeout(timerRef.current);
    };
  }, [attempt]);

  return { data, loading, error };
}

// ── useBackendStatus — pings /health every 4s ─────────────────────────────────
export function useBackendStatus() {
  const [status, setStatus] = useState(null); // null=checking, true=up, false=down
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    const check = async () => {
      try {
        await API.get("/health", { timeout: 4000 });
        if (mountedRef.current) setStatus(true);
      } catch {
        if (mountedRef.current) setStatus(false);
      }
    };

    check();
    const id = setInterval(check, 4000);
    return () => { mountedRef.current = false; clearInterval(id); };
  }, []);

  return status;
}

// ── useSolarData ──────────────────────────────────────────────────────────────
export function useSolarData(interval = 5000) {
  const fetchFn = useCallback(async () => {
    const res = await API.get("/solar-data?n=60");
    return Array.isArray(res.data) ? res.data : [];
  }, []);

  const { data, loading, error } = usePoll(fetchFn, interval);
  return { data: data || [], loading, error };
}

// ── usePrediction ─────────────────────────────────────────────────────────────
export function usePrediction(interval = 8000) {
  const fetchFn = useCallback(async () => {
    const res = await API.get("/prediction");
    const d   = res.data || {};

    return {
      timestamp:            d.timestamp           || new Date().toISOString(),
      current_speed:        safeNum(d.current_speed,       1) ?? 0,
      rf_prediction:        safeNum(d.rf_prediction,       1),   // null = not trained
      xgb_prediction:       safeNum(d.xgb_prediction,      1),
      lstm_prediction:      safeNum(d.lstm_prediction,     1),
      ensemble_prediction:  safeNum(d.ensemble_prediction, 1),
      confidence:           safeNum(d.confidence,          3) ?? 0,
      storm_level:          d.storm_level         || "LOW",
      storm_probability:    safeNum(d.storm_probability,   3) ?? 0,
      ensemble_weights:     d.ensemble_weights    || { rf: 0.3, xgb: 0.35, lstm: 0.35 },
      alert:                d.alert               || null,
    };
  }, []);

  const { data, loading, error } = usePoll(fetchFn, interval);
  return { pred: data, predLoading: loading, predError: error };
}

// ── useMultistep ──────────────────────────────────────────────────────────────
export function useMultistep(horizon = 12) {
  const fetchFn = useCallback(async () => {
    const res = await API.get(`/multistep?horizon=${horizon}`);
    const d   = res.data || {};
    const len = d.horizon || horizon;

    const ensureArr = (v) =>
      Array.isArray(v) && v.length > 0
        ? v.map((x) => safeNum(x, 1) ?? 0)
        : Array(len).fill(0);

    return {
      horizon:           len,
      rf_forecast:       ensureArr(d.rf_forecast),
      xgb_forecast:      ensureArr(d.xgb_forecast),
      lstm_forecast:     ensureArr(d.lstm_forecast),
      ensemble_forecast: ensureArr(d.ensemble_forecast),
    };
  }, [horizon]);

  const { data, loading, error, refetch } = usePoll(fetchFn, 20000);
  return { forecast: data, forecastLoading: loading, forecastError: error, refetch };
}

// ── useMetrics — retries on 202 ───────────────────────────────────────────────
export function useMetrics() {
  const { data, loading, error } = useRetryOn202("/metrics", 3000, 40);
  return { metrics: data?.models ?? null, metricsLoading: loading, metricsError: error };
}

// ── useLogs ───────────────────────────────────────────────────────────────────
export function useLogs(interval = 4000) {
  const fetchFn = useCallback(async () => {
    const res = await API.get("/logs?n=40");
    return Array.isArray(res.data?.logs) ? res.data.logs : [];
  }, []);

  const { data } = usePoll(fetchFn, interval);
  return data || [];
}

// ── useFeatureImportance ──────────────────────────────────────────────────────
export function useFeatureImportance() {
  const [fi, setFi]           = useState(null);
  const [loading, setLoading] = useState(true);
  const mountedRef            = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    API.get("/feature-importance")
      .then((r) => { if (mountedRef.current) { setFi(r.data); setLoading(false); } })
      .catch(() => { if (mountedRef.current) setLoading(false); });
    return () => { mountedRef.current = false; };
  }, []);

  return { fi, loading };
}