import React from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Analytics from "./pages/Analytics";
import Forecast from "./pages/Forecast";
import Scene3D from "./pages/Scene3D";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <nav className="sidebar">
          <div className="logo">
            <span className="logo-icon">☀️</span>
            <span className="logo-text">SolarAI</span>
          </div>
          <NavLink to="/"         className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>📊 Dashboard</NavLink>
          <NavLink to="/forecast" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>🔮 Forecast</NavLink>
          <NavLink to="/analytics"className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>🧠 Analytics</NavLink>
          <NavLink to="/3d"       className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>🌌 3D View</NavLink>
        </nav>
        <main className="content">
          <Routes>
            <Route path="/"          element={<Dashboard />} />
            <Route path="/forecast"  element={<Forecast />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/3d"        element={<Scene3D />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}