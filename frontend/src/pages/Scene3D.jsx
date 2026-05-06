import React, { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Stars, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { usePrediction } from "../hooks/useApi";

// ── Severity → colour map ────────────────────────────────────────────────────
const LEVEL_COLORS = {
  LOW:      { sun: "#ffcc44", corona: "#ff8800", particles: "#ffdd88", intensity: 0.3 },
  MODERATE: { sun: "#ff9900", corona: "#ff5500", particles: "#ffaa44", intensity: 0.55 },
  HIGH:     { sun: "#ff5500", corona: "#ff2200", particles: "#ff8844", intensity: 0.75 },
  EXTREME:  { sun: "#ff2200", corona: "#cc0000", particles: "#ff4444", intensity: 1.0 },
};

// ── Sun ───────────────────────────────────────────────────────────────────────
function Sun({ level }) {
  const meshRef  = useRef();
  const glowRef  = useRef();
  const colors   = LEVEL_COLORS[level] || LEVEL_COLORS.LOW;

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.003;
      const s = 1 + Math.sin(t * 1.2) * 0.025 * (1 + colors.intensity);
      meshRef.current.scale.setScalar(s);
    }
    if (glowRef.current) {
      const gs = 1.4 + Math.sin(t * 0.8) * 0.1 * colors.intensity;
      glowRef.current.scale.setScalar(gs);
      glowRef.current.material.opacity = 0.12 + 0.1 * Math.sin(t * 1.5) * colors.intensity;
    }
  });

  return (
    <group position={[-6, 0, 0]}>
      {/* Core */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[1.8, 64, 64]} />
        <meshStandardMaterial
          color={colors.sun}
          emissive={colors.sun}
          emissiveIntensity={1.2 + colors.intensity}
          roughness={0.6}
        />
      </mesh>
      {/* Corona glow */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[2.4, 32, 32]} />
        <meshBasicMaterial color={colors.corona} transparent opacity={0.15} side={THREE.BackSide} />
      </mesh>
      {/* Point light */}
      <pointLight color={colors.sun} intensity={4 + colors.intensity * 4} distance={40} />
    </group>
  );
}

// ── Earth ─────────────────────────────────────────────────────────────────────
function Earth({ level }) {
  const meshRef  = useRef();
  const atmoRef  = useRef();
  const colors   = LEVEL_COLORS[level] || LEVEL_COLORS.LOW;

  useFrame(() => {
    if (meshRef.current) meshRef.current.rotation.y += 0.004;
  });

  return (
    <group position={[6, 0, 0]}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial color="#1a6faa" roughness={0.7} metalness={0.1} />
      </mesh>
      {/* Atmosphere — glows redder during storms */}
      <mesh ref={atmoRef}>
        <sphereGeometry args={[1.12, 32, 32]} />
        <meshBasicMaterial
          color={level === "EXTREME" ? "#ff4444" : level === "HIGH" ? "#ff8800" : "#3a9ad9"}
          transparent
          opacity={0.08 + colors.intensity * 0.12}
          side={THREE.BackSide}
        />
      </mesh>
    </group>
  );
}

// ── Solar Wind Particle Stream ────────────────────────────────────────────────
function SolarWindParticles({ level }) {
  const meshRef  = useRef();
  const colors   = LEVEL_COLORS[level] || LEVEL_COLORS.LOW;
  const N        = Math.round(300 + colors.intensity * 700);

  const { positions, velocities } = useMemo(() => {
    const positions  = new Float32Array(N * 3);
    const velocities = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      positions[i * 3]     = -6 + Math.random() * 14;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 5;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 5;
      velocities[i]        = 0.02 + Math.random() * 0.04 * (1 + colors.intensity);
    }
    return { positions, velocities };
  }, [N, colors.intensity]);

  useFrame(() => {
    if (!meshRef.current) return;
    const pos = meshRef.current.geometry.attributes.position.array;
    for (let i = 0; i < N; i++) {
      pos[i * 3] += velocities[i];
      // Reset particle to sun side when it passes earth
      if (pos[i * 3] > 8) {
        pos[i * 3]     = -6.5;
        pos[i * 3 + 1] = (Math.random() - 0.5) * 4;
        pos[i * 3 + 2] = (Math.random() - 0.5) * 4;
      }
    }
    meshRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        color={colors.particles}
        size={0.05 + colors.intensity * 0.04}
        transparent
        opacity={0.7}
        sizeAttenuation
      />
    </points>
  );
}

// ── Magnetic field lines (simplified arc) ────────────────────────────────────
function MagneticLines({ level }) {
  const colors = LEVEL_COLORS[level] || LEVEL_COLORS.LOW;

  // useMemo must be called unconditionally — before any early return
  const lines = useMemo(() => {
    return Array.from({ length: 5 }, (_, i) => {
      const points = [];
      const offset = (i - 2) * 0.8;
      for (let t = 0; t <= 1; t += 0.02) {
        points.push(
          new THREE.Vector3(
            -6 + t * 14,
            offset + Math.sin(t * Math.PI) * (1.5 + colors.intensity),
            (i - 2) * 0.3
          )
        );
      }
      return new THREE.CatmullRomCurve3(points).getPoints(50);
    });
  }, [colors.intensity]);

  if (level === "LOW") return null;

  return (
    <>
      {lines.map((pts, i) => {
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        return (
          <line key={i} geometry={geo}>
            <lineBasicMaterial color={colors.corona} transparent opacity={0.25} />
          </line>
        );
      })}
    </>
  );
}

// ── HUD overlay ───────────────────────────────────────────────────────────────
function HUD({ pred, level }) {
  
  const levelColors = { LOW: "#22d3a0", MODERATE: "#f59e0b", HIGH: "#f97316", EXTREME: "#ef4444" };

  return (
    <div style={{
      position: "absolute", top: 16, left: 16, display: "flex", flexDirection: "column", gap: 8,
    }}>
      <div style={{ background: "rgba(8,12,20,0.85)", border: "1px solid #1e2d42", borderRadius: 10, padding: "10px 14px", minWidth: 180 }}>
        <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>STORM LEVEL</div>
        <div style={{ fontSize: 18, fontWeight: 700, color: levelColors[level] || "#22d3a0" }}>{level}</div>
      </div>
      {pred && (
        <div style={{ background: "rgba(8,12,20,0.85)", border: "1px solid #1e2d42", borderRadius: 10, padding: "10px 14px" }}>
          <div style={{ fontSize: 10, color: "#64748b", marginBottom: 6 }}>LIVE TELEMETRY</div>
          <div style={{ fontSize: 12, color: "#e2e8f0", lineHeight: 1.8 }}>
            <div>Speed: <b style={{ color: "#38bdf8" }}>{pred.current_speed} km/s</b></div>
            <div>Ensemble: <b style={{ color: "#38bdf8" }}>{pred.ensemble_prediction} km/s</b></div>
            <div>Confidence: <b style={{ color: "#22d3a0" }}>{Math.round((pred.confidence || 0) * 100)}%</b></div>
          </div>
        </div>
      )}
      <div style={{ background: "rgba(8,12,20,0.85)", border: "1px solid #1e2d42", borderRadius: 10, padding: "8px 14px", fontSize: 11, color: "#64748b" }}>
        Drag to rotate · Scroll to zoom
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Scene3D() {
  const { pred } = usePrediction(6000);
  const level    = pred?.storm_level || "LOW";

  return (
    <div style={{ position: "relative", height: "calc(100vh - 3rem)", borderRadius: 12, overflow: "hidden", background: "#000" }}>
      <div className="page-title" style={{ position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)", zIndex: 10, background: "rgba(8,12,20,0.7)", padding: "4px 16px", borderRadius: 8 }}>
        🌌 Heliospheric View
      </div>

      <Canvas
        camera={{ position: [0, 4, 18], fov: 55 }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.15} />
        <Stars radius={100} depth={50} count={5000} factor={4} fade speed={1} />
        <Sun level={level} />
        <Earth level={level} />
        <SolarWindParticles level={level} />
        <MagneticLines level={level} />
        <OrbitControls
          enablePan={false}
          minDistance={8}
          maxDistance={35}
          autoRotate
          autoRotateSpeed={0.4}
        />
      </Canvas>

      <HUD pred={pred} level={level} />
    </div>
  );
}