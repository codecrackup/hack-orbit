"""
Training data generator — real-world calibrated (spec §2.2-2.4).

Real datasets used for calibration:
  datasets/noaa/geomagnetic/kp_index.json   → Kp storm distribution
  datasets/noaa/solar/solar_wind.json       → solar wind speed distribution
  datasets/noaa/solar/magnetic_field.json   → magnetic field Bt distribution
  datasets/processed/collision_event.csv    → TLE orbital altitudes → radiation profile

Run from backend/: python ml/generate_data.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd

from app.schemas.telemetry import FEATURE_SCHEMA, FEATURE_NAMES
from app.services.health_score.scorer import compute_health_score

np.random.seed(42)

N_TOTAL       = 12000
HEALTHY_FRAC  = 0.70
DOUBLE_FAULT_PROB = 0.25

# ---------------------------------------------------------------------------
# Load and extract real-world distributions from pulled datasets
# ---------------------------------------------------------------------------

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")


def _load_real_distributions() -> dict:
    dist = {}

    # --- Kp index (NOAA, June 2026) ---
    try:
        kp_path = os.path.join(_ROOT, "noaa", "geomagnetic", "kp_index.json")
        with open(kp_path) as f:
            kp_data = json.load(f)
        kp_values = np.array([r["Kp"] for r in kp_data if isinstance(r.get("Kp"), (int, float))])
        dist["kp_mean"]  = float(kp_values.mean())
        dist["kp_std"]   = float(kp_values.std())
        dist["kp_p95"]   = float(np.percentile(kp_values, 95))
        dist["storm_prob"] = float((kp_values >= 5).mean())
        print(f"[real] Kp: mean={dist['kp_mean']:.2f}  p95={dist['kp_p95']:.1f}  storm_prob={dist['storm_prob']:.1%}")
    except Exception as e:
        print(f"[warn] Kp load failed ({e}) — using defaults")
        dist.update({"kp_mean": 2.0, "kp_std": 1.2, "kp_p95": 5.0, "storm_prob": 0.05})

    # --- Solar wind speed (NOAA) ---
    try:
        sw_path = os.path.join(_ROOT, "noaa", "solar", "solar_wind.json")
        with open(sw_path) as f:
            sw = json.load(f)
        speeds = np.array(list(sw["speed"].values()), dtype=float)
        speeds = speeds[~np.isnan(speeds)]
        dist["solar_wind_mean"] = float(speeds.mean())
        dist["solar_wind_std"]  = float(speeds.std())
        print(f"[real] Solar wind: mean={dist['solar_wind_mean']:.0f} km/s  std={dist['solar_wind_std']:.0f}")
    except Exception as e:
        print(f"[warn] Solar wind load failed ({e}) — using defaults")
        dist.update({"solar_wind_mean": 450.0, "solar_wind_std": 50.0})

    # --- Magnetic field Bt (NOAA) ---
    try:
        mf_path = os.path.join(_ROOT, "noaa", "solar", "magnetic_field.json")
        with open(mf_path) as f:
            mf_data = json.load(f)
        bt_values = np.array([float(r["bt"]) for r in mf_data if r.get("bt") not in (None, "", "null")])
        dist["bt_mean"] = float(bt_values.mean())
        dist["bt_std"]  = float(bt_values.std())
        dist["bt_p95"]  = float(np.percentile(bt_values, 95))
        print(f"[real] Magnetic Bt: mean={dist['bt_mean']:.2f} nT  p95={dist['bt_p95']:.2f}")
    except Exception as e:
        print(f"[warn] Magnetic field load failed ({e}) — using defaults")
        dist.update({"bt_mean": 5.0, "bt_std": 2.5, "bt_p95": 9.5})

    # --- TLE altitude distribution (CelesTrak) ---
    try:
        tle_path = os.path.join(_ROOT, "processed", "collision_event.csv")
        tle_df = pd.read_csv(tle_path, usecols=["MEAN_MOTION", "INCLINATION"])
        tle_df = tle_df[(tle_df.MEAN_MOTION > 0)]
        tle_df["altitude_km"] = ((398600.4418 / ((tle_df.MEAN_MOTION * 2 * np.pi / 86400) ** 2)) ** (1/3)) - 6371
        leo = tle_df[(tle_df.altitude_km >= 200) & (tle_df.altitude_km <= 2000)]
        dist["leo_alt_mean"]  = float(leo.altitude_km.mean())
        dist["leo_alt_std"]   = float(leo.altitude_km.std())
        dist["leo_incl_mean"] = float(leo.INCLINATION.mean())
        dist["leo_incl_std"]  = float(leo.INCLINATION.std())
        print(f"[real] LEO altitude: mean={dist['leo_alt_mean']:.0f} km  std={dist['leo_alt_std']:.0f}")
        print(f"[real] LEO inclination: mean={dist['leo_incl_mean']:.1f}°  std={dist['leo_incl_std']:.1f}°")
    except Exception as e:
        print(f"[warn] TLE load failed ({e}) — using defaults")
        dist.update({"leo_alt_mean": 550.0, "leo_alt_std": 80.0,
                     "leo_incl_mean": 53.0, "leo_incl_std": 20.0})

    return dist


# ---------------------------------------------------------------------------
# Healthy sample — calibrated to real orbital environment
# ---------------------------------------------------------------------------

def sample_healthy(dist: dict) -> dict:
    row = {f: np.random.normal(cfg["mean"], cfg["std"])
           for f, cfg in FEATURE_SCHEMA.items()}

    # Real-world calibration: radiation dose scales with altitude & Bt
    alt = max(200, np.random.normal(dist["leo_alt_mean"], dist["leo_alt_std"] * 0.5))
    alt_factor = (alt / 550.0) ** 0.6           # radiation ∝ altitude^0.6 (rough LEO model)
    bt_factor  = max(0.5, np.random.normal(dist["bt_mean"], dist["bt_std"]) / 5.0)
    row["radiation_dose"] = abs(np.random.normal(
        FEATURE_SCHEMA["radiation_dose"]["mean"] * alt_factor * bt_factor,
        FEATURE_SCHEMA["radiation_dose"]["std"]
    ))

    # Real-world: solar panel current slightly affected by solar wind speed
    wind_speed = np.random.normal(dist["solar_wind_mean"], dist["solar_wind_std"])
    wind_factor = 1.0 + 0.03 * ((wind_speed - 450) / 100)    # ±3% effect
    row["solar_panel_current"] = abs(np.random.normal(
        FEATURE_SCHEMA["solar_panel_current"]["mean"] * wind_factor,
        FEATURE_SCHEMA["solar_panel_current"]["std"]
    ))

    # Real-world: magnetic field Bt affects gyro rate micro-disturbances
    bt_sample = max(0.1, np.random.normal(dist["bt_mean"], dist["bt_std"]))
    gyro_noise = 0.005 * (bt_sample / dist["bt_mean"])
    row["gyro_rate"] = abs(row["gyro_rate"] + np.random.normal(0, gyro_noise))

    return row


# ---------------------------------------------------------------------------
# Fault injection — continuous ranges starting just outside band (spec §2.3)
# ---------------------------------------------------------------------------

def inject_fault(row: dict, fault_type: str, dist: dict) -> dict:
    row = dict(row)

    if fault_type == "thruster_overheat":
        row["thruster_temp"] = np.random.uniform(61, 400)

    elif fault_type == "battery_fault":
        row["battery_voltage"] = np.random.uniform(22.0, 26.4)
        row["battery_temp"]    = np.random.uniform(35.1, 55.0)

    elif fault_type == "attitude_loss":
        row["gyro_rate"] = np.random.uniform(0.61, 3.0)
        if np.random.random() < 0.5:
            row["reaction_wheel_rpm"] = np.random.uniform(200, 1499)
        else:
            row["reaction_wheel_rpm"] = np.random.uniform(5001, 7000)

    elif fault_type == "power_drop":
        # Real calibration: solar wind dropout drives current low
        wind_ratio = max(0.3, np.random.normal(dist["solar_wind_mean"], dist["solar_wind_std"]) / 450)
        base_drop = np.random.uniform(1.0, 6.9)
        row["solar_panel_current"] = base_drop * wind_ratio
        row["battery_voltage"]     = np.random.uniform(23.0, 26.4)

    elif fault_type == "solar_storm":
        # Real calibration: storm severity sampled from real Kp distribution tail
        storm_kp = np.random.uniform(5.0, max(7.0, dist["kp_p95"] * 1.5))
        storm_factor = storm_kp / 5.0
        row["radiation_dose"]        = np.random.uniform(20.1, 100.0) * storm_factor * 0.5
        row["comms_signal_strength"] = np.random.uniform(-120.0, -96.0)
        # Real: high Bt during storm → gyro disturbance
        row["gyro_rate"]             = np.random.uniform(0.61, 2.5) * min(2.0, dist["bt_p95"] / 5.0)

    elif fault_type == "solar_storm_compound":
        # Combined: storm + power degradation (real-world co-occurrence pattern)
        row["radiation_dose"]        = np.random.uniform(20.1, 80.0)
        row["comms_signal_strength"] = np.random.uniform(-110.0, -96.0)
        row["solar_panel_current"]   = np.random.uniform(4.0, 6.9)   # storm shadows panel output
        row["battery_voltage"]       = np.random.uniform(24.0, 26.4)

    return row


FAULT_TYPES = [
    "thruster_overheat",
    "battery_fault",
    "attitude_loss",
    "power_drop",
    "solar_storm",
    "solar_storm_compound",
]


def sigmoid_p_fail(score: int) -> float:
    return 1 / (1 + np.exp((score - 70) / 8))


def build_row(readings: dict, is_anomaly: int) -> dict:
    score, _, _, _ = compute_health_score(readings)
    p_fail   = sigmoid_p_fail(score)
    will_fail = int(np.random.random() < p_fail)
    return {**readings, "is_anomaly": is_anomaly, "will_fail": will_fail, "health_score": score}


def main():
    print("Loading real-world calibration data...")
    dist = _load_real_distributions()
    print()

    n_healthy = int(N_TOTAL * HEALTHY_FRAC)
    n_faulty  = N_TOTAL - n_healthy

    rows = []

    print(f"Generating {n_healthy} healthy rows (real-world calibrated)...")
    for _ in range(n_healthy):
        rows.append(build_row(sample_healthy(dist), is_anomaly=0))

    print(f"Generating {n_faulty} faulty rows ({len(FAULT_TYPES)} fault modes)...")
    for i in range(n_faulty):
        base  = sample_healthy(dist)
        fault = FAULT_TYPES[i % len(FAULT_TYPES)]
        readings = inject_fault(base, fault, dist)

        if np.random.random() < DOUBLE_FAULT_PROB:
            second = FAULT_TYPES[(i + 2) % len(FAULT_TYPES)]
            readings = inject_fault(readings, second, dist)

        rows.append(build_row(readings, is_anomaly=1))

    df = pd.DataFrame(rows)

    output_dir  = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_data.csv")
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} rows -> {output_path}")
    print(f"Anomaly rate  : {df['is_anomaly'].mean():.1%}  (expected ~30%)")
    print(f"Failure rate  : {df['will_fail'].mean():.1%}  (expected ~20-35%)")
    print(f"Health score  : mean={df['health_score'].mean():.1f}  "
          f"min={df['health_score'].min()}  max={df['health_score'].max()}")
    print("\nFailure rate by anomaly label:")
    print(df.groupby("is_anomaly")["will_fail"].mean().rename("failure_rate"))


if __name__ == "__main__":
    main()
