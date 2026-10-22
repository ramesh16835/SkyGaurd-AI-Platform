import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# =========================================================================
# 1. CONFIGURATION & RUNTIME HYPERPARAMETERS
# =========================================================================
ARTIFACTS_DIR = "./skyguard_model_package"
BEST_THRESHOLD = 0.037624
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 2. NEURAL NETWORK ARCHITECTURE (Matches Trained Weights)
# =========================================================================
class AWSAutoencoder(nn.Module):
    def __init__(self, n_features: int = 5, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.enc1 = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.enc2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        self.dec1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.dec2 = nn.LSTM(hidden_dim, n_features, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        out, _ = self.enc1(x)
        _, (h_n, _) = self.enc2(out)
        latent = h_n.squeeze(0).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.dec1(latent)
        reconstruction, _ = self.dec2(dec_out)
        return reconstruction


# =========================================================================
# 3. REAL-TIME STREAMING KALMAN FILTER (Self-Healing Engine)
# =========================================================================
class StreamingKalmanFilter:
    """
    1D Kalman Filter for real-time sensor smoothing and missing/anomalous value imputation.
    """
    def __init__(self, q: float = 1e-3, r: float = 0.1):
        self.q = q      # Process variance (atmospheric rate of change)
        self.r = r      # Measurement noise uncertainty
        self.x = None   # State estimate
        self.p = 1.0    # Estimation error covariance

    def step(self, value: Optional[float], is_anomalous: bool = False) -> float:
        # Initial cold start
        if self.x is None:
            self.x = value if value is not None and not np.isnan(value) else 25.0
            self.p = 1.0
            return float(self.x)

        # Time Update (Predict next state)
        self.p = self.p + self.q

        # Measurement Update (Skip if anomalous or missing, rely on state propagation)
        if value is None or np.isnan(value) or is_anomalous:
            return float(self.x)

        # Kalman Gain update
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (value - self.x)
        self.p = (1.0 - k) * self.p
        return float(self.x)


# =========================================================================
# 4. PYDANTIC SCHEMAS (Validation & Documentation)
# =========================================================================
class HourlyReading(BaseModel):
    temperature_C: float = Field(..., description="Air temperature in Celsius")
    pressure_hPa: float = Field(..., description="Barometric pressure in hPa")
    relative_humidity_percent: float = Field(..., ge=0, le=100, description="Relative humidity %")
    hour: int = Field(..., ge=0, le=23, description="Hour of observation (0-23)")

class TelemetryWindowRequest(BaseModel):
    station_id: str = Field(default="AWS-DELHI-001", example="AWS-DELHI-001")
    readings: List[HourlyReading] = Field(..., min_items=24, max_items=24, description="Chronological 24-hour sequence")

class DiagnosisReport(BaseModel):
    status: str
    is_anomaly: bool
    fault_type: str
    reconstruction_mse: float
    threshold: float
    alert_triggered: bool
    observed_values: dict
    healed_values: dict


# =========================================================================
# 5. FASTAPI APPLICATION SETUP & LIFECYCLE
# =========================================================================
app = FastAPI(
    title="SkyGuard AWS Anomaly Engine",
    description="Real-Time Meteorological Fault Detection and Self-Healing Telemetry API",
    version="1.0.0"
)

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

# Global model and scaler holders
model = AWSAutoencoder(n_features=5, hidden_dim=64, latent_dim=32).to(DEVICE)
scaler = None

@app.on_event("startup")
def startup_event():
    global model, scaler
    model_path = os.path.join(ARTIFACTS_DIR, "lstm_autoencoder_best.pt")
    scaler_path = os.path.join(ARTIFACTS_DIR, "robust_scaler.pkl")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded weights from: {model_path}")
    else:
        print(f"Warning: Model weights not found at {model_path}")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded RobustScaler from: {scaler_path}")
    else:
        print(f"Warning: RobustScaler not found at {scaler_path}")


# =========================================================================
# 6. API ENDPOINTS
# =========================================================================
@app.get("/health")
def healthcheck():
    """Confirms service vitality and artifact load status."""
    return {
        "status": "ONLINE",
        "artifacts_loaded": scaler is not None,
        "device": str(DEVICE)
    }

@app.post("/api/v1/diagnose", response_model=DiagnosisReport, status_code=status.HTTP_200_OK)
def diagnose_telemetry(payload: TelemetryWindowRequest):
    """
    Ingests a 24-hour telemetry window, applies multi-tier anomaly detection,
    and returns fault classification with self-healed imputations.
    """
    if scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model scaler or weights not loaded. Verify ./skyguard_model_package."
        )

    # 1. Feature Engineering (Hour -> Cyclical sin/cos)
    matrix = []
    for item in payload.readings:
        hour_rad = 2 * np.pi * item.hour / 24.0
        matrix.append([
            item.temperature_C,
            item.pressure_hPa,
            item.relative_humidity_percent,
            np.sin(hour_rad),
            np.cos(hour_rad)
        ])

    raw_arr = np.array(matrix, dtype=np.float32)

    # 2. Scale & Inference
    scaled_arr = scaler.transform(raw_arr)
    input_tensor = torch.tensor(scaled_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon_tensor = model(input_tensor)
        # The configured threshold was calibrated for the full 24-hour window.
        # Scoring only the final timestep makes normal readings at the edge of
        # the sequence look anomalous much more often than the threshold allows.
        param_mse = torch.mean((recon_tensor[:, :, :3] - input_tensor[:, :, :3]) ** 2).item()

    # 3. Physics & Statistical Signal Checks
    t_series = raw_arr[:, 0]
    p_series = raw_arr[:, 1]
    rh_series = raw_arr[:, 2]

    rh_stuck = bool(np.std(rh_series[-6:]) < 0.05)
    t_stuck = bool(np.std(t_series[-6:]) < 0.05)
    p_jump = bool(abs(p_series[-1] - p_series[-2]) > 5.0)
    lstm_anomaly = bool(param_mse > BEST_THRESHOLD)

    # Severe weather suppression (Sudden squall / cold front)
    delta_p = p_series[-1] - p_series[-2]
    delta_rh = rh_series[-1] - rh_series[-2]
    is_severe_storm = (delta_p < -1.8) and (delta_rh > 12.0)

    is_fault = (lstm_anomaly or rh_stuck or t_stuck or p_jump) and not is_severe_storm

    # 4. Fault Categorization
    fault_type = "NORMAL"
    if is_severe_storm:
        fault_type = "SEVERE_WEATHER_PASS"
    elif rh_stuck:
        fault_type = "FAULT_FLATLINE_HUMIDITY"
    elif t_stuck:
        fault_type = "FAULT_FLATLINE_TEMPERATURE"
    elif p_jump:
        fault_type = "FAULT_STEP_SPIKE_PRESSURE"
    elif lstm_anomaly:
        fault_type = "FAULT_TEMPORAL_DRIFT_COMPLEX"

    # 5. Self-Healing with History Warm-up
    # Kalman filter ko sequence ke 0 se 22 clean steps se warm-up karein
    k_temp = StreamingKalmanFilter()
    k_pres = StreamingKalmanFilter()
    k_hum = StreamingKalmanFilter()

    for idx in range(len(t_series) - 1):
        k_temp.step(float(t_series[idx]), is_anomalous=False)
        k_pres.step(float(p_series[idx]), is_anomalous=False)
        k_hum.step(float(rh_series[idx]), is_anomalous=False)

    # 23rd (current) reading par step karein with actual fault status
    healed_t = k_temp.step(float(t_series[-1]), is_anomalous=is_fault)
    healed_p = k_pres.step(float(p_series[-1]), is_anomalous=is_fault)
    healed_rh = k_hum.step(float(rh_series[-1]), is_anomalous=is_fault)

    # 6. Response Payload
    return DiagnosisReport(
        status="FAULT_FLAGGED" if is_fault else "HEALTHY",
        is_anomaly=is_fault,
        fault_type=fault_type,
        reconstruction_mse=round(param_mse, 6),
        threshold=BEST_THRESHOLD,
        alert_triggered=is_fault,
        observed_values={
            "temperature_C": float(t_series[-1]),
            "pressure_hPa": float(p_series[-1]),
            "relative_humidity_percent": float(rh_series[-1])
        },
        healed_values={
            "temperature_C": round(healed_t, 2),
            "pressure_hPa": round(healed_p, 2),
            "relative_humidity_percent": round(healed_rh, 2)
        }
    )
