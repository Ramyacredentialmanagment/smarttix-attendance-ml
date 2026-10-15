from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import sqlite3
import os
from datetime import datetime, timezone

app = FastAPI(title="SmartTix Attendance ML Service", version="1.0.0")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model pipeline
MODEL_PATH = "attendance_model.joblib"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

FALLBACK_STORE = {
    "evt_101": {"event_type": "Conference", "tickets_sold": 500, "is_weekend": 1},
    "evt_102": {"event_type": "Concert", "tickets_sold": 800, "is_weekend": 1},
    "evt_103": {"event_type": "Workshop", "tickets_sold": 100, "is_weekend": 0},
    "EVT-12345": {"event_type": "Conference", "tickets_sold": 500, "is_weekend": 1},
    "Race 5657": {"event_type": "Concert", "tickets_sold": 1200, "is_weekend": 1}
}

class ForecastRequest(BaseModel):
    forecastHorizonDays: int = 7

class ForecastResponse(BaseModel):
    predictedAttendance: int
    ticketsSold: int
    modelVersion: str
    forecastedAt: str

def fetch_event(event_id: str):
    if os.path.exists("events.db"):
        try:
            conn = sqlite3.connect("events.db")
            cursor = conn.cursor()
            cursor.execute("SELECT event_type, tickets_sold, is_weekend FROM events WHERE event_id = ?", (event_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {"event_type": row[0], "tickets_sold": int(row[1]), "is_weekend": int(row[2])}
        except Exception:
            pass
    return FALLBACK_STORE.get(event_id)

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SmartTix Attendance Service is Running. (index.html not found)</h1>"

@app.post("/api/events/{eventId}/attendance-forecast", response_model=ForecastResponse)
def get_attendance_forecast(eventId: str, request: ForecastRequest):
    event = fetch_event(eventId)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event '{eventId}' not found in database.")

    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact attendance_model.joblib not found.")

    df = pd.DataFrame([{
        "event_type": event["event_type"],
        "tickets_sold": event["tickets_sold"],
        "days_to_event": request.forecastHorizonDays,
        "is_weekend": event["is_weekend"]
    }])

    raw_pred = model.predict(df)[0]
    bounded_pred = int(max(0, min(round(raw_pred), event["tickets_sold"])))

    return ForecastResponse(
        predictedAttendance=bounded_pred,
        ticketsSold=event["tickets_sold"],
        modelVersion="v1.0.0",
        forecastedAt=datetime.now(timezone.utc).isoformat()
    )
