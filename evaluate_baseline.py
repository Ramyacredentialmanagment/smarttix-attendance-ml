import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# 1. Load data
data = pd.read_csv("events.csv")
X = data[["event_type", "tickets_sold", "days_to_event", "is_weekend"]]
y = data["actual_attendance"]

# 2. Split exactly as during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Simple Heuristic Baseline (Train check-in rate * test tickets)
train_attendance_rate = (y_train / X_train["tickets_sold"]).mean()
baseline_preds = X_test["tickets_sold"] * train_attendance_rate

base_mae = mean_absolute_error(y_test, baseline_preds)
base_rmse = root_mean_squared_error(y_test, baseline_preds)

# 4. Load trained Random Forest model
model = joblib.load("attendance_model.joblib")
ml_preds = model.predict(X_test)

ml_mae = mean_absolute_error(y_test, ml_preds)
ml_rmse = root_mean_squared_error(y_test, ml_preds)

print(f"Historical Attendance Rate: {train_attendance_rate * 100:.1f}%\n")
print(f"{'Method':<22} | {'MAE (attendees)':<16} | {'RMSE (attendees)':<16}")
print("-" * 58)
print(f"{'Heuristic Baseline':<22} | {base_mae:<16.2f} | {base_rmse:<16.2f}")
print(f"{'Random Forest ML':<22} | {ml_mae:<16.2f} | {ml_rmse:<16.2f}")
