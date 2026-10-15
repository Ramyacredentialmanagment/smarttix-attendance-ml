import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# 1. Load dataset
data = pd.read_csv("events.csv")
print("First five events:")
print(data.head())

# 2. Features and target
X = data[["event_type", "tickets_sold", "days_to_event", "is_weekend"]]
y = data["actual_attendance"]

# 3. Preprocessor for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("event_type", OneHotEncoder(handle_unknown="ignore"), ["event_type"])
    ],
    remainder="passthrough",
)

# 4. Pipeline
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Fit
model.fit(X_train, y_train)

# 7. Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)

print("\nActual attendance:", list(y_test))
print("Predicted attendance:", predictions.round(1))
print("Test MAE:", round(mae, 2), "people")
print("Test RMSE:", round(rmse, 2), "people")

# 8. Save artifact
joblib.dump(model, "attendance_model.joblib")
print("\nModel saved successfully as 'attendance_model.joblib'.")
