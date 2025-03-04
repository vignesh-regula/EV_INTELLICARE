import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

 
DB_USER = "root"
DB_PASSWORD = "regula"
DB_HOST = "localhost"
DB_NAME = "VehicleRecord"
TABLE_NAME = "health_metrics"

 
def get_db_connection():
    return create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

 
def fetch_data():
    engine = get_db_connection()
    query = f"SELECT * FROM {TABLE_NAME} ORDER BY date DESC LIMIT 100"
    df = pd.read_sql(query, engine)

    if df.empty:
        print("❌ ERROR: No data found in health_metrics table.")
        return None

    print("✅ Data successfully loaded from MySQL!")
 
    if "date" in df.columns:
        df.drop(columns=["date"], inplace=True)
 
    column_mapping = {
        "battery_health_now": "Battery_Health_Now",
        "motor_temperature_now": "Motor_Temperature_Now",
        "avg_battery_last_week": "Avg_Battery_Last_Week",
        "avg_motor_temp_last_week": "Avg_Motor_Temp_Last_Week",
        "status": "Status"
    }
    df.rename(columns=column_mapping, inplace=True)
 
    status_mapping = {"Improved": 0, "Degraded": 1, "Maintained": 2}
    df["Status"] = df["Status"].map(status_mapping)

    return df

 
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
 
df = fetch_data()

if df is None:
    print("❌ ERROR: Training aborted due to missing data.")
    exit()
 
X = df.drop(["Status"], axis=1)
y = df["Status"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
health_status_model = RandomForestClassifier(n_estimators=100, random_state=42)
health_status_model.fit(X_train, y_train)
 
health_status_model_path = os.path.join(MODEL_DIR, "health_status_model.pkl")
joblib.dump(health_status_model, health_status_model_path)
print(f"✅ Health Status Prediction Model saved at {health_status_model_path}")