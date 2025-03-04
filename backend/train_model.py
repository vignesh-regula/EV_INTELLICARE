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
TABLE_NAME = "mainmodel"
 
def get_db_connection():
    return create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")



def fetch_data():
    engine = get_db_connection()
    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql(query, engine)
    print("✅ Data successfully loaded from MySQL!")
    column_mapping = {
        "Driver_Score": "driver_score",
        "Predictive_Failed_Component": "failed_component",
        "Road_Type": "road_type"
    }
    df.rename(columns=column_mapping, inplace=True)
    required_columns = ["Vehicle_ID", "driver_score", "Battery_Health", "Motor_Temperature", 
                        "Brake_Wear", "Suspension_Stress", "road_type", "Past_Failures", "failed_component"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"❌ Missing columns in database: {missing_cols}")
    return df

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
df = fetch_data()
road_type_mapping = {
    "Highway": 0,
    "Urban": 1,
    "Rural": 2
}
df["road_type"] = df["road_type"].map(road_type_mapping)
df["road_type"].fillna(0, inplace=True) 
df["failure_risk"] = df["failed_component"].notna().astype(int)
X = df.drop(["Past_Failures", "failure_risk", "failed_component", "Vehicle_ID"], axis=1)
y_failure = df["failure_risk"]
y_component = df[df["failure_risk"] == 1]["failed_component"].dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y_failure, test_size=0.2, random_state=42)
failure_model = RandomForestClassifier(n_estimators=100, random_state=42)
failure_model.fit(X_train, y_train)

failure_model_path = os.path.join(MODEL_DIR, "failure_model.pkl")

joblib.dump(failure_model, failure_model_path)

print(f"✅ Failure Prediction Model saved at {failure_model_path}")
failure_indices = y_train[y_train == 1].index
X_train_comp = X_train.loc[failure_indices]
y_train_comp = y_component.loc[failure_indices]
min_length = min(len(X_train_comp), len(y_train_comp))
X_train_comp, y_train_comp = X_train_comp.iloc[:min_length], y_train_comp.iloc[:min_length]
component_model = RandomForestClassifier(n_estimators=100, random_state=42)
component_model.fit(X_train_comp, y_train_comp)
component_model_path = os.path.join(MODEL_DIR, "component_model.pkl")
joblib.dump(component_model, component_model_path)
print(f"✅ Component Prediction Model saved at {component_model_path}")