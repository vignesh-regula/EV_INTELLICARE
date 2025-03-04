import mysql.connector
import joblib
import numpy as np

failure_model = joblib.load("models/failure_model.pkl")
component_model = joblib.load("models/component_model.pkl")
health_status_model = joblib.load("models/health_status_model.pkl")

# Failure Solutions
solutions = {
    0: "Battery: Reduce rapid acceleration & avoid deep discharges.",
    1: "Brakes: Avoid harsh braking. Inspect brake fluid & pads regularly.",
    2: "Motor: Monitor overheating. Ensure proper airflow around motor.",
    3: "Suspension: Avoid rough roads. Adjust preload settings & check shock absorbers.",
}
def predict_failure(input_data):
    # ✅ Reload the model inside the function
    failure_model = joblib.load("models/failure_model.pkl")
    component_model = joblib.load("models/component_model.pkl")

    failure_risk = failure_model.predict([input_data])[0]

    if failure_risk == 1:
        failed_component = component_model.predict([input_data])[0]
        solution = solutions.get(failed_component, "No specific solution found.")
    else:
        failed_component = "None"
        solution = "Your bike is in good condition."
    
    return {"failure_risk": failure_risk, "failed_component": failed_component, "solution": solution}

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="yourpassword",
        database="ev_health"
    )

# Fetch Latest Health Record from MySQL
def fetch_latest_health_record():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM health_metrics ORDER BY date DESC LIMIT 1")
    result = cursor.fetchone()
    conn.close()
    return result

# Store New Health Data into MySQL
def store_health_data(data, status):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = '''
        INSERT INTO health_metrics (battery_health_now, motor_temperature_now, 
                                    avg_battery_last_week, avg_motor_temp_last_week, status)
        VALUES (%s, %s, %s, %s, %s)
    '''
    cursor.execute(query, (
        data["battery_health_now"],
        data["motor_temperature_now"],
        data["avg_battery_last_week"],
        data["avg_motor_temp_last_week"],
        status
    ))

    conn.commit()
    conn.close()

# Predict Health Status
def predict_health_status(input_data):
    model = joblib.load("models/health_status_model.pkl")

    # Convert input data to array for prediction
    input_array = np.array([[
        input_data["battery_health_now"],
        input_data["motor_temperature_now"],
        input_data["avg_battery_last_week"],
        input_data["avg_motor_temp_last_week"]
    ]])

    prediction = model.predict(input_array)[0]
    status_mapping = {0: "Improved", 1: "Degraded", 2: "Maintained"}
    
    return status_mapping[prediction]
