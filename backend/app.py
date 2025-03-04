from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import mysql.connector
import numpy as np
from prediction import predict_failure  # Keep only the necessary import

app = Flask(__name__, static_folder="../frontend/static", template_folder="../frontend")
CORS(app)

# ✅ Load the trained health model
model_path = "models/failure_model.pkl"
health_status_model = joblib.load(model_path)

# ✅ MySQL Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="regula",
        database="VehicleRecord"
    )

# ✅ Fetch latest recorded health data
def fetch_latest_health_record():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM health_metrics ORDER BY date DESC LIMIT 1")
    result = cursor.fetchone()
    conn.close()
    return result

# ✅ Store new health data into MySQL
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
 
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Route to render the Failure Prediction Page
@app.route("/failure-prediction", methods=["GET"])
def failure_prediction():
    return render_template("bikeFut.html")

#  Route to render the Health Status Prediction Page
@app.route("/health-status", methods=["GET"])
def health_status():
    return render_template("users.html")

# API Route for Failure Prediction
@app.route("/predict", methods=["POST"])
def get_prediction():
    try:
        # Debugging: Print received inputs
        print("🔍 Received Form Data:", request.form)

        battery_health = float(request.form["battery_health"])
        motor_temp = float(request.form["motor_temp"])
        brake_wear = float(request.form["brake_wear"])
        suspension_stress = float(request.form["suspension_stress"])
        driver_score = float(request.form["driver_score"])
        road_type = int(request.form["road_type"])

        input_list = [battery_health, motor_temp, brake_wear, suspension_stress, driver_score, road_type]
        
        # Debugging: Print input list before prediction
        print("📊 Input Data for Prediction:", input_list)

        result = predict_failure(input_list)

        # Debugging: Print model result
        print("✅ Prediction Output:", result)

        return render_template("bikeFut.html", failure_risk=result["failure_risk"],
                               failed_component=result["failed_component"], solution=result["solution"])

    except Exception as e:
        print("❌ Error:", str(e))
        return render_template("bikeFut.html", error=str(e))

 
@app.route("/predict-health", methods=["POST"])
def predict_health():
    try:
        battery_health_now = float(request.form.get("battery_health_now", 0))
        motor_temp_now = float(request.form.get("motor_temp_now", 0))
        avg_battery_last_week = float(request.form.get("avg_battery_last_week", 0))
        avg_motor_temp_last_week = float(request.form.get("avg_motor_temp_last_week", 0))

        input_data = {
            "battery_health_now": battery_health_now,
            "motor_temperature_now": motor_temp_now,
            "avg_battery_last_week": avg_battery_last_week,
            "avg_motor_temp_last_week": avg_motor_temp_last_week,
        }

        input_array = np.array([[battery_health_now, motor_temp_now, avg_battery_last_week, avg_motor_temp_last_week]])
        prediction = health_status_model.predict(input_array)[0]

        status_mapping = {0: "Improved", 1: "Degraded", 2: "Maintained"}
        predicted_status = status_mapping.get(prediction, "Unknown")

        #  Fetch last recorded data
        last_record = fetch_latest_health_record()
        
        if last_record:
            if predicted_status == last_record["status"]:
                comparison_result = "Maintained"
            elif predicted_status == "Degraded":
                comparison_result = "Vehicle Performance Has Degraded."
            else:
                comparison_result = "Vehicle Performance Has Improved."
        else:
            comparison_result = "No past record found. This is the first record."

        #  Store new data into DB
        store_health_data(input_data, predicted_status)

        return render_template("users.html", health_status=predicted_status, comparison=comparison_result)

    except Exception as e:
        return render_template("users.html", error=str(e))

#  Fix: Correct __name__ check
if __name__ == "__main__":
    app.run(debug=True, port=5000)