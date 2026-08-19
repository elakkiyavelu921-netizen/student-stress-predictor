from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load Logistic Regression model
model = joblib.load("student_stress_logistic_model2.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get JSON data from HTML
        data = request.get_json()

        # Create student DataFrame
        student = pd.DataFrame([{
            "academic_pressure_score": float(data["academic_pressure_score"]),
            "anxiety_score": float(data["anxiety_score"]),
            "depression_score": float(data["depression_score"]),
            "social_support_score": float(data["social_support_score"]),
            "screen_time_hours": float(data["screen_time_hours"]),
            "daily_sleep_hours": float(data["daily_sleep_hours"]),
            "attendance_percentage": float(data["attendance_percentage"]),
            "cgpa": float(data["cgpa"])
        }])

        # Predict stress level
        prediction = model.predict(student)[0]

        # Get probabilities
        probabilities = model.predict_proba(student)[0]

        probability_data = {}

        for class_name, probability in zip(
            model.classes_,
            probabilities
        ):
            probability_data[class_name] = round(
                probability * 100,
                2
            )

        # Send result to HTML
        return jsonify({
            "prediction": prediction,
            "probabilities": probability_data
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Run Flask
if __name__ == "__main__":
    app.run(debug=True)