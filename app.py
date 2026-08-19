from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained Random Forest model
model = joblib.load("student_stress_logistic_model.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    academic_pressure_score = int(
        request.form["academic_pressure_score"]
    )

    anxiety_score = int(
        request.form["anxiety_score"]
    )

    depression_score = int(
        request.form["depression_score"]
    )

    social_support_score = int(
        request.form["social_support_score"]
    )

    screen_time_hours = float(
        request.form["screen_time_hours"]
    )

    daily_sleep_hours = float(
        request.form["daily_sleep_hours"]
    )

    attendance_percentage = float(
        request.form["attendance_percentage"]
    )

    cgpa = float(
        request.form["cgpa"]
    )


    # Create student DataFrame
    new_student = pd.DataFrame({
        "academic_pressure_score": [academic_pressure_score],
        "anxiety_score": [anxiety_score],
        "depression_score": [depression_score],
        "social_support_score": [social_support_score],
        "screen_time_hours": [screen_time_hours],
        "daily_sleep_hours": [daily_sleep_hours],
        "attendance_percentage": [attendance_percentage],
        "cgpa": [cgpa]
    })


    # Predict stress level
    prediction = model.predict(new_student)[0]


    # Prediction probabilities
    probabilities = model.predict_proba(new_student)[0]


    probability_data = []

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):
        probability_data.append({
            "class_name": class_name,
            "probability": round(probability * 100, 2)
        })


    return render_template(
        "index.html",
        prediction=prediction,
        probabilities=probability_data
    )


if __name__ == "__main__":
    app.run(debug=True)