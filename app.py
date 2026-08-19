from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load encoder
with open("encoder.pkl", "rb") as file:
    ohencoder = pickle.load(file)


@app.route("/")
def home():
    # return jsonify({"message": "Welcome to the Stock Prediction API!"})
    return render_template("index.html")    

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    # print(data)

    # Get only 3 inputs from user
    Category_input = data["category"]
    Current_Stock_input = float(data["current_stock"])
    Units_Sold_Last_7_Days_input = float(data["units_sold"])

    # Calculate the other two values
    Average_Daily_Sales_input = Units_Sold_Last_7_Days_input / 7

    Demand_Forecast_input = Average_Daily_Sales_input * 7

    # Create dataframe
    new_data = pd.DataFrame({
        "Category": [Category_input],
        "Current_Stock": [Current_Stock_input],
        "Units_Sold_Last_7_Days": [Units_Sold_Last_7_Days_input],
        "Average_Daily_Sales": [Average_Daily_Sales_input],
        "Demand_Forecast": [Demand_Forecast_input]
    })

    # Encode category
    encoded = ohencoder.transform(new_data[["Category"]])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohencoder.get_feature_names_out(["Category"])
    )

    # Create final input
    new_X = pd.concat([
        encoded_df,
        new_data[[
            "Current_Stock",
            "Units_Sold_Last_7_Days",
            "Average_Daily_Sales",
            "Demand_Forecast"
        ]]
    ], axis=1)

    # Prediction
    prediction = model.predict(new_X)[0]

    if prediction == 1:
        result = "IT NEED TO BE RESTOCKED"
    else:
        result = "DO NOT RESTOCK"

    return jsonify({
        "prediction": int(prediction),
        "result": result,
        
    })


if __name__ == "__main__":
    app.run(debug=True)