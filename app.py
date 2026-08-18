from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model + feature list
model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.get("/")
def home():
    return jsonify({
        "message": "House Price Prediction API is running",
        "required_features": feature_columns,
        "how_to_use": {
            "GET /predict": "Shows instructions",
            "POST /predict": "Send JSON to get prediction"
        }
    })


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # If user opens /predict in browser -> GET request
    if request.method == "GET":
        return jsonify({
            "message": "Use POST with JSON body to get prediction.",
            "required_features": feature_columns,
            "example_json": {
                "MedInc": 8.3,
                "HouseAge": 41,
                "AveRooms": 6.98,
                "AveBedrms": 1.02,
                "Population": 322,
                "AveOccup": 2.55,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        })

    # POST request (prediction)
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON received. Please send JSON body."}), 400

    # Build one-row dataframe with correct column order
    row = {}
    missing = []
    for col in feature_columns:
        if col not in data:
            missing.append(col)
        else:
            row[col] = data[col]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing,
            "required_features": feature_columns
        }), 400

    X = pd.DataFrame([row], columns=feature_columns)
    pred = float(model.predict(X)[0])

    return jsonify({
        "prediction_med_house_value": round(pred, 3),
        "prediction_dollars": round(pred * 100000, 0),
        "note_english": "Target is Median House Value in $100,000 units (sklearn California Housing)."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)