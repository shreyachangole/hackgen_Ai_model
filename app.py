from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("waste_collection_model.pkl")
encoder = joblib.load("label_encoder.pkl")

THRESHOLD = 0.7  # 🔥 key improvement

@app.route("/")
def home():
    return "Smart Waste Bin API Running 🚀"

# =========================
# 🔹 Single Prediction
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        location = data["location_type"].capitalize()
        location_encoded = encoder.transform([location])[0]

        features = [
            data["fill_percent"],
            location_encoded,
            data["is_weekend"],
            data["hour"]
        ]

        final_input = np.array([features])

        # 🔥 probability आधारित prediction
        prob = model.predict_proba(final_input)[0][1]
        prediction = prob > THRESHOLD

        return jsonify({
            "collection_required": bool(prediction),
            "confidence": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# 🔥 Multiple Prediction
# =========================
@app.route("/predict_multiple", methods=["POST"])
def predict_multiple():
    try:
        data = request.json
        bins = data["bins"]

        X = []
        ids = []

        for b in bins:
            location = encoder.transform([b["location_type"].capitalize()])[0]

            fill_percent = b["fill_percent"]
            hour = b["hour"]

            # 🔥 NEW FEATURES
            fill_rate = fill_percent / 10   # approx (since hours_to_full नाहीये)
            is_peak_hour = 1 if hour in [8,9,18,19] else 0

            X.append([
                fill_percent,
                location,
                b["is_weekend"],
                hour,
                
            ])

            ids.append(b["bin_id"])

        X = np.array(X)

        # 🔥 Probability based prediction
        proba = model.predict_proba(X)[:,1]
        preds = proba > 0.7

        return jsonify({
            "total_bins": len(bins),
            "need_collection": int(sum(preds)),
            "bin_ids": [ids[i] for i in range(len(ids)) if preds[i]]
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
   app.run(debug=True)