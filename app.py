import numpy as np
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Model + scaler
scaler = StandardScaler()
model = LogisticRegression()

# Generate training data
def generate_data(n=1000):
    np.random.seed(42)
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.random.randint(20, 120, n)
    total_charges = tenure * monthly_charges + np.random.randint(0, 500, n)
    contract = np.random.randint(0, 3, n)

    churn = (
        (contract == 0).astype(int) * 2 +
        (monthly_charges > 80).astype(int) +
        (tenure < 12).astype(int) +
        np.random.randint(0, 2, n)
    ) >= 3

    X = np.column_stack([tenure, monthly_charges, total_charges, contract])
    y = churn.astype(int)
    return X, y

# Train model
X, y = generate_data()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)

@app.route("/")
def home():
    return jsonify({
        "message": "Customer Churn Prediction API",
        "example": {
            "tenure": 10,
            "monthly_charges": 90,
            "total_charges": 900,
            "contract": 0
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = np.array([[
            float(data["tenure"]),
            float(data["monthly_charges"]),
            float(data["total_charges"]),
            float(data["contract"])
        ]])
    except:
        return jsonify({"error": "Invalid input"}), 400

    features_scaled = scaler.transform(features)

    pred = int(model.predict(features_scaled)[0])
    prob = float(model.predict_proba(features_scaled)[0][1])

    return jsonify({
        "prediction": pred,
        "label": "churn" if pred == 1 else "no_churn",
        "churn_probability": round(prob, 4),
        "risk_level": "high" if prob > 0.6 else "medium" if prob > 0.3 else "low"
    })

if __name__ == "__main__":
    app.run(debug=True)
