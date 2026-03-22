import numpy as np
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

np.random.seed(42)
n_samples = 1000

tenure = np.random.randint(1, 72, n_samples).astype(float)
monthly_charges = np.random.uniform(20, 120, n_samples)
total_charges = tenure * monthly_charges + np.random.normal(0, 50, n_samples)
contract = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]).astype(float)

churn_score = (
    -0.05 * tenure
    + 0.02 * monthly_charges
    - 0.5 * contract
    + np.random.normal(0, 0.5, n_samples)
)
churn_prob = 1 / (1 + np.exp(-churn_score))
y = (churn_prob > 0.5).astype(int)

X = np.column_stack([tenure, monthly_charges, total_charges, contract])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)


def get_risk_level(prob):
    if prob < 0.3:
        return "low"
    elif prob < 0.6:
        return "medium"
    return "high"


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "message": "Customer Churn Prediction API",
            "example_input": {
                "tenure": 12,
                "monthly_charges": 65.5,
                "total_charges": 786.0,
                "contract": 0,
            },
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    required_fields = ["tenure", "monthly_charges", "total_charges", "contract"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: '{field}'"}), 400

    values = {}
    for field in required_fields:
        try:
            values[field] = float(data[field])
        except (TypeError, ValueError):
            return jsonify({"error": f"Field '{field}' must be numeric"}), 400

    if values["contract"] not in (0.0, 1.0, 2.0):
        return jsonify({"error": "Field 'contract' must be 0, 1, or 2"}), 400

    features = np.array(
        [
            [
                values["tenure"],
                values["monthly_charges"],
                values["total_charges"],
                values["contract"],
            ]
        ]
    )

    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    churn_probability = float(model.predict_proba(features_scaled)[0][1])

    return jsonify(
        {
            "prediction": prediction,
            "label": "churn" if prediction == 1 else "no_churn",
            "churn_probability": round(churn_probability, 4),
            "risk_level": get_risk_level(churn_probability),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
