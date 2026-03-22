# Customer Churn Prediction API

A Machine Learning-powered REST API that predicts whether a customer is likely to churn based on usage and subscription features.

---

## Overview

Customer churn is a critical problem for subscription-based businesses.
This project uses a **Logistic Regression model** to predict churn probability and classify customers into risk categories.

---

## Features

* Machine Learning model (Logistic Regression)
* Feature scaling using StandardScaler
* REST API built with Flask
* Input validation and error handling
* Probability-based risk classification

---

## Input Parameters

| Feature         | Description                                         |
| --------------- | --------------------------------------------------- |
| tenure          | Number of months customer stayed                    |
| monthly_charges | Monthly bill amount                                 |
| total_charges   | Total amount spent                                  |
| contract        | Contract type (0 = monthly, 1 = yearly, 2 = 2-year) |

---

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

---

## API Endpoint

### POST `/predict`

### Example Request

```json
{
  "tenure": 10,
  "monthly_charges": 90,
  "total_charges": 900,
  "contract": 0
}
```

---

## Example Response

```json
{
  "prediction": 1,
  "label": "churn",
  "churn_probability": 0.9867,
  "risk_level": "high"
}
```

---

## Tech Stack

* Python
* Flask
* Scikit-learn
* NumPy

---

## Future Improvements

* Deploy API (Render / AWS)
* Add frontend dashboard
* Train on real-world dataset
* Save & load trained model

---

## Author

Harshit Bora
