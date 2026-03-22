# Customer Churn Prediction API

This project predicts whether a customer will churn using a Machine Learning model.

## Features

* Logistic Regression model
* Feature scaling
* REST API using Flask
* Risk classification

## Input

* tenure
* monthly_charges
* total_charges
* contract

## Run

pip install -r requirements.txt
python app.py

## Example Request

POST /predict

{
"tenure": 10,
"monthly_charges": 90,
"total_charges": 900,
"contract": 0
}
