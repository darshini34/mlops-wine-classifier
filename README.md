# Wine Type Classification - MLOps Practical

Flask web app and REST API that predicts the wine class (class_0, class_1, class_2)
from 13 chemical properties using a Decision Tree pipeline.

## Run locally
    python train_model.py
    python app.py
Open http://127.0.0.1:5000

## Routes
- GET  /            web form
- GET  /health      health check (JSON)
- POST /predict     form prediction
- POST /api/predict JSON prediction (see sample.json)

## Docker
    docker build -t wine-classifier:1.0 .
    docker run -d --name wine-classifier-app -p 5000:5000 wine-classifier:1.0
