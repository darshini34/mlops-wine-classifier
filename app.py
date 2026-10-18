from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import load_wine

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

model = joblib.load(BASE_DIR / "model.pkl")

wine = load_wine()
FEATURES = list(wine.feature_names)
CLASS_NAMES = list(wine.target_names)
DEFAULTS = {name: round(float(wine.data[:, i].mean()), 2)
            for i, name in enumerate(FEATURES)}
MAX_VALUES = {name: float(wine.data[:, i].max()) * 2
              for i, name in enumerate(FEATURES)}


def validate(data):
    """Return (values, errors) after checking every wine feature."""
    values, errors = [], []
    for name in FEATURES:
        raw = data.get(name)
        if raw is None or str(raw).strip() == "":
            errors.append(f"{name} is required")
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{name} must be a number")
            continue
        if number < 0 or number > MAX_VALUES[name]:
            errors.append(f"{name} is out of the allowed range")
            continue
        values.append(number)
    return values, errors


def predict_class(values):
    class_id = int(model.predict([values])[0])
    return class_id, CLASS_NAMES[class_id]


@app.route("/")
def index():
    return render_template("index.html", features=FEATURES,
                           values=DEFAULTS, result=None, errors=[])


@app.route("/health")
def health():
    info = {"status": "ok", "model_loaded": model is not None}
    best = request.accept_mimetypes.best_match(["application/json", "text/html"])
    if best == "text/html":
        return render_template("health.html", info=info,
                               n_features=len(FEATURES), classes=CLASS_NAMES)
    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    values, errors = validate(request.form)
    result = None
    if not errors:
        _, result = predict_class(values)
    return render_template("index.html", features=FEATURES,
                           values=request.form, result=result, errors=errors)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    values, errors = validate(data)
    if errors:
        return jsonify({"errors": errors}), 400
    class_id, class_name = predict_class(values)
    return jsonify({"class_id": class_id, "prediction": class_name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)