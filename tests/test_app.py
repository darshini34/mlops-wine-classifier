import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.datasets import load_wine

from app import app, FEATURES


def client():
    app.config["TESTING"] = True
    return app.test_client()


def sample_payload():
    wine = load_wine()
    return {name: float(wine.data[0][i]) for i, name in enumerate(FEATURES)}


def test_health():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_home_page():
    response = client().get("/")
    assert response.status_code == 200
    assert b"Wine Type Classification" in response.data


def test_api_prediction():
    response = client().post("/api/predict", json=sample_payload())
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "class_0"


def test_api_missing_field():
    payload = sample_payload()
    payload.pop("alcohol")
    response = client().post("/api/predict", json=payload)
    assert response.status_code == 400
