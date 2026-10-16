import requests

API_URL = "http://127.0.0.1:8000"


def test_health():
    response = requests.get(f"{API_URL}/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction():
    payload = {
        "features": [0.0] * 20
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability_class_0"] <= 1
    assert 0 <= result["probability_class_1"] <= 1

    total = (
        result["probability_class_0"]
        + result["probability_class_1"]
    )

    assert abs(total - 1.0) < 0.001
