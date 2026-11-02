import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def make_image(fmt: str = "PNG", size=(320, 240), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def test_index_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "SnapLabel" in res.text


def test_health_reports_model_loaded():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model"]["loaded"] is True


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_predict_returns_top5_sorted(fmt, mime):
    res = client.post("/api/predict", files={"file": (f"x.{fmt.lower()}", make_image(fmt), mime)})
    assert res.status_code == 200
    preds = res.json()["predictions"]
    assert len(preds) == 5
    probs = [p["probability"] for p in preds]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert sum(probs) <= 1.0 + 1e-6


def test_predict_small_image_is_upscaled():
    res = client.post("/api/predict", files={"file": ("tiny.png", make_image(size=(10, 30)), "image/png")})
    assert res.status_code == 200


def test_rejects_non_image_type():
    res = client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 415


def test_rejects_corrupt_image():
    res = client.post("/api/predict", files={"file": ("bad.png", b"not really a png", "image/png")})
    assert res.status_code == 400


def test_rejects_empty_file():
    res = client.post("/api/predict", files={"file": ("empty.png", b"", "image/png")})
    assert res.status_code == 400


def test_rejects_too_large_file():
    big = b"\x00" * (5 * 1024 * 1024 + 10)
    res = client.post("/api/predict", files={"file": ("big.png", big, "image/png")})
    assert res.status_code == 413
