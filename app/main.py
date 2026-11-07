"""SnapLabel - 사진 속 사물 인식 웹 서비스 (FastAPI)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model import Classifier, InvalidImageError

APP_VERSION = "1.1.0"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("snaplabel")

app = FastAPI(title="SnapLabel", version=APP_VERSION)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_classifier: Classifier | None = None
_load_error: str | None = None


def get_classifier() -> Classifier:
    """모델을 처음 요청 시 한 번만 로드 (lazy loading)."""
    global _classifier, _load_error
    if _classifier is None:
        try:
            _classifier = Classifier()
            _load_error = None
            logger.info("model loaded: %s", _classifier.model_name)
        except FileNotFoundError as exc:
            _load_error = str(exc)
            raise HTTPException(status_code=503, detail=_load_error) from exc
    return _classifier


def deploy_info() -> dict:
    commit = os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or "local"
    return {"version": APP_VERSION, "commit": commit[:7]}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    try:
        clf = get_classifier()
        model = {"loaded": True, "name": clf.model_name}
    except HTTPException:
        model = {"loaded": False, "error": _load_error}
    return {"status": "ok", "model": model, **deploy_info()}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="지원하지 않는 파일 형식입니다. JPG, PNG, WEBP, BMP, GIF 이미지만 올려주세요.",
        )
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다. 5MB 이하 이미지를 올려주세요.")

    clf = get_classifier()
    started = time.perf_counter()
    try:
        predictions = clf.predict(data, top_k=5)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info("predict %s -> %s (%.1fms)", file.filename, predictions[0]["label"], elapsed_ms)

    return {
        "filename": file.filename,
        "model": clf.model_name,
        "inference_ms": elapsed_ms,
        "predictions": predictions,
        "notice": "결과는 ImageNet 1000개 클래스 기준 모델의 추정치이며 정확하지 않을 수 있습니다.",
        **deploy_info(),  # 응답한 서버의 버전·커밋 (무중단 배포 검증용)
    }
