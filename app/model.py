"""MobileNetV2 (ImageNet) 이미지 분류기 - ONNX Runtime CPU 추론."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "mobilenetv2-12.onnx"
LABELS_PATH = Path(__file__).resolve().parent / "imagenet_classes.txt"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224


class InvalidImageError(ValueError):
    """업로드된 파일을 이미지로 읽을 수 없을 때."""


class Classifier:
    def __init__(self, model_path: str | os.PathLike | None = None) -> None:
        path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
        if not path.exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {path}. `python scripts/download_model.py`를 먼저 실행하세요."
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = LABELS_PATH.read_text(encoding="utf-8").splitlines()
        self.model_name = path.name

    @staticmethod
    def preprocess(data: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(data))
            image = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("이미지 파일을 읽을 수 없습니다.") from exc

        # 짧은 변을 256으로 맞춘 뒤 가운데 224x224를 잘라냄 (ImageNet 표준 전처리)
        width, height = image.size
        scale = 256 / min(width, height)
        image = image.resize(
            (max(INPUT_SIZE, round(width * scale)), max(INPUT_SIZE, round(height * scale))),
            Image.BILINEAR,
        )
        width, height = image.size
        left = (width - INPUT_SIZE) // 2
        top = (height - INPUT_SIZE) // 2
        image = image.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - MEAN) / STD
        return array.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(self, data: bytes, top_k: int = 5) -> list[dict]:
        tensor = self.preprocess(data)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        top = np.argsort(probs)[::-1][:top_k]
        return [
            {"rank": rank + 1, "label": self.labels[i], "class_id": int(i), "probability": float(probs[i])}
            for rank, i in enumerate(top)
        ]
