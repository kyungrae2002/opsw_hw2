"""ImageNet 이미지 분류기 - ONNX Runtime CPU 추론.

사용할 모델 설정은 환경변수 MODEL_NAME 으로 고릅니다 (기본값: DEFAULT_MODEL).
모델 비교 실험 결과는 reports/model_update.md 참고.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LABELS_PATH = Path(__file__).resolve().parent / "imagenet_classes.txt"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224
RESIZE_SHORT = 256  # 짧은 변을 256으로 맞춘 뒤 가운데 224x224 (crop 비율 0.875)


@dataclass(frozen=True)
class ModelSpec:
    file: str
    layout: str  # "NCHW" 또는 "NHWC"
    normalize: str  # "imagenet": (x/255-mean)/std, "pm1": (x-127)/128
    output: str  # "logits" (softmax 필요) 또는 "probs" (이미 확률)
    resample: int = Image.BILINEAR
    flip_tta: bool = False  # 좌우 반전 이미지와 결과 평균 (test-time augmentation)


# 기존(v1.0) 모델 + 개선 실험에서 시도한 설정
MODEL_SPECS: dict[str, ModelSpec] = {
    # 기준선: 기존 서비스 모델
    "mobilenetv2-12": ModelSpec("mobilenetv2-12.onnx", "NCHW", "imagenet", "logits"),
    # 시도 1: 같은 모델, 전처리 개선 (BICUBIC 리사이즈 + 좌우반전 TTA)
    "mobilenetv2-12-tta": ModelSpec(
        "mobilenetv2-12.onnx", "NCHW", "imagenet", "logits", Image.BICUBIC, flip_tta=True
    ),
    # 시도 2: 사전학습 모델 교체 (ONNX Model Zoo, MIT)
    "efficientnet-lite4-11": ModelSpec("efficientnet-lite4-11.onnx", "NHWC", "pm1", "probs"),
}
DEFAULT_MODEL = "efficientnet-lite4-11"  # scripts/download_model.py 의 기본값과 같아야 함


class InvalidImageError(ValueError):
    """업로드된 파일을 이미지로 읽을 수 없을 때."""


def load_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("이미지 파일을 읽을 수 없습니다.") from exc


class Classifier:
    def __init__(self, model_name: str | None = None, model_path: str | os.PathLike | None = None) -> None:
        name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)
        if name not in MODEL_SPECS:
            raise ValueError(f"알 수 없는 모델: {name} (가능: {', '.join(MODEL_SPECS)})")
        self.spec = MODEL_SPECS[name]
        path = Path(model_path or os.getenv("MODEL_PATH") or MODELS_DIR / self.spec.file)
        if not path.exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {path}. `python scripts/download_model.py {name}`를 먼저 실행하세요."
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = LABELS_PATH.read_text(encoding="utf-8").splitlines()
        self.model_name = name

    def _to_tensor(self, image: Image.Image) -> np.ndarray:
        spec = self.spec
        width, height = image.size
        scale = RESIZE_SHORT / min(width, height)
        image = image.resize(
            (max(INPUT_SIZE, round(width * scale)), max(INPUT_SIZE, round(height * scale))),
            spec.resample,
        )
        width, height = image.size
        left = (width - INPUT_SIZE) // 2
        top = (height - INPUT_SIZE) // 2
        image = image.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

        array = np.asarray(image, dtype=np.float32)  # HWC, 0~255
        if spec.normalize == "imagenet":
            array = (array / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        else:
            array = (array - 127.0) / 128.0
        if spec.layout == "NCHW":
            array = array.transpose(2, 0, 1)
        return array[np.newaxis, ...].astype(np.float32)

    def preprocess(self, data: bytes) -> list[np.ndarray]:
        image = load_image(data)
        tensors = [self._to_tensor(image)]
        if self.spec.flip_tta:
            tensors.append(self._to_tensor(ImageOps.mirror(image)))
        return tensors

    def probabilities(self, data: bytes) -> np.ndarray:
        probs = []
        for tensor in self.preprocess(data):
            out = self.session.run(None, {self.input_name: tensor})[0][0].astype(np.float64)
            if self.spec.output == "logits":
                exp = np.exp(out - out.max())
                out = exp / exp.sum()
            probs.append(out)
        return np.mean(probs, axis=0)

    def predict(self, data: bytes, top_k: int = 5) -> list[dict]:
        probs = self.probabilities(data)
        top = np.argsort(probs)[::-1][:top_k]
        return [
            {"rank": rank + 1, "label": self.labels[i], "class_id": int(i), "probability": float(probs[i])}
            for rank, i in enumerate(top)
        ]
