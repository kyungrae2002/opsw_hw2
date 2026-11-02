"""사전 학습된 MobileNetV2 ONNX 모델(ONNX Model Zoo)을 내려받고 무결성을 확인합니다.

사용법:  python scripts/download_model.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/"
    "validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
)
MODEL_SHA256 = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
TARGET = Path(__file__).resolve().parent.parent / "models" / "mobilenetv2-12.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if TARGET.exists() and sha256(TARGET) == MODEL_SHA256:
        print(f"이미 준비됨: {TARGET}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_suffix(".part")
    print(f"다운로드 중: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    actual = sha256(tmp)
    if actual != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        print(f"SHA256 불일치: {actual}", file=sys.stderr)
        return 1
    tmp.replace(TARGET)
    print(f"완료: {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
