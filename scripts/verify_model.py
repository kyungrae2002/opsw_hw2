"""이미지 빌드 시 모델이 정상적으로 로드·추론되는지 검증합니다.

사용법:  python scripts/verify_model.py   (실패 시 종료코드 1 → docker build 중단)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

from app.model import Classifier  # noqa: E402


def main() -> int:
    clf = Classifier()
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (120, 160, 200)).save(buf, format="PNG")
    result = clf.predict(buf.getvalue(), top_k=5)

    probs = [r["probability"] for r in result]
    assert len(clf.labels) == 1000, f"라벨 수 오류: {len(clf.labels)}"
    assert len(result) == 5, "top-5 결과가 아님"
    assert all(np.isfinite(p) and 0 <= p <= 1 for p in probs), "확률 값 오류"
    assert probs == sorted(probs, reverse=True), "정렬 오류"

    print(f"모델 검증 OK: {clf.model_name} (onnxruntime {ort.__version__}, numpy {np.__version__})")
    print(f"  top-1: {result[0]['label']} ({result[0]['probability']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
