"""사전 학습된 ONNX 모델(ONNX Model Zoo, MIT)을 내려받고 SHA256으로 무결성을 확인합니다.

사용법:
  python scripts/download_model.py                 # 서비스 기본 모델
  python scripts/download_model.py mobilenetv2-12  # 특정 모델
  python scripts/download_model.py --all           # 비교 실험용 전체
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ZOO = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification"
MODELS = {
    "mobilenetv2-12": (
        f"{ZOO}/mobilenet/model/mobilenetv2-12.onnx",
        "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
    ),
    "efficientnet-lite4-11": (
        f"{ZOO}/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "d111689907c06eea7c82e4833ddef758da6453b9d4cf60b7e99ca05c7cbd9c12",
    ),
}
DEFAULT_MODEL = "efficientnet-lite4-11"  # app/model.py 의 DEFAULT_MODEL 과 같아야 함
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str) -> bool:
    url, expected = MODELS[name]
    target = MODELS_DIR / url.rsplit("/", 1)[1]
    if target.exists() and sha256(target) == expected:
        print(f"이미 준비됨: {target}")
        return True
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    print(f"다운로드 중: {url}")
    urllib.request.urlretrieve(url, tmp)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        print(f"SHA256 불일치 ({name}): {actual}", file=sys.stderr)
        return False
    tmp.replace(target)
    print(f"완료: {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def main(argv: list[str]) -> int:
    if "--all" in argv:
        names = list(MODELS)
    else:
        names = [a for a in argv if not a.startswith("-")] or [DEFAULT_MODEL]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"알 수 없는 모델: {unknown} (가능: {list(MODELS)})", file=sys.stderr)
        return 1
    return 0 if all(download(n) for n in names) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
