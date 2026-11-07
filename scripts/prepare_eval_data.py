"""모델 비교용 공개 평가 데이터 100건을 내려받습니다.

- 출처: Hugging Face `vaishaal/ImageNetV2` (MIT License)
  ImageNet 분류 모델 평가용으로 새로 수집된 테스트셋. matched-frequency 버전(행 0~9999)만 사용.
- 고정 시드로 100건을 뽑아 30건(tune: 개선 방법 선택용) / 70건(test: 최종 평가 전용)으로 분리.
- 이미지는 data/eval/ 에 저장되며 Git에는 올리지 않습니다(.gitignore).

사용법:  python scripts/prepare_eval_data.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "eval"
IMG_DIR = OUT_DIR / "images"
MANIFEST = OUT_DIR / "manifest.csv"
LABELS = (ROOT / "app" / "imagenet_classes.txt").read_text(encoding="utf-8").splitlines()

DATASET = "vaishaal/ImageNetV2"
ROWS_API = "https://datasets-server.huggingface.co/rows"
MATCHED_FREQUENCY_ROWS = 10_000
SEED = 2026
N_TOTAL, N_TUNE = 100, 30


ROW_CACHE = OUT_DIR / "rows_cache.json"


def fetch_row(offset: int) -> dict:
    """행 메타데이터 조회 (요청 한도 429를 피하려고 간격을 두고, 결과는 캐시)."""
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "train", "offset": offset, "length": 1}
    )
    for attempt in range(8):
        try:
            time.sleep(1.0)
            with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=30) as res:
                return json.load(res)["rows"][0]["row"]
        except Exception as exc:  # 429 등 일시적 오류는 점점 길게 기다렸다가 재시도
            if attempt == 7:
                raise
            wait = 15 * (attempt + 1)
            print(f"  재시도 {offset} ({exc}) - {wait}초 대기")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download_image(url: str, path: Path) -> None:
    tmp = path.with_suffix(".part")
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
            return
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"  이미지 재시도 {path.name} ({exc})")
            time.sleep(5 * (attempt + 1))


def main() -> int:
    rng = random.Random(SEED)
    offsets = rng.sample(range(MATCHED_FREQUENCY_ROWS), N_TOTAL)
    splits = ["tune"] * N_TUNE + ["test"] * (N_TOTAL - N_TUNE)  # 이미 무작위 순서

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(ROW_CACHE.read_text()) if ROW_CACHE.exists() else {}
    rows = []
    for i, (offset, split) in enumerate(zip(offsets, splits), 1):
        path = IMG_DIR / f"{offset:05d}.jpg"
        if str(offset) in cache and path.exists():
            key = cache[str(offset)]
        else:
            row = fetch_row(offset)
            key = row["__key__"]
            if not path.exists():
                download_image(row["jpeg"]["src"], path)
            cache[str(offset)] = key
            ROW_CACHE.write_text(json.dumps(cache))
        # key 형식: imagenetv2-matched-frequency-format-val/<class_id>/<hash>
        assert key.startswith("imagenetv2-matched-frequency"), key
        class_id = int(key.split("/")[1])
        rows.append(
            {"file": path.name, "offset": offset, "split": split,
             "class_id": class_id, "label": LABELS[class_id], "source_key": key}
        )
        print(f"[{i:3d}/{N_TOTAL}] {split:4s} {path.name} -> {class_id} {LABELS[class_id]}")

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"완료: {MANIFEST} (tune {N_TUNE} / test {N_TOTAL - N_TUNE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
