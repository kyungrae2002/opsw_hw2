"""모델 설정별 분류 성능 평가 (top-1 / top-5 정확도 + 오류 분석).

평가 데이터는 scripts/prepare_eval_data.py 로 준비합니다.
  - tune (30건): 개선 방법을 고르는 데만 사용
  - test (70건): 최종 비교 전용 - 개선 방법 선택에 사용하지 않음

사용법:
  python scripts/evaluate.py --split tune mobilenetv2-12 mobilenetv2-12-tta efficientnet-lite4-11
  python scripts/evaluate.py --split test mobilenetv2-12 efficientnet-lite4-11
결과: reports/eval_<split>.json (요약 지표 + 이미지별 예측)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model import Classifier  # noqa: E402

DATA_DIR = ROOT / "data" / "eval"
REPORTS = ROOT / "reports"


def evaluate(name: str, rows: list[dict]) -> dict:
    clf = Classifier(model_name=name)
    items, elapsed = [], 0.0
    for row in rows:
        data = (DATA_DIR / "images" / row["file"]).read_bytes()
        started = time.perf_counter()
        preds = clf.predict(data, top_k=5)
        elapsed += time.perf_counter() - started
        truth = int(row["class_id"])
        ids = [p["class_id"] for p in preds]
        items.append({
            "file": row["file"], "truth": row["label"], "truth_id": truth,
            "pred": preds[0]["label"], "pred_prob": round(preds[0]["probability"], 4),
            "top1": ids[0] == truth, "top5": truth in ids,
        })
    n = len(items)
    return {
        "model": name,
        "n": n,
        "top1": sum(i["top1"] for i in items) / n,
        "top5": sum(i["top5"] for i in items) / n,
        "avg_ms": round(elapsed / n * 1000, 1),
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["tune", "test"], required=True)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()

    with (DATA_DIR / "manifest.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]

    results = [evaluate(m, rows) for m in args.models]
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"eval_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "results": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    print(f"\n[{args.split}] {len(rows)}건")
    print(f"{'model':24s} {'top-1':>7s} {'top-5':>7s} {'ms/img':>7s}")
    for r in results:
        print(f"{r['model']:24s} {r['top1']:7.1%} {r['top5']:7.1%} {r['avg_ms']:7.1f}")

    base = results[0]
    wrong = [i for i in base["items"] if not i["top1"]]
    print(f"\n{base['model']} 오답 {len(wrong)}건:")
    for i in wrong:
        fixed = [r["model"] for r in results[1:] if next(x for x in r["items"] if x["file"] == i["file"])["top1"]]
        print(f"  {i['file']}: 정답 '{i['truth']}' → 예측 '{i['pred']}' ({i['pred_prob']:.2f})"
              f"{'  ✔ 고쳐짐: ' + ', '.join(fixed) if fixed else ''}")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
