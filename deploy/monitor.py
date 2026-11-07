"""무중단 배포 검증용 독립 모니터 (표준 라이브러리만 사용).

배포와 별개의 프로세스로 실행되어, 고정 테스트 이미지를 실제 분석 API(/api/predict)에
1초 간격으로 보내고 요청마다 시작·종료 시각, 응답 시간, HTTP 상태, 분석 정상 여부,
응답한 서버 버전을 기록합니다. 배포가 끝난 뒤에도 30초 더 요청한 다음 종료합니다.

하위 명령:
  run         모니터 실행 (백그라운드로 띄움)
  wait-ready  모니터가 정상 응답을 확인할 때까지 대기
  mark        배포 시작/종료 시각 기록
  report      모니터 종료를 기다린 뒤 로그·요약(GitHub Summary) 생성
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
UNVERIFIED_GAP_SEC = 2.0  # 응답 기록 사이 간격이 이보다 길면 "검증되지 않은 구간"으로 표시


def ts(epoch: float, ms: bool = True) -> str:
    fmt = "%H:%M:%S.%f" if ms else "%H:%M:%S"
    s = datetime.fromtimestamp(epoch, KST).strftime(fmt)
    return s[:-3] if ms else s


def full_ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, KST).strftime("%Y-%m-%d %H:%M:%S KST")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- run
def post_image(url: str, image: bytes, filename: str, timeout: float) -> tuple[int, bytes, str]:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + image + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read(), ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), f"HTTP {exc.code}"
    except Exception as exc:  # 연결 거부, 타임아웃 등
        return 0, b"", f"{type(exc).__name__}: {exc}"


def one_request(seq: int, args, image: bytes, lock: threading.Lock, log_f, jsonl_f) -> None:
    start = time.time()
    status, body, error = post_image(f"{args.url}/api/predict", image, Path(args.image).name, args.timeout)
    end = time.time()
    rec = {"seq": seq, "start": start, "end": end, "latency_ms": round((end - start) * 1000, 1),
           "status": status, "ok": False, "version": "", "commit": "", "model": "", "top1": "",
           "error": error}
    if status == 200:
        try:
            data = json.loads(body)
            preds = data.get("predictions") or []
            rec.update(version=data.get("version", ""), commit=data.get("commit", ""),
                       model=data.get("model", ""), top1=preds[0]["label"] if preds else "")
            rec["ok"] = len(preds) == 5 and all(0 <= p["probability"] <= 1 for p in preds)
            if not rec["ok"]:
                rec["error"] = "분석 결과 형식 오류"
        except Exception as exc:
            rec["error"] = f"응답 파싱 실패: {exc}"
    line = (f"#{seq:04d} start={ts(start)} end={ts(end)} {rec['latency_ms']:7.1f}ms "
            f"HTTP {status} {'OK  ' if rec['ok'] else 'FAIL'} "
            f"ver={rec['version']}/{rec['commit']}/{rec['model']} top1={rec['top1']} {rec['error']}")
    with lock:
        log_f.write(line + "\n"); log_f.flush()
        jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); jsonl_f.flush()
    print(line, flush=True)


def cmd_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image = Path(args.image).read_bytes()
    lock = threading.Lock()
    log_f = (out / "requests.log").open("a", encoding="utf-8")
    jsonl_f = (out / "requests.jsonl").open("a", encoding="utf-8")
    (out / "monitor.json").write_text(json.dumps(
        {"pid": os.getpid(), "url": args.url, "image": args.image, "interval": args.interval,
         "post_seconds": args.post_seconds, "started": time.time()}), encoding="utf-8")

    threads: list[threading.Thread] = []
    t0 = time.time()
    seq = 0
    ready_written = False
    while True:
        due = t0 + seq * args.interval
        time.sleep(max(0.0, due - time.time()))
        seq += 1
        t = threading.Thread(target=one_request, args=(seq, args, image, lock, log_f, jsonl_f), daemon=True)
        t.start()
        threads.append(t)

        # 첫 정상 응답 → ready 표시 (배포는 이걸 확인한 뒤 시작)
        if not ready_written and seq >= 2:
            with lock:
                ok_seen = any(json.loads(l)["ok"] for l in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines())
            if ok_seen:
                (out / "ready").write_text(str(time.time()), encoding="utf-8")
                ready_written = True

        end_mark = read_json(out / "deploy_end.json")
        if end_mark and time.time() >= end_mark["at"] + args.post_seconds:
            break
        if time.time() - t0 > args.max_seconds:
            print("최대 실행 시간 초과 → 모니터 종료", flush=True)
            break

    for t in threads:
        t.join(timeout=args.timeout + 1)
    (out / "monitor_done").write_text(str(time.time()), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- helpers
def cmd_wait_ready(args) -> int:
    out = Path(args.out)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if (out / "ready").exists():
            print("모니터: 기존 서비스 정상 응답 확인 → 배포 시작 가능")
            return 0
        time.sleep(0.5)
    print("::warning::모니터가 정상 응답을 확인하지 못했습니다 (실행 중인 서비스 없음 = 첫 배포?). 배포는 계속 진행합니다.")
    (out / "no_baseline").write_text("1", encoding="utf-8")
    return 0


def cmd_mark(args) -> int:
    out = Path(args.out)
    data = {"at": time.time()}
    if args.result is not None:
        data["result"] = args.result
    (out / f"{args.event}.json").write_text(json.dumps(data), encoding="utf-8")
    print(f"{args.event}: {full_ts(data['at'])}" + (f" (exit={args.result})" if args.result is not None else ""))
    return 0


# ---------------------------------------------------------------- report
def cmd_report(args) -> int:
    out = Path(args.out)
    meta = read_json(out / "monitor.json") or {}
    # 배포 단계가 비정상 종료되어 종료 시각이 없으면 지금을 종료로 기록 (모니터가 멈추도록)
    if not (out / "deploy_end.json").exists():
        (out / "deploy_end.json").write_text(json.dumps({"at": time.time(), "result": "unknown"}), encoding="utf-8")
    end_mark = read_json(out / "deploy_end.json")
    deadline = time.time() + (meta.get("post_seconds", 30) + 60)
    while not (out / "monitor_done").exists() and time.time() < deadline:
        time.sleep(1)

    recs = []
    if (out / "requests.jsonl").exists():
        recs = [json.loads(l) for l in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    recs.sort(key=lambda r: r["start"])
    start_mark = read_json(out / "deploy_start.json")
    d_start = start_mark["at"] if start_mark else None
    d_end = end_mark["at"]
    deploy_result = end_mark.get("result")
    deploy_ok = str(deploy_result) == "0"

    def phase(r):
        if d_start is None or r["start"] < d_start:
            return "배포 전"
        return "배포 중" if r["start"] <= d_end else "배포 후"

    for r in recs:
        r["phase"] = phase(r)

    # 전체 로그 (UTF-8) - Artifacts 업로드용
    with (out / "requests.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seq", "단계", "시작", "종료", "응답시간_ms", "HTTP", "분석정상", "버전", "커밋", "모델", "top1", "오류"])
        for r in recs:
            w.writerow([r["seq"], r["phase"], ts(r["start"]), ts(r["end"]), r["latency_ms"], r["status"],
                        "Y" if r["ok"] else "N", r["version"], r["commit"], r["model"], r["top1"], r["error"]])

    # Actions 로그에 전체 기록 출력
    print("::group::전체 요청 기록")
    for r in recs:
        print(f"#{r['seq']:04d} [{r['phase']}] {ts(r['start'])} → {ts(r['end'])} {r['latency_ms']:7.1f}ms "
              f"HTTP {r['status']} {'OK' if r['ok'] else 'FAIL'} {r['version']}/{r['commit']}/{r['model']} {r['error']}")
    print("::endgroup::")

    total = len(recs)
    ok = sum(r["ok"] for r in recs)
    fail = total - ok
    by_phase = {p: [r for r in recs if r["phase"] == p] for p in ("배포 전", "배포 중", "배포 후")}

    starts = [r["start"] for r in recs]
    intervals = [b - a for a, b in zip(starts, starts[1:])]
    ok_ends = sorted(r["end"] for r in recs if r["ok"])
    gaps = [(a, b, b - a) for a, b in zip(ok_ends, ok_ends[1:])]
    max_gap = max(gaps, key=lambda g: g[2]) if gaps else None
    unverified = [g for g in gaps if g[2] > UNVERIFIED_GAP_SEC]

    def ver(r):
        return f"{r['version']} / `{r['commit']}` / {r['model']}" if r["version"] else "-"

    versions = []
    for r in recs:
        if r["ok"] and (not versions or versions[-1][0] != (r["version"], r["commit"], r["model"])):
            versions.append(((r["version"], r["commit"], r["model"]), r))
    final_commit = versions[-1][0][1] if versions else ""
    switched = final_commit == args.expected_commit
    no_baseline = (out / "no_baseline").exists()

    lines = [
        "## 🛡️ 무중단 배포 검증 결과 — SnapLabel",
        "",
        "| 항목 | 내용 |",
        "|---|---|",
        f"| 검증 주소 | `{meta.get('url', '')}/api/predict` (고정 이미지 `{Path(meta.get('image', '')).name}`, {meta.get('interval', 1)}초 간격, 배포 후 {meta.get('post_seconds', 30)}초 추가) |",
    ]
    if recs:
        lines.append(f"| 검증 시간 | {full_ts(recs[0]['start'])} ~ {ts(recs[-1]['end'], ms=False)} (총 {recs[-1]['end'] - recs[0]['start']:.0f}초) |")
    if d_start:
        lines.append(f"| 배포 구간 | {ts(d_start, ms=False)} ~ {ts(d_end, ms=False)} ({d_end - d_start:.1f}초) · 배포 결과: {'✅ 성공' if deploy_ok else f'❌ 실패 (exit={deploy_result})'} |")
    if versions:
        chain = " → ".join(f"{v[0][0]} `{v[0][1]}` {v[0][2]}" for v in versions)
        first_new = next((r for (k, r) in versions if k[1] == args.expected_commit), None)
        lines.append(f"| 버전 전환 | {chain}" + (f" (새 버전 첫 응답 {ts(first_new['end'])})" if first_new else "") +
                     f" · 기대 커밋 `{args.expected_commit}` {'✅' if switched else '❌'} |")
    lines.append(f"| 요청 결과 | **총 {total}건 · 성공 {ok} · 실패 {fail}** (전 {len(by_phase['배포 전'])} / 중 {len(by_phase['배포 중'])} / 후 {len(by_phase['배포 후'])}) |")
    if intervals:
        lines.append(f"| 실제 요청 간격 | 평균 {sum(intervals) / len(intervals):.2f}초 · 최소 {min(intervals):.2f}초 · 최대 {max(intervals):.2f}초 |")
    if max_gap:
        lines.append(f"| 최대 기록 공백 | {max_gap[2]:.2f}초 ({ts(max_gap[0])} ~ {ts(max_gap[1])}, 정상 응답 완료 사이 간격) |")
    if unverified:
        spans = ", ".join(f"{ts(a)}~{ts(b)} ({g:.1f}초)" for a, b, g in unverified)
        lines.append(f"| ⚠️ 검증되지 않은 구간 | {spans} — {UNVERIFIED_GAP_SEC:.0f}초 넘게 정상 응답 기록이 없던 구간 |")
    else:
        lines.append(f"| 검증되지 않은 구간 | 없음 (모든 정상 응답 사이 간격 ≤ {UNVERIFIED_GAP_SEC:.0f}초) |")

    lines += ["", "**단계별 대표 응답**", "", "| 단계 | 시각 | HTTP | 응답시간 | 분석 | 응답 버전 | top-1 |", "|---|---|---|---|---|---|---|"]
    for p, rs in by_phase.items():
        if not rs:
            lines.append(f"| {p} | - | - | - | - | - | - |")
            continue
        picks = [rs[0], rs[-1]] if p != "배포 중" else [rs[0], rs[len(rs) // 2], rs[-1]]
        if p == "배포 중":
            picks += [r for r in rs if not r["ok"]][:3]
        seen = set()
        for r in picks:
            if r["seq"] in seen:
                continue
            seen.add(r["seq"])
            lines.append(f"| {p} | {ts(r['start'])} | {r['status']} | {r['latency_ms']:.0f}ms | {'✅' if r['ok'] else '❌ ' + r['error'][:40]} | {ver(r)} | {r['top1']} |")

    lines.append("")
    passed = deploy_ok and fail == 0 and switched and total > 0 and not no_baseline
    if no_baseline:
        lines.append("> ℹ️ 배포 전에 실행 중인 서비스가 없어(첫 배포) 무중단 여부는 판단하지 않았습니다.")
    elif passed:
        lines.append(
            f"> ✅ **관측 범위({ts(recs[0]['start'], ms=False)} ~ {ts(recs[-1]['end'], ms=False)})에서 {total}건의 실제 분석 요청이 모두 정상 응답했고, "
            f"응답 버전이 `{versions[0][0][1]}` → `{final_commit}` 으로 전환되었습니다. 즉 이 범위에서는 서비스 중단 없이 재배포되었습니다.** "
            f"단, 요청은 {meta.get('interval', 1)}초 간격 표본이므로 요청 사이(최대 기록 공백 {max_gap[2] if max_gap else 0:.2f}초)보다 짧은 순간의 중단까지 증명하지는 않습니다."
        )
    else:
        reasons = []
        if not deploy_ok:
            reasons.append(f"배포 실패(exit={deploy_result})")
        if fail:
            reasons.append(f"실패 요청 {fail}건")
        if not switched:
            reasons.append(f"새 버전 `{args.expected_commit}` 미확인(최종 `{final_commit}`)")
        lines.append(f"> ❌ **검증 실패: {', '.join(reasons)}** — 전체 기록은 Artifacts의 `requests.csv` / `requests.log` 참고")
    lines.append("")
    lines.append("전체 요청 기록(UTF-8): Artifacts → `zero-downtime-monitor` (requests.csv, requests.log, requests.jsonl, summary.md)")

    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary, encoding="utf-8")
    target = os.getenv("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(summary)
    print(summary)
    if no_baseline:
        return 0 if deploy_ok and switched else 1
    return 0 if passed else 1


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--interval", type=float, default=1.0)
    r.add_argument("--post-seconds", type=float, default=30.0)
    r.add_argument("--timeout", type=float, default=10.0)
    r.add_argument("--max-seconds", type=float, default=900.0)
    w = sub.add_parser("wait-ready")
    w.add_argument("--out", required=True)
    w.add_argument("--timeout", type=float, default=60.0)
    m = sub.add_parser("mark")
    m.add_argument("--out", required=True)
    m.add_argument("event", choices=["deploy_start", "deploy_end"])
    m.add_argument("--result")
    rp = sub.add_parser("report")
    rp.add_argument("--out", required=True)
    rp.add_argument("--expected-commit", required=True)
    args = p.parse_args()
    return {"run": cmd_run, "wait-ready": cmd_wait_ready, "mark": cmd_mark, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
