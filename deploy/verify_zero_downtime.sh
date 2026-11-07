#!/usr/bin/env bash
# 배포 전·중·후에 실제 예측 API(/api/predict)로 계속 요청을 보내
# 모든 요청이 성공하는지, 새 버전(커밋)으로 전환되는지 확인합니다.
# 결과는 GitHub Actions Summary에 표로 남습니다.
#
# 사용법: deploy/verify_zero_downtime.sh <이미지:태그> <기대 커밋(7자리)>
set -uo pipefail

IMAGE="${1:?이미지 필요}"
EXPECTED="${2:?기대 커밋 필요}"
PORT="${DEPLOY_PORT:-8080}"
URL="http://127.0.0.1:$PORT"
DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE="$DIR/../screenshots/1_preview.png"
WORK="$(mktemp -d)"
LOG="$WORK/requests.log"; PHASE="$WORK/phase"; STOP="$WORK/stop"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/stdout}"

# 요청 발생기: 0.2초 간격으로 예측 요청, "단계 HTTP코드 커밋" 기록
traffic() {
  while [ ! -f "$STOP" ]; do
    resp=$(curl -s -m 15 -w '\n%{http_code}' -F "file=@$SAMPLE;type=image/png" "$URL/api/predict" || true)
    code=$(printf '%s' "$resp" | tail -n1)
    commit=$(printf '%s' "$resp" | grep -o '"commit":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo "$(cat "$PHASE") ${code:-000} ${commit:-none}" >> "$LOG"
    sleep 0.2
  done
}

FIRST_DEPLOY=0
curl -fsS -m 5 "$URL/api/health" >/dev/null 2>&1 || FIRST_DEPLOY=1

if [ "$FIRST_DEPLOY" = 1 ]; then
  echo "실행 중인 서비스가 없음 → 첫 배포이므로 무중단 검증은 생략"
  "$DIR/deploy.sh" "$IMAGE" || exit 1
else
  echo before > "$PHASE"
  traffic & TPID=$!
  sleep 5
  echo during > "$PHASE"
  if ! "$DIR/deploy.sh" "$IMAGE"; then
    touch "$STOP"; wait "$TPID"; exit 1
  fi
  echo after > "$PHASE"
  sleep 8
  touch "$STOP"; wait "$TPID"
fi

# 배포 후 최종 버전 확인
FINAL=$(curl -fsS -m 5 "$URL/api/health" | grep -o '"commit":"[^"]*"' | cut -d'"' -f4)

{
  echo "## 🚀 배포 결과"
  echo ""
  echo "- 이미지: \`$IMAGE\`"
  echo "- 서비스 주소: http://localhost:$PORT (API 문서: http://localhost:$PORT/docs)"
  echo "- 배포 후 버전(커밋): \`$FINAL\` (기대값 \`$EXPECTED\`)"
} >> "$SUMMARY"

FAIL=0
[ "$FINAL" = "$EXPECTED" ] || FAIL=1

if [ "$FIRST_DEPLOY" = 0 ]; then
  total=$(wc -l < "$LOG" | tr -d ' ')
  ok=$(awk '$2==200' "$LOG" | wc -l | tr -d ' ')
  bad=$((total - ok))
  {
    echo ""
    echo "## ✅ 무중단 배포 검증 (실제 /api/predict 요청)"
    echo ""
    echo "| 단계 | 요청 수 | 성공(200) | 실패 | 응답한 버전 |"
    echo "|---|---|---|---|---|"
    for p in before during after; do
      label=$( [ $p = before ] && echo "배포 전" || { [ $p = during ] && echo "배포 중" || echo "배포 후"; } )
      n=$(awk -v p=$p '$1==p' "$LOG" | wc -l | tr -d ' ')
      s=$(awk -v p=$p '$1==p && $2==200' "$LOG" | wc -l | tr -d ' ')
      v=$(awk -v p=$p '$1==p {print $3}' "$LOG" | sort -u | tr '\n' ' ')
      echo "| $label | $n | $s | $((n - s)) | $v |"
    done
    echo "| **합계** | **$total** | **$ok** | **$bad** | |"
    echo ""
    if [ "$bad" = 0 ] && [ "$FAIL" = 0 ]; then
      echo "**결과: 배포 전·중·후 총 ${total}건의 요청이 모두 성공했고, 새 버전(\`$EXPECTED\`)으로 전환되었습니다. → 서비스 중단 없이 재배포 완료**"
    else
      echo "**결과: 실패 ${bad}건 / 최종 버전 \`$FINAL\` → 검증 실패**"
    fi
  } >> "$SUMMARY"
  [ "$bad" = 0 ] || FAIL=1
fi

rm -rf "$WORK"
exit $FAIL
