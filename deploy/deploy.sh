#!/usr/bin/env bash
# Blue-Green 무중단 배포 스크립트 (self-hosted runner에서 실행)
#
#   [사용자] → localhost:8080 → nginx(snaplabel-proxy) → snaplabel-blue 또는 snaplabel-green
#
# 1) 새 이미지를 받아 "대기 중인 색" 컨테이너로 실행
# 2) 새 컨테이너에서 모델이 로드될 때까지 헬스체크 (실패 시 기존 서비스는 그대로 두고 중단)
# 3) nginx 설정을 새 컨테이너로 바꾸고 graceful reload (진행 중인 요청은 끊기지 않음)
# 4) 잠시 기다린 뒤 기존 컨테이너 종료
#
# 사용법: deploy/deploy.sh <이미지:태그>
set -euo pipefail

IMAGE="${1:?사용법: deploy/deploy.sh <이미지:태그>}"
PORT="${DEPLOY_PORT:-8080}"
NET=snaplabel-net
PROXY=snaplabel-proxy
APP=snaplabel
NGINX_IMAGE=nginx:1.27.5-alpine

log() { echo "[deploy $(date +%H:%M:%S)] $*"; }

log "이미지 받는 중: $IMAGE"
docker pull "$IMAGE" || docker image inspect "$IMAGE" >/dev/null  # 로컬 테스트용 이미지 허용

docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null

# 프록시(nginx)가 없으면 생성 - 외부 포트는 프록시만 열어둠
if ! docker ps --format '{{.Names}}' | grep -qx "$PROXY"; then
  docker rm -f "$PROXY" >/dev/null 2>&1 || true
  log "프록시 생성: localhost:$PORT"
  docker run -d --name "$PROXY" --network "$NET" -p "$PORT:80" \
    --restart unless-stopped "$NGINX_IMAGE" >/dev/null
fi

# 현재 트래픽을 받는 색 확인 (nginx 설정 기준)
ACTIVE=$(docker exec "$PROXY" cat /etc/nginx/conf.d/default.conf 2>/dev/null \
  | grep -o "$APP-\(blue\|green\)" | head -1 | sed "s/$APP-//" || true)
if [ "$ACTIVE" = "blue" ]; then NEW=green; else NEW=blue; fi
log "현재 활성: ${ACTIVE:-없음} → 새 버전: $NEW"

# 새 컨테이너 실행 (외부 포트 없이 내부 네트워크에만 연결)
docker rm -f "$APP-$NEW" >/dev/null 2>&1 || true
docker run -d --name "$APP-$NEW" --network "$NET" --restart unless-stopped "$IMAGE" >/dev/null

# 새 컨테이너 헬스체크: 모델이 로드되어야 통과
log "헬스체크 대기 중..."
ok=0
for _ in $(seq 1 60); do
  if docker exec "$APP-$NEW" python -c \
    "import urllib.request,sys; b=urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3).read(); sys.exit(0 if b'\"loaded\":true' in b else 1)" \
    >/dev/null 2>&1; then
    ok=1; break
  fi
  sleep 1
done
if [ "$ok" != 1 ]; then
  log "새 컨테이너가 정상 기동하지 않음 → 배포 중단 (기존 서비스 유지)"
  docker logs --tail 50 "$APP-$NEW" || true
  docker rm -f "$APP-$NEW" >/dev/null
  exit 1
fi
log "헬스체크 통과"

# nginx 전환 (설정 검증 후 graceful reload)
docker exec -i "$PROXY" sh -c 'cat > /etc/nginx/conf.d/default.conf' <<CONF
upstream snaplabel_app {
    server $APP-$NEW:8000;
}
server {
    listen 80;
    client_max_body_size 6m;
    location / {
        proxy_pass http://snaplabel_app;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }
}
CONF
docker exec "$PROXY" nginx -t -q
docker exec "$PROXY" nginx -s reload

# reload 반영 확인: 프록시를 통해 모델 로드된 응답이 올 때까지 대기
for _ in $(seq 1 20); do
  curl -fsS -m 3 "http://127.0.0.1:$PORT/api/health" 2>/dev/null | grep -q '"loaded":true' && break
  sleep 0.5
done
log "트래픽 전환 완료: $APP-$NEW"

# 기존 컨테이너는 진행 중인 요청이 끝나도록 잠시 기다린 뒤 종료
if [ -n "$ACTIVE" ]; then
  sleep 5
  docker stop -t 30 "$APP-$ACTIVE" >/dev/null 2>&1 || true
  docker rm "$APP-$ACTIVE" >/dev/null 2>&1 || true
  log "기존 컨테이너 종료: $APP-$ACTIVE"
fi

docker image prune -f >/dev/null 2>&1 || true
log "배포 완료 → http://localhost:$PORT"
