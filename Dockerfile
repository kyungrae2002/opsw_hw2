FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) 의존성 설치 (버전 고정된 requirements.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) 모델은 Git에 올리지 않고, 이미지 빌드 시 내려받아 SHA256으로 검증
COPY scripts/ scripts/
RUN python scripts/download_model.py

# 3) 앱 코드 복사 후, 모델이 실제로 로드·추론되는지 빌드 단계에서 검증
COPY app/ app/
RUN python scripts/verify_model.py

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# 배포된 커밋을 /api/health 에 표시
ARG GIT_COMMIT=local
ENV GIT_COMMIT=${GIT_COMMIT} \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
