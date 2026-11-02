# SnapLabel — 사진 속 사물 인식 웹 서비스

사진을 올리면 사전 학습된 **MobileNetV2 (ImageNet 1000 클래스)** 모델이 사진 속 사물을 추정해 상위 5개 결과와 확률을 보여주는 웹앱입니다.
GPU 없이 일반 CPU에서 **ONNX Runtime**으로 동작하며, `main` 브랜치에 push하면 **테스트 → Docker 빌드 → 자동 재배포**까지 이어지는 CI/CD 파이프라인을 갖추고 있습니다.

> 결과는 모델의 추정치이며 정확하지 않을 수 있습니다. ImageNet에 없는 사물(예: 사람 얼굴, 한국 음식 일부)은 엉뚱하게 분류될 수 있습니다.

## 기술 스택

| 구분 | 사용 기술 |
| --- | --- |
| Backend | Python 3.12, FastAPI, Uvicorn |
| Model | MobileNetV2 (ONNX Model Zoo, `mobilenetv2-12.onnx`, 14MB), ONNX Runtime (CPU) |
| Frontend | HTML / CSS / Vanilla JavaScript |
| Test | pytest, FastAPI TestClient |
| CI | GitHub Actions (테스트 + Docker 이미지 빌드 + 컨테이너 스모크 테스트) |
| CD | Docker + Render (CI 통과 시 자동 배포) |

## 주요 기능

- 이미지 업로드 (클릭 또는 드래그 앤 드롭) 및 미리보기
- 상위 5개 추정 결과와 확률 막대 표시, 추론 시간 표시
- 오류 처리: 지원하지 않는 형식(415), 손상·빈 파일(400), 5MB 초과(413), 모델 미준비(503)
- `/api/health`에서 모델 로드 상태와 **배포된 커밋 해시** 확인 → push 후 서버가 실제로 갱신됐는지 화면 하단에서 바로 확인 가능

## 프로젝트 구조

```
.
├── app/
│   ├── main.py               # FastAPI 라우트 (/, /api/health, /api/predict)
│   ├── model.py              # 전처리 + ONNX Runtime 추론
│   ├── imagenet_classes.txt  # 1000개 클래스 이름
│   └── static/               # index.html, style.css, app.js
├── scripts/download_model.py # 모델 다운로드 + SHA256 검증
├── tests/test_api.py         # API 테스트
├── .github/workflows/ci.yml  # CI 파이프라인
├── Dockerfile
├── render.yaml               # Render 배포 설정 (Blueprint)
├── requirements.txt
└── requirements-dev.txt
```

## 로컬에서 실행하기

Python 3.10 이상이 필요합니다. (Windows PowerShell 기준, macOS/Linux도 동일)

```powershell
# 1. 저장소 받기
git clone https://github.com/kyungrae2002/opsw_hw2.git
cd opsw_hw2

# 2. 패키지 설치
python -m pip install -r requirements.txt

# 3. 모델 준비 (models/mobilenetv2-12.onnx, 약 14MB, SHA256 자동 검증)
python scripts/download_model.py

# 4. 서버 실행
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 <http://127.0.0.1:8000> 을 열고 이미지를 올려 **분석하기**를 누르면 됩니다.
API 문서는 <http://127.0.0.1:8000/docs> 에서 볼 수 있습니다.

### Docker로 실행하기

```bash
docker build -t snaplabel .
docker run -p 8000:8000 snaplabel
```

Docker 이미지는 빌드 단계에서 모델을 내려받으므로 따로 준비할 필요가 없습니다.

## 테스트

```powershell
python -m pip install -r requirements-dev.txt
python scripts/download_model.py
python -m pytest -v
```

## API

| Method | Path | 설명 |
| --- | --- | --- |
| GET | `/` | 웹 화면 |
| GET | `/api/health` | 서버·모델 상태, 버전, 배포 커밋 |
| POST | `/api/predict` | `multipart/form-data`의 `file` 필드로 이미지 전송 → 상위 5개 결과 |

응답 예시:

```json
{
  "filename": "dog.jpg",
  "model": "mobilenetv2-12.onnx",
  "inference_ms": 31.3,
  "predictions": [
    {"rank": 1, "label": "Samoyed", "class_id": 258, "probability": 0.83},
    {"rank": 2, "label": "Pomeranian", "class_id": 259, "probability": 0.07}
  ],
  "notice": "결과는 ImageNet 1000개 클래스 기준 모델의 추정치이며 정확하지 않을 수 있습니다."
}
```

## 배포 파이프라인 (DevOps / MLOps)

```
코드 수정 → git push (main)
        │
        ▼
GitHub Actions CI
  ├─ test   : 의존성 설치 → 모델 다운로드(SHA256 검증, 캐시) → pytest
  └─ docker : 이미지 빌드 → 컨테이너 실행 → /api/health 스모크 테스트
        │  (모든 체크 통과 시)
        ▼
Render 자동 배포 (autoDeployTrigger: checksPass)
  └─ Docker 이미지 빌드 → /api/health 헬스체크 통과 후 트래픽 전환
```

- 모델 파일은 Git에 올리지 않고(`.gitignore`), 빌드할 때 내려받아 **SHA256 해시로 무결성을 검증**합니다.
- CI가 실패하면 배포되지 않으므로 망가진 코드가 서버에 올라가지 않습니다.
- 화면 하단의 `commit xxxxxxx`가 새 커밋 해시로 바뀌면 자동 배포가 완료된 것입니다.

### Render에 배포하기 (최초 1회)

1. <https://render.com> 에 GitHub 계정으로 로그인
2. **New → Blueprint** 선택 후 이 저장소(`opsw_hw2`) 연결
3. `render.yaml`이 자동으로 인식됨 → **Apply**
4. 빌드가 끝나면 `https://snaplabel-xxxx.onrender.com` 형태의 주소로 접속

이후에는 코드를 수정해 `main`에 push하기만 하면 CI 통과 후 자동으로 서버가 갱신됩니다.
(무료 플랜은 15분간 요청이 없으면 잠들어 첫 접속이 30초가량 걸릴 수 있습니다.)

## 모델 출처

- MobileNetV2 — Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks* (CVPR 2018)
- ONNX 변환 모델: [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet) (Apache-2.0)
