# 모델 업데이트 실험 리포트 (v1.0.0 → v1.1.0)

## 1. 실험 설계

| 항목 | 내용 |
|---|---|
| 목적 | 대규모 재학습 없이(전처리 개선 또는 사전학습 모델 교체) 분류 정확도 향상 |
| 평가 데이터 | Hugging Face [`vaishaal/ImageNetV2`](https://huggingface.co/datasets/vaishaal/ImageNetV2) matched-frequency (**MIT License**). ImageNet 모델 평가용으로 새로 수집된 테스트셋 |
| 표본 | 고정 시드(2026)로 무작위 100건, 97개 클래스 → **tune 30건 / test 70건** |
| 규칙 | 개선 방법 선택은 tune 30건으로만 함. test 70건은 최종 비교에 한 번만 사용. 개선 시도 최대 2회 |
| 지표 | top-1 정확도(주 지표), top-5 정확도, 이미지당 추론 시간(CPU, ONNX Runtime 1.31.1) |
| 실행 환경 | 서비스와 같은 라이브러리 버전(requirements.txt), 같은 전처리 코드(`app/model.py`) |

재현 방법:
```bash
python scripts/download_model.py --all
python scripts/prepare_eval_data.py          # data/eval/ (Git 제외)
python scripts/evaluate.py --split tune mobilenetv2-12 mobilenetv2-12-tta efficientnet-lite4-11
python scripts/evaluate.py --split test mobilenetv2-12 efficientnet-lite4-11
```

## 2. 기존 모델 오류 분석 (tune 30건, MobileNetV2: 오답 14건)

| 오류 유형 | 예시 (정답 → 예측) |
|---|---|
| 비슷한 세부 클래스 혼동 (견종 등) | redbone → bloodhound, Border terrier → wire-haired fox terrier |
| 주변 사물/배경에 끌려감 | traffic light → pole, oxygen mask → military uniform, black grouse → dogsled |
| 낮은 확신의 엉뚱한 예측 (확률 < 0.2) | triceratops → knee pad (0.14), Arabian camel → volcano (0.11), axolotl → green lizard (0.18) |

→ 작은 모델(MobileNetV2, ImageNet top-1 약 70%)의 표현력 한계로 판단해, 두 가지 방향을 시도함.

## 3. 개선 시도 (tune 30건으로만 선택)

| 설정 | 내용 | top-1 | top-5 | ms/img |
|---|---|---|---|---|
| 기준선 `mobilenetv2-12` | 기존 서비스 모델 | 53.3% | 80.0% | 4.9 |
| 시도 1 `mobilenetv2-12-tta` | 전처리 개선: BICUBIC 리사이즈 + 좌우반전 TTA 평균 | 50.0% | 76.7% | 9.3 |
| **시도 2 `efficientnet-lite4-11`** | 사전학습 모델 교체 (ONNX Model Zoo, MIT, ImageNet top-1 80.4%) | **63.3%** | **83.3%** | 15.2 |

→ tune 기준 **시도 2 선택** (시도 1은 오히려 하락해 제외).

## 4. 최종 비교 (test 70건, 선택에 사용하지 않은 데이터)

| 모델 | top-1 | top-5 | ms/img |
|---|---|---|---|
| 변경 전 MobileNetV2 | 58.6% (41/70) | 80.0% (56/70) | 4.6 |
| 변경 후 EfficientNet-Lite4 | 58.6% (41/70) | **87.1% (61/70)** | 15.0 |
| 차이 | ±0.0%p | **+7.1%p** | +10.4ms |

이미지별 변화 (top-1): **고쳐짐 7건 / 새로 틀림 7건**

| 구분 | 이미지 (정답: 변경 전 → 변경 후) |
|---|---|
| 고쳐짐 | banana: pot(0.10) → banana(1.00) · analog clock: street sign → analog clock(0.97) · punching bag: pay-phone → punching bag(1.00) · cabbage butterfly: ringlet → 정답(0.98) · Old English sheepdog: wire-haired fox terrier → 정답(0.99) · Pembroke: Chihuahua → 정답(0.99) · fur coat: stole → 정답(0.72) |
| 새로 틀림 | cup → coffee mug · projectile → missile · siamang → gibbon · paddle → canoe · steel arch bridge → pier · magnetic compass → stopwatch · whistle → torch |

## 5. 결론

- **top-1은 test 70건에서 개선되지 않았다(58.6% → 58.6%).** tune에서 본 +10%p 향상은 표본이 작아(30건) 재현되지 않았다. 차이도 통계적으로 의미 없다(McNemar p = 1.0).
- **top-5는 80.0% → 87.1%로 향상**되었다. 서비스 화면은 상위 5개 결과를 보여 주므로 사용자가 정답을 찾을 확률이 높아졌다.
- 오류의 질이 바뀌었다. 변경 전 모델이 고친 7건은 대부분 **엉뚱한 오답**(바나나 → 냄비, 시계 → 표지판)이었다. 변경 후 모델이 새로 틀린 7건은 대부분 **의미가 거의 같은 클래스 혼동**(cup ↔ coffee mug, projectile ↔ missile, siamang ↔ gibbon)이었다.
- 대가: 추론 시간 약 3배(4.6 → 15.0ms), 모델 크기 14MB → 52MB. 웹 서비스 응답 시간(수십 ms) 안에서는 감당할 수 있는 수준이다.
- 사전에 정한 절차(tune으로 선택 → test로 확인)에 따라 EfficientNet-Lite4를 v1.1.0으로 배포했다. top-1 향상은 이 실험 규모로는 확인되지 않았다는 점을 함께 기록한다.

### 한계
- 표본 70건이라 1건 차이가 1.4%p다. 정확도 차이를 확정하려면 더 큰 평가셋이 필요하다.
- ImageNetV2의 원래 분포에서 무작위로 뽑았으며 서비스 실제 사용자 사진 분포와는 다를 수 있다.
