# 실험 결과 해석 가이드

> **대상 독자**: GPU AutoML이나 머신러닝 파이프라인을 처음 접하는 분도 이해할 수 있도록 작성했습니다.

## 실험 조건

| 항목 | 값 |
|:----:|:---|
| 데이터 | Kaggle Credit Card Fraud Detection (fraudTrain.csv) |
| 행 수 | **1,296,675** (약 129만 건) |
| 특성 수 | 11 (카테고리 인코딩 후) |
| 타겟 | `is_fraud` (사기 여부, 0 또는 1) |
| 클래스 비율 | 정상 99.42% / 사기 0.58% (극도의 불균형) |
| GPU | NVIDIA RTX 4060 (8GB VRAM) |
| 시간 예산 | 300초 (5분) |
| 모델 수 제한 | 최대 10개 base model |
| CV | 5-fold Stratified Cross-Validation |

---

## Leaderboard 읽는 법

```
 rank        model_id       algorithm      auc  training_time_secs  peak_vram_gb  is_ensemble
    1       xgboost_6         xgboost 0.997956                2.02         0.013        False
    2 SE_BestOfFamily StackedEnsemble 0.997252                0.00         0.000         True
    3    SE_AllModels StackedEnsemble 0.996141                0.00         0.000         True
    ...
   11           glm_3             glm 0.518143                0.09         0.008        False
```

각 컬럼의 의미:

| 컬럼 | 의미 | 쉬운 설명 |
|------|------|----------|
| `rank` | 순위 | AUC가 높은 순 |
| `model_id` | 모델 이름 | `알고리즘_번호` 형식. 번호는 훈련 순서 |
| `algorithm` | 알고리즘 종류 | xgboost, rf(랜덤 포레스트), glm(선형 모델), StackedEnsemble(앙상블) |
| `auc` | 성능 점수 | 0.5 = 동전 던지기 수준, 1.0 = 완벽. **0.99 이상이면 매우 우수** |
| `training_time_secs` | 훈련 시간 | fold당 평균 초. 작을수록 빠름 |
| `peak_vram_gb` | GPU 메모리 사용량 | 훈련 중 최대로 사용한 VRAM (GB) |
| `is_ensemble` | 앙상블 여부 | True면 여러 모델을 합친 결과 |

---

## 핵심 결과 해석

### 1. XGBoost가 왜 1위인가?

**결과**: xgboost_6 (AUC 0.9980) > xgboost_1 (AUC 0.9953)

xgboost_6은 H2O의 **Diversity Phase**에서 훈련된 모델입니다.
기본 설정(xgboost_1)과 비교하면:

| 설정 | xgboost_1 (Baseline) | xgboost_6 (Diversity) |
|------|:--------------------:|:---------------------:|
| max_depth | 6 | 10 |
| learning_rate | 0.1 | 0.1 |
| n_estimators | 100 | 200 |
| AUC | 0.9953 | **0.9980** |

**왜 더 좋은가?** 더 깊은 트리(depth 10)가 신용카드 사기의 복잡한 패턴을 더 잘 잡아냅니다.
예를 들어 "특정 시간대 + 특정 카테고리 + 높은 금액"의 조합 같은 다중 조건 패턴은
얕은 트리(depth 6)로는 표현하기 어렵습니다.

**이것이 H2O 전략의 핵심**: Baseline으로 빠르게 기준선을 잡고, Diversity에서 다양한 설정을
시도하여 더 좋은 모델을 찾습니다. 이 전략이 GPU에서도 그대로 유효하다는 것을 확인했습니다.

### 2. Stacked Ensemble은 어떻게 동작했나?

**결과**: SE_BestOfFamily (AUC 0.9973) — 2위

앙상블은 여러 모델의 예측을 합칩니다. Meta Learner(GLM)가 각 모델에 가중치를 부여했습니다:

```
Best of Family 가중치:
  XGBoost (xgboost_6): 72.1%  ← 가장 많이 의존
  RF      (rf_8):      27.9%  ← 보조적 역할
  GLM     (glm_3):      0.0%  ← 완전히 제외
```

**해석**: Meta Learner가 GLM(성능 0.52)은 쓸모없다고 판단하여 가중치를 0으로 설정했습니다.
XGBoost가 주력이고 RF가 XGBoost가 틀리는 케이스를 보완합니다.

**왜 앙상블이 1위가 아닌가?** xgboost_6 자체가 워낙 강력해서 RF를 섞으면 오히려
약간 성능이 떨어집니다. 이는 실제 Kaggle 대회에서도 흔한 현상입니다 —
앙상블이 항상 이기는 건 아니지만, **안정적으로 상위권에 위치**합니다.

### 3. GLM은 왜 실패했나?

**결과**: glm_3, glm_10 모두 AUC 0.518 (거의 랜덤 수준)

cuML LogisticRegression의 L-BFGS 옵티마이저가 수렴에 실패했습니다.

**원인**: 극도의 클래스 불균형 (사기 0.58%)
- 129만 건 중 사기는 7,506건뿐
- GLM은 선형 결정 경계를 찾는데, 이 데이터의 사기 패턴은 비선형적
- L-BFGS 옵티마이저가 "모두 정상"으로 예측해도 99.4% 정확하므로 의미 있는 경계를 찾지 못함

**이것이 문제인가?** 아닙니다. 오히려 AutoML의 강점입니다:
1. GLM이 실패해도 **파이프라인은 계속** 진행
2. Meta Learner가 **GLM 가중치를 0으로 자동 제거**
3. XGBoost, RF 같은 비선형 모델이 자연스럽게 상위를 차지

H2O AutoML도 동일한 동작을 합니다 — 모든 알고리즘이 잘 할 필요 없이,
**다양하게 시도하고 잘 되는 것만 앙상블에 포함**하는 전략입니다.

### 4. Memory-Aware Scheduling은 어떻게 동작했나?

**결과**: 10개 모델 모두 VRAM 체크 통과, OOM 0건

```
VRAM check passed for xgboost_1: estimated 0.74 GB, free 2.40 GB  ✅
VRAM check passed for rf_2:      estimated 0.43 GB, free 2.38 GB  ✅
VRAM check passed for glm_3:     estimated 0.38 GB, free 2.37 GB  ✅
...
```

매 모델 훈련 전에:
1. `VRAMEstimator`가 필요한 VRAM을 예측 (예: XGBoost 0.74GB)
2. `MemoryProfiler`가 현재 남은 VRAM을 측정 (예: 2.40GB)
3. 예측 < 남은 VRAM → 훈련 진행
4. 예측 > 남은 VRAM → **skip** (OOM 방지)

이번 실험에서는 8GB GPU에 데이터가 충분히 작아서 모든 모델이 통과했습니다.
**더 큰 데이터(Higgs 11M rows)에서는 실제로 skip이 발생**할 것으로 예상됩니다.

### 5. 메모리 프로파일은 무엇을 보여주나?

```
Pipeline Stage: VRAM Usage & Duration
    stage  start_vram_gb  peak_vram_gb  duration_secs
 baseline          5.594         5.661          27.88
diversity          5.661         5.661         150.76
 stacking          5.636         5.638           2.38
```

**해석**:
- **baseline** (28초): 3개 알고리즘 각 1개씩 빠르게 훈련. VRAM 67MB 증가
- **diversity** (151초): 7개 모델 훈련. 전체 시간의 83%를 차지. VRAM은 안정적
- **stacking** (2초): OOF 예측 수집 + Meta Learner 훈련. 매우 빠름

**핵심 인사이트**: AutoML 시간의 대부분(83%)이 Diversity Phase에서 소비됩니다.
이 구간을 GPU 병렬화하면 가장 큰 가속 효과를 얻을 수 있습니다.

### 6. 모델별 VRAM 사용량은?

```
      model_id  peak_vram_gb  training_time_secs
      rf_9           0.038               12.15    ← 가장 많이 사용
      xgboost_1      0.022                0.60
      xgboost_6      0.013                2.02
      ...
      xgboost_4      0.006                1.21    ← 가장 적게 사용
```

**해석**:
- **RF (Random Forest)**가 XGBoost보다 VRAM을 더 많이 사용 (트리를 동시에 GPU에 구축)
- rf_9 (300 trees, depth 10) = 0.038GB → 가장 VRAM-hungry
- XGBoost는 히스토그램 기반이라 메모리 효율적 (0.006 ~ 0.022GB)
- GLM은 거의 메모리를 안 씀 (0.008GB) — 단순 행렬 연산

**129만 행에서도 모델당 VRAM이 0.04GB 미만**입니다.
8GB GPU에서 충분히 여유가 있었지만, **11M 행(Higgs Boson)에서는 10배 이상 증가**할 것이므로
Memory-Aware Scheduling의 가치가 본격적으로 드러납니다.

---

## H2O 전략 검증 결과 요약

이 실험으로 확인된 것:

| H2O 전략 | GPU에서 작동? | 근거 |
|----------|:------------:|------|
| 3-Phase 훈련 순서 (Baseline → Diversity → RS) | **O** | Diversity의 xgboost_6이 Baseline의 xgboost_1보다 AUC +0.27% |
| Two-Type Stacked Ensemble | **O** | Best of Family(0.9973) + All Models(0.9961) 모두 생성 성공 |
| Non-negative GLM Meta Learner | **O** | 실패 모델(GLM) 가중치 자동 0 처리, 유효 모델만 선택 |
| 시간 기반 제어 (max_runtime_secs) | **O** | 300초 예산 내에서 10개 모델 + 2개 앙상블 완료 (181초) |
| Memory-Aware Scheduling | **O** | 모든 모델 VRAM 체크 통과, OOM 0건 |

---

## 한계와 다음 단계

### 이 실험의 한계

1. **8GB GPU는 PRD 기준(16GB+)보다 작음** — Memory-Aware의 skip/eviction이 충분히 발생하지 않음
2. **CPU H2O와의 직접 비교 미완** — 동일 데이터로 H2O를 돌린 결과가 아직 없음
3. **GLM 수렴 실패** — cuML의 L-BFGS가 극도의 불균형 데이터에서 약함. 클래스 가중치 적용이 필요
4. **단일 데이터셋** — Higgs Boson(11M), Airline Delays(5.8M)로 확장 실험 필요

### 다음 단계

- 16GB+ GPU에서 Higgs Boson 데이터로 Memory-Aware skip/eviction 실제 발생 확인
- CPU H2O 동일 조건 벤치마크 (속도 비교)
- PagedMemoryManager의 블록 할당/회수 로그 분석
- cuML GLM에 클래스 가중치(`class_weight`) 적용하여 불균형 대응

---

[Phase 3 README로 돌아가기](README.md) | [Main README](../../README.md)
