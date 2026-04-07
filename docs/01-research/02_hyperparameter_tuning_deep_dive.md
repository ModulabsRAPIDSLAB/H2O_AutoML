# Hyperparameter Tuning 심화 이해

> H2O AutoML이 하이퍼파라미터를 어떻게 탐색하는지, 왜 그 방식을 선택했는지까지 설명할 수 있도록 정리한 자료이다.

---

## 1. 하이퍼파라미터란 무엇인가

### 1-1. 파라미터 vs 하이퍼파라미터

**파라미터 (Parameter):** 모델이 학습 과정에서 **스스로 찾는** 값
```
예: 선형 회귀의 기울기(w), 절편(b)
    신경망의 가중치(weight)
    → 학습 데이터로부터 자동으로 결정됨
```

**하이퍼파라미터 (Hyperparameter):** 학습 **전에 사람이 설정**하는 값
```
예: XGBoost의 max_depth=6, learning_rate=0.1, n_estimators=100
    → 이 값에 따라 모델의 구조와 학습 방식이 완전히 달라짐
    → "어떤 값이 최적인가?"를 찾는 것이 HPO
```

### 1-2. 왜 하이퍼파라미터 튜닝이 중요한가

동일한 XGBoost 알고리즘이라도:

```
설정 A: max_depth=3, learning_rate=0.01, n_estimators=1000
→ AUC: 0.95 (느리지만 정확)

설정 B: max_depth=10, learning_rate=0.3, n_estimators=100
→ AUC: 0.88 (빠르지만 과적합)

설정 C: max_depth=6, learning_rate=0.1, n_estimators=500
→ AUC: 0.97 (최적의 균형)
```

같은 알고리즘인데 **설정만 바꿔도 성능이 크게 달라진다**.
최적의 설정을 찾는 것이 HPO(Hyperparameter Optimization)이다.

### 1-3. 주요 하이퍼파라미터 예시

**XGBoost / GBM:**

| 하이퍼파라미터 | 의미 | 영향 |
|----------------|------|------|
| `max_depth` | 트리의 최대 깊이 | 클수록 복잡 → 과적합 위험 증가 |
| `learning_rate` (eta) | 학습률 | 작을수록 안정적이지만 느림 |
| `n_estimators` | 트리 수 | 많을수록 성능 ↑ 하지만 수렴 후 효과 없음 |
| `min_child_weight` | 리프 노드의 최소 가중치 | 클수록 보수적 (과적합 방지) |
| `subsample` | 각 트리에 사용할 데이터 비율 | 1.0 미만이면 랜덤성 추가 |
| `colsample_bytree` | 각 트리에 사용할 특성 비율 | 1.0 미만이면 특성 다양성 추가 |
| `reg_alpha` (L1) | L1 정규화 강도 | 특성 선택 효과 |
| `reg_lambda` (L2) | L2 정규화 강도 | 가중치 축소 효과 |

**Random Forest:**

| 하이퍼파라미터 | 의미 | 영향 |
|----------------|------|------|
| `n_estimators` | 트리 수 | 많을수록 안정적 (보통 100~500) |
| `max_depth` | 트리 깊이 | RF는 보통 깊게 키움 (None = 제한 없음) |
| `max_features` | 분할 시 고려할 특성 수 | 다양성과 성능의 균형 |

**GLM:**

| 하이퍼파라미터 | 의미 | 영향 |
|----------------|------|------|
| `alpha` | 정규화 혼합 비율 | 0=Ridge, 1=Lasso, 0~1=Elastic Net |
| `lambda` | 정규화 강도 | 클수록 모델 단순화 |

---

## 2. 탐색 방법: Grid Search vs Random Search

### 2-1. Grid Search (격자 탐색)

모든 가능한 조합을 **빠짐없이** 시도한다.

```
max_depth = [3, 6, 9]
learning_rate = [0.01, 0.1, 0.3]
n_estimators = [100, 500, 1000]

총 조합: 3 × 3 × 3 = 27가지 → 27개 모델 훈련
```

**장점:** 최적 조합을 반드시 찾음 (탐색 범위 내에서)
**치명적 단점:** 차원의 저주

```
하이퍼파라미터가 5개, 각각 5개 값이면:
5^5 = 3,125개 조합

하이퍼파라미터가 8개, 각각 5개 값이면:
5^8 = 390,625개 조합 → 현실적으로 불가능
```

### 2-2. Random Search (랜덤 탐색)

정해진 범위에서 **랜덤으로 조합을 뽑아** 시도한다.

```
max_depth: 범위 [2, 15] 중 랜덤
learning_rate: 범위 [0.001, 0.5] 중 랜덤 (log scale)
n_estimators: 범위 [50, 2000] 중 랜덤

100번 랜덤 샘플링 → 100개 모델 훈련
```

### 2-3. 왜 Random Search가 더 효과적인가

**Bergstra & Bengio (2012)의 핵심 발견:**

대부분의 ML 문제에서 **모든 하이퍼파라미터가 동등하게 중요하지 않다.**
보통 1~2개의 하이퍼파라미터가 성능의 대부분을 결정한다.

```
예: XGBoost에서 learning_rate가 성능의 70%를 결정하고,
    max_depth가 20%를 결정하고,
    나머지는 10% 미만의 영향

Grid Search (27개):
  learning_rate 값: 0.01, 0.1, 0.3  → 3가지만 시도
  max_depth 값: 3, 6, 9             → 3가지만 시도
  
Random Search (27개):
  learning_rate 값: 0.003, 0.015, 0.042, 0.087, ..., 0.45  → 27가지 다양한 값
  max_depth 값: 2, 4, 5, 7, 8, 11, ...                     → 27가지 다양한 값
```

**Grid Search**는 중요한 learning_rate에 대해 3가지만 시도한다.
**Random Search**는 같은 27번의 시도로 27가지 다양한 learning_rate를 탐색한다.

→ **같은 예산(시도 횟수)으로 중요한 하이퍼파라미터의 탐색 범위가 훨씬 넓다.**

이것이 H2O가 Random Search를 핵심으로 사용하는 이유이다.

### 2-4. 직관적 비유

바닥에 보물이 묻혀 있고, 27번 삽질할 수 있다고 하자.

```
Grid Search: 바닥을 3×9 격자로 나눠서 정확히 교차점만 파봄
→ 보물이 격자점 사이에 있으면 영원히 못 찾음

Random Search: 27번 랜덤으로 파봄
→ 보물 근처를 지나갈 확률이 훨씬 높음
```

---

## 3. H2O AutoML의 훈련 순서 전략

### 3-1. 전체 순서

H2O AutoML은 단순히 "모델을 마구 돌리는" 것이 아니라, **의도적인 순서**로 훈련한다.

```
Phase 1: Pre-specified Models (기본 모델)
─────────────────────────────────────────
│ 검증된 기본 파라미터로 각 알고리즘별 1개씩 훈련
│ 예: XGBoost(default), GBM(default), RF(default), GLM(default), DNN(default)
│ 목적: 빠르게 baseline 성능 확보
│ 소요 시간: 전체의 ~20%
│
Phase 2: Grid Search (다양성 확장)
─────────────────────────────────────────
│ 알고리즘별로 소수의 파라미터 조합을 시도
│ 예: XGBoost(max_depth=3), XGBoost(max_depth=9), ...
│ 목적: model diversity 확보 → Stacking 성능 향상
│ 소요 시간: 전체의 ~30%
│
Phase 3: Random Search (탐색 범위 확대)
─────────────────────────────────────────
│ 넓은 범위에서 랜덤 하이퍼파라미터 샘플링
│ 목적: Phase 1-2에서 놓친 좋은 조합 발견
│ 소요 시간: 전체의 ~40%
│
Phase 4: Stacked Ensemble 생성
─────────────────────────────────────────
│ 위에서 만들어진 모든 base model로 앙상블 구성
│ All Models + Best of Family 두 가지 생성
│ 소요 시간: 전체의 ~5%
│
Phase 5: Adaptive Extension (시간이 남으면)
─────────────────────────────────────────
│ 상위 Random Search 그리드 재시작
│ 더 많은 조합 탐색 → 새 base model 추가 → 앙상블 재구성
│ 소요 시간: 남은 시간 전부
```

### 3-2. 왜 이 순서인가

**Phase 1이 먼저인 이유:**
- baseline이 있어야 "이 모델이 좋은 건지 나쁜 건지" 판단 가능
- 기본 파라미터가 의외로 성능이 좋은 경우가 많음 (특히 XGBoost)
- 시간이 부족해서 중간에 멈춰도 최소한의 결과물이 보장됨

**Phase 2가 Grid Search인 이유:**
- 이 단계의 목적은 "최적 하이퍼파라미터 찾기"가 아님
- **서로 다른 특성의 모델을 만들기** 위함 (Stacking을 위한 diversity)
- 소수의 검증된 조합만 시도하면 충분

**Phase 3이 Random Search인 이유:**
- Phase 1-2에서 다양성은 확보됨
- 이제 넓은 범위에서 "혹시 놀라운 조합이 있나" 탐색
- 시간 기반으로 제어 가능 (max_runtime_secs)

**Phase 5 (Adaptive)의 의미:**
```
max_runtime_secs=3600 (1시간)으로 설정했는데:
Phase 1-4가 40분에 끝남
→ 남은 20분으로 Phase 3의 Random Search를 계속 수행
→ 더 많은 모델 훈련 → 더 나은 앙상블 가능
```

### 3-3. 시간 기반 제어의 장점

H2O AutoML은 `max_runtime_secs`로 **시간 예산**을 관리한다.

```python
aml = H2OAutoML(max_runtime_secs=300)  # 5분
aml = H2OAutoML(max_runtime_secs=3600) # 1시간
```

이 접근의 장점:
1. **예측 가능성**: "5분이면 결과 나옵니다" → 비즈니스 일정에 맞출 수 있음
2. **자동 확장**: 시간을 더 주면 자동으로 더 많은 모델 탐색
3. **점진적 개선**: 5분 → 1시간으로 바꾸면 성능이 올라가는 게 보임

### 3-4. 알고리즘별 시간 분배

H2O는 모든 알고리즘에 동일한 시간을 주지 않는다.

```
시간 분배 (대략적):
XGBoost:        ████████████████  35%
GBM:            ████████████      25%
Random Forest:  ████████          15%
DNN:            ██████            12%
GLM:            ████               8%
XRT:            ██                 5%
```

**왜 XGBoost에 더 많은 시간을 주는가?**
- 대부분의 tabular 데이터에서 XGBoost가 최고 성능을 보임
- 하이퍼파라미터에 민감 → 더 많이 탐색하면 더 좋은 결과
- GBM도 gradient boosting family라서 비슷한 이유로 시간을 많이 받음

**GLM에 적은 시간을 주는 이유:**
- GLM은 하이퍼파라미터가 적음 (alpha, lambda 정도)
- 학습 자체가 빠름
- 적은 시간으로도 최적에 가까운 결과를 얻을 수 있음

---

## 4. Random Search의 실제 동작

### 4-1. 탐색 범위 예시 (H2O XGBoost)

H2O가 내부적으로 사용하는 XGBoost Random Search 범위:

```
max_depth:           [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
learning_rate:       [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
n_estimators:        내부적으로 early stopping으로 자동 결정
sample_rate:         [0.6, 0.7, 0.8, 0.9, 1.0]
col_sample_rate:     [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
min_rows:            [1, 5, 10, 25, 50, 100]
reg_alpha:           [0, 0.001, 0.01, 0.1, 0.5, 1]
reg_lambda:          [0, 0.001, 0.01, 0.1, 1, 5, 10]
```

가능한 조합 수: `11 × 8 × 5 × 6 × 6 × 6 × 7 = 약 4백만 개`

Grid Search로는 불가능. Random Search로 수십~수백 개만 샘플링해서 시도한다.

### 4-2. Early Stopping

H2O는 `n_estimators`를 직접 탐색하지 않고 **early stopping**을 사용한다.

```
설정: max_trees=10000, stopping_rounds=5
동작:
  트리 1~50: 성능 지속 상승
  트리 51~100: 성능 상승 둔화
  트리 101~105: 성능 변화 없음 (5라운드 연속)
  → 자동 중단, 최적 트리 수 = 100으로 결정
```

이렇게 하면 n_estimators를 별도로 탐색하지 않아도 된다.
→ 탐색 공간이 1차원 줄어드는 효과.

### 4-3. 한 번의 Random Search 시도에서 일어나는 일

```
1. 랜덤으로 파라미터 조합 생성
   예: max_depth=7, learning_rate=0.05, sample_rate=0.8, ...

2. 이 조합으로 5-fold CV 수행
   Fold 1: 훈련 → 검증 AUC = 0.9712
   Fold 2: 훈련 → 검증 AUC = 0.9698
   Fold 3: 훈련 → 검증 AUC = 0.9725
   Fold 4: 훈련 → 검증 AUC = 0.9701
   Fold 5: 훈련 → 검증 AUC = 0.9718

3. CV 평균 성능 계산: 0.9711

4. Leaderboard에 등록

5. OOF prediction 생성 (Stacking용)

6. 다음 랜덤 조합으로 이동
```

이것이 시간이 소진될 때까지 반복된다.

---

## 5. Grid Search vs Random Search 핵심 비교

| 기준 | Grid Search | Random Search |
|------|-------------|---------------|
| 탐색 방식 | 모든 조합 시도 | 랜덤 샘플링 |
| 차원 확장성 | 차원의 저주 (조합 폭발) | 차원에 거의 무관 |
| 중요 HP 커버리지 | 격자점에 제한 | 연속적 탐색 가능 |
| 비중요 HP 처리 | 동일한 비용 소모 | 자연스럽게 무시됨 |
| 최적해 보장 | 격자 내에서 보장 | 보장하지 않음 (확률적) |
| 시간 효율성 | 낮음 | 높음 |
| H2O AutoML 사용 | Phase 2 (소규모) | **Phase 3 (핵심)** |

---

## 6. 시간 기반 제어의 실제 동작

### 6-1. max_runtime_secs=300 (5분)일 때

```
0:00  Phase 1 시작 — Pre-specified 모델 훈련
0:45  XGBoost(default) 완료 → Leaderboard 등록
1:10  GBM(default) 완료
1:35  RF(default) 완료
1:50  GLM(default) 완료
2:05  DNN(default) 완료

2:05  Phase 2 시작 — Grid Search
2:40  XGBoost(max_depth=3) 완료
3:15  XGBoost(max_depth=9) 완료
3:45  GBM(ntrees=200) 완료

3:45  Phase 3 시작 — Random Search
4:10  XGBoost(random_1) 완료
4:30  GBM(random_1) 완료

4:30  Phase 4 — Stacked Ensemble 생성
4:45  StackedEnsemble_AllModels 완료
4:55  StackedEnsemble_BestOfFamily 완료

5:00  시간 종료 → Leaderboard 반환
```

### 6-2. max_runtime_secs=3600 (1시간)이면?

Phase 1-2는 동일. Phase 3(Random Search)에 **55분** 이상 투입.
→ 수십~수백 개의 추가 모델 탐색 가능
→ 더 좋은 하이퍼파라미터 발견 확률 급증
→ Stacking의 base model이 더 다양해짐
→ 앙상블 성능 향상

**이것이 H2O의 핵심 철학:**
> "빠른 인프라로 단순한 서칭을 많이 돌리자"
> 시간을 더 주면 자동으로 더 좋은 결과가 나온다.
