# Stacked Ensemble 심화 이해

> 이 문서는 발표에서 "왜 이렇게 하는 건가요?"라는 질문에 답할 수 있을 정도로 깊이 있게 정리한 자료이다.

---

## 1. Stacking이란 무엇인가

### 1-1. 가장 단순한 설명

여러 모델의 예측 결과를 **새로운 모델의 입력**으로 사용하는 기법이다.

```
[입력 데이터] → 모델A → 예측A
             → 모델B → 예측B     →  [예측A, 예측B, 예측C]  →  Meta 모델  →  최종 예측
             → 모델C → 예측C
```

단순 평균(averaging)이나 투표(voting)와 다른 점:
- 평균/투표: 각 모델의 기여도가 동일하거나 수동 설정
- **Stacking**: 어떤 모델을 얼마나 신뢰할지를 **데이터로부터 학습**

### 1-2. 왜 이게 성능을 높이는가

핵심 원리는 **모델 간 오류 패턴이 다르다**는 점이다.

예를 들어 사기 거래 탐지에서:
- XGBoost는 금액이 큰 사기를 잘 잡지만, 소액 사기를 놓침
- Random Forest는 소액 사기를 잘 잡지만, 특정 시간대 패턴을 놓침
- GLM은 전체적으로 무난하지만 극단적 케이스에 약함

Meta 모델이 학습하는 것:
> "XGBoost가 0.9 이상을 예측했으면 대부분 맞는데, 0.3 ~ 0.6 사이일 때는 Random Forest 예측을 더 신뢰하는 게 낫더라"

이것을 **자동으로** 학습한다. 사람이 규칙을 만드는 게 아니라 데이터가 알려주는 것이다.

### 1-3. 수학적으로 왜 작동하는가 (직관적 설명)

각 모델의 예측 오차를 `e_A`, `e_B`, `e_C`라 하자.

**단순 평균의 경우:**
```
오차 = (e_A + e_B + e_C) / 3
```
→ 오차들이 서로 상관없으면(uncorrelated) 평균만 내도 오차가 줄어든다.

**Stacking의 경우:**
```
오차 = w_A × e_A + w_B × e_B + w_C × e_C
여기서 w_A, w_B, w_C는 학습된 가중치
```
→ 오차가 큰 모델의 가중치를 낮추고, 신뢰할 수 있는 모델의 가중치를 높인다.
→ 단순 평균보다 **항상 같거나 더 나은** 결과를 얻을 수 있다.

**핵심 조건: Model Diversity (모델 다양성)**

모든 모델이 같은 실수를 하면 Stacking도 소용없다.
H2O가 의도적으로 XGBoost, GBM, RF, GLM, DNN, XRT 등
**서로 다른 family의 알고리즘**을 섞는 이유가 바로 이것이다.

```
같은 family (다양성 낮음):     다른 family (다양성 높음):
XGBoost-v1                    XGBoost
XGBoost-v2    → 비슷한 실수    Random Forest  → 다른 실수
XGBoost-v3                    GLM
                               → Stacking 효과 큼
```

---

## 2. 데이터 누수(Data Leakage) 문제와 OOF 해결책

### 2-1. 왜 데이터 누수가 발생하는가

**잘못된 Stacking:**
```
1. 전체 학습 데이터로 모델A 훈련
2. 같은 학습 데이터로 모델A의 예측값 생성
3. 이 예측값을 Meta 모델의 입력으로 사용
```

문제: 모델A는 학습 데이터를 **이미 본 상태**에서 예측한다.
→ 과적합된 예측값이 Meta 모델에 전달됨
→ Meta 모델은 "모델A가 항상 정확하다"고 학습함
→ 테스트 데이터에서 성능 급락

**비유:**
시험 문제를 미리 본 학생(모델A)의 점수를 기준으로 "이 학생이 실력이 좋다"고 판단(Meta 모델)하면, 새 시험에서 망한다.

### 2-2. Out-of-Fold (OOF) Prediction이 해결하는 방법

**핵심 아이디어:** 모델이 **보지 못한 데이터**에 대한 예측만 사용한다.

5-fold CV 기준으로 구체적으로 설명하면:

```
전체 학습 데이터: [1000개 행]
├── Fold 1: 행 1 ~ 200    (검증용)
├── Fold 2: 행 201 ~ 400  (검증용)
├── Fold 3: 행 401 ~ 600  (검증용)
├── Fold 4: 행 601 ~ 800  (검증용)
└── Fold 5: 행 801 ~ 1000 (검증용)
```

**모델A에 대한 OOF 생성 과정:**

```
Round 1: 행 201 ~ 1000으로 훈련 → 행 1 ~ 200에 대해 예측    → OOF[1:200]
Round 2: 행 1 ~ 200 + 401 ~ 1000으로 훈련 → 행 201 ~ 400 예측 → OOF[201:400]
Round 3: 행 1 ~ 400 + 601 ~ 1000으로 훈련 → 행 401 ~ 600 예측 → OOF[401:600]
Round 4: 행 1 ~ 600 + 801 ~ 1000으로 훈련 → 행 601 ~ 800 예측 → OOF[601:800]
Round 5: 행 1 ~ 800으로 훈련 → 행 801 ~ 1000에 대해 예측    → OOF[801:1000]
```

**결과:** 모든 1000개 행에 대해 "모델이 보지 못한 상태에서의 예측값"이 만들어진다.

이것이 **공정한 예측값**이다. 시험 문제를 모르는 상태에서 본 시험 점수와 같다.

### 2-3. Level-One Data (메타 특성 행렬)

L개의 base model에 대해 위 과정을 반복하면:

```
         모델A_예측  모델B_예측  모델C_예측  ...  모델L_예측
행 1     0.82       0.75       0.91       ...  0.88
행 2     0.15       0.23       0.12       ...  0.19
행 3     0.67       0.71       0.55       ...  0.63
...
행 1000  0.93       0.89       0.95       ...  0.91
```

이것이 **Level-One Data** (= 메타 특성 행렬)이다.
- 행: 원본 학습 데이터의 각 샘플 (N개)
- 열: 각 base model의 OOF 예측값 (L개)
- 크기: N × L

Meta 모델은 이 Level-One Data를 입력으로, 원본 타겟(y)을 출력으로 학습한다.

### 2-4. 왜 5-fold인가

- **fold 수가 적으면 (예: 2-fold):** 각 fold의 훈련 데이터가 50%뿐 → base model 성능 저하 → OOF 예측 품질 저하
- **fold 수가 많으면 (예: 10-fold):** OOF 품질은 좋지만 훈련 횟수가 10 × L번 → 시간 급증
- **5-fold:** 훈련 데이터의 80%를 사용하면서 합리적인 시간 → **성능과 비용의 균형점**

H2O AutoML은 기본적으로 **5-fold CV**를 사용한다.

---

## 3. H2O의 Two-Type Ensemble 전략

### 3-1. All Models Ensemble

```
포함 모델: 학습된 모든 base model
목적: 최고 성능 달성
```

가능한 많은 모델을 포함시켜서 model diversity를 극대화한다.
Leaderboard에서 보통 1등을 차지하는 것이 이 앙상블이다.

**장점:** 최고 정확도
**단점:** 모델 수가 많으면 예측 시간이 길고 메모리 사용량이 큼

### 3-2. Best of Family Ensemble

```
포함 모델: 각 알고리즘 family에서 최고 1개씩만
- Best XGBoost (1개)
- Best GBM (1개)
- Best RF (1개)
- Best GLM (1개)
- Best DNN (1개)

총 5 ~ 6개 모델만 포함
```

**장점:**
- 예측 속도가 빠름 (모델 수가 적으니까)
- 메모리 효율적
- 프로덕션 배포에 적합
- **model diversity는 유지** (서로 다른 family이므로)

**단점:** All Models보다 정확도가 약간 낮을 수 있음

### 3-3. 왜 두 가지를 모두 만드는가

```
연구/실험 단계: All Models Ensemble → "이 데이터에서 달성 가능한 최고 성능이 이거다"
프로덕션 배포: Best of Family Ensemble → "현실적 제약(속도, 메모리) 하에서 최적의 모델"
```

이것은 **"성능 vs 속도" 트레이드오프를 동시에 해결**하는 설계이다.
사용자가 선택할 수 있게 두 가지를 다 제공하는 것이 H2O의 접근이다.

### 3-4. 실제 Leaderboard에서 어떻게 보이는가

01번 노트북에서 실행하면 다음과 비슷한 결과가 나온다:

```
순위  모델 ID                              AUC
1    StackedEnsemble_AllModels_...         0.9823
2    StackedEnsemble_BestOfFamily_...      0.9819
3    XGBoost_3_...                         0.9801
4    GBM_5_...                             0.9795
...
```

All Models가 1등, Best of Family가 2등인 경우가 매우 흔하다.
둘 다 개별 모델보다 높은 성능을 보인다.

---

## 4. Meta Learner: 왜 GLM인가

### 4-1. Meta Learner의 역할

Meta Learner는 Level-One Data(각 base model의 OOF 예측값)를 입력으로 받아서
**각 모델의 예측을 어떻게 결합할지** 학습한다.

```
Meta Learner 학습:
입력 = [XGB_pred, GBM_pred, RF_pred, GLM_pred, DNN_pred]
출력 = 실제 타겟 (is_fraud)

학습 결과 예시:
최종 예측 = 0.35 × XGB_pred + 0.25 × GBM_pred + 0.20 × RF_pred + 0.15 × GLM_pred + 0.05 × DNN_pred
```

### 4-2. H2O가 GLM을 선택한 이유

**이유 1: 안정성**

Level-One Data는 이미 잘 훈련된 모델들의 예측값이다.
이 위에 복잡한 모델(XGBoost 등)을 올리면 **이중 과적합** 위험이 있다.
GLM(선형 모델)은 복잡도가 낮아서 이 위험을 최소화한다.

**이유 2: 해석 가능성**

GLM의 계수(coefficient)가 곧 각 모델의 가중치다.
```
GLM 계수: XGB=0.35, GBM=0.25, RF=0.20, GLM=0.15, DNN=0.05
→ "XGBoost가 가장 중요하고 DNN은 거의 기여하지 않는다"는 걸 바로 알 수 있다
```

**이유 3: 빠른 연산**

GLM은 수학적으로 closed-form solution에 가까워서
CPU에서도 수 초 안에 학습이 끝난다. AutoML 전체 시간에 거의 영향을 주지 않는다.

### 4-3. Non-negative 제약 조건

H2O의 GLM meta learner는 **가중치가 0 이상**이어야 한다는 제약을 건다.

```
허용: w_XGB=0.35, w_GBM=0.25, w_RF=0.00 (RF 제외)
불허: w_XGB=0.50, w_GBM=-0.20 (음수 가중치)
```

**왜?**
- 음수 가중치는 "이 모델의 예측을 반대로 사용하라"는 의미
- 이론적으로 가능하지만 실전에서는 불안정한 결과를 초래
- Non-negative로 제한하면 앙상블이 더 안정적

### 4-4. Lasso/Elastic Net Regularization → Sparse Ensemble

H2O는 GLM에 **L1 정규화(Lasso)** 또는 **Elastic Net**을 적용한다.

Lasso의 효과:
```
학습 전: w_XGB=?, w_GBM=?, w_RF=?, w_GLM=?, w_DNN=?
학습 후: w_XGB=0.45, w_GBM=0.30, w_RF=0.25, w_GLM=0.00, w_DNN=0.00
                                             ↑ Lasso가 0으로 만듦
```

불필요한 모델의 가중치를 **정확히 0**으로 만든다.
→ **Sparse Ensemble**: 실제로 예측에 사용되는 모델 수가 줄어든다.
→ 예측 속도 향상 + 모델 저장 용량 절약

이것이 Best of Family Ensemble과 맞물리면:
- 5 ~ 6개 모델 중 실제로 가중치가 0이 아닌 것은 3 ~ 4개
- 매우 가벼운 프로덕션 모델이 완성된다

---

## 5. 전체 Stacking 흐름 (처음부터 끝까지)

아래는 H2O AutoML에서 Stacked Ensemble이 만들어지는 전체 과정이다.

```
Step 1: 데이터 분할
========================================
전체 데이터 → 5-fold 분할 생성

Step 2: Base Model 훈련 + OOF 생성 (가장 오래 걸리는 단계)
========================================
각 base model(XGB, GBM, RF, GLM, DNN, ...)에 대해:
  └─ 5-fold CV 수행
     ├─ Fold 1: 나머지 4개 fold로 훈련 → Fold 1에 대해 예측 → OOF[fold1]
     ├─ Fold 2: 나머지 4개 fold로 훈련 → Fold 2에 대해 예측 → OOF[fold2]
     ├─ Fold 3: 나머지 4개 fold로 훈련 → Fold 3에 대해 예측 → OOF[fold3]
     ├─ Fold 4: 나머지 4개 fold로 훈련 → Fold 4에 대해 예측 → OOF[fold4]
     └─ Fold 5: 나머지 4개 fold로 훈련 → Fold 5에 대해 예측 → OOF[fold5]
  OOF 전체 = [OOF[fold1], OOF[fold2], ..., OOF[fold5]] 합침

Step 3: Level-One Data 구성
========================================
         XGB_OOF  GBM_OOF  RF_OOF  GLM_OOF  DNN_OOF
행 1     0.82     0.75     0.91    0.70     0.88
행 2     0.15     0.23     0.12    0.18     0.19
...
행 N     0.93     0.89     0.95    0.87     0.91

Step 4: Meta Learner 훈련
========================================
GLM(non-negative, L1 regularization)을 Level-One Data로 훈련
→ 각 모델의 최적 가중치 학습

Step 5: 앙상블 예측 (새 데이터에 대해)
========================================
새 데이터 → 각 base model이 예측 → Meta Learner가 결합 → 최종 예측
```

**시간 분배 (대략적):**
- Step 2 (Base Model 훈련 + OOF): 전체 시간의 **~95%**
- Step 3 (Level-One Data): 거의 0
- Step 4 (Meta Learner): 전체 시간의 **~1%**
- Step 5 (예측): 실시간

→ **GPU 가속이 가장 큰 효과를 주는 곳은 Step 2**이다.
