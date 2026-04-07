# GPU 네이티브 적용 방향 심화 이해

> H2O의 기법을 커널 재작성 없이 기존 GPU 라이브러리로 어떻게 적용하는지,
> 각 단계에서 **구체적으로 무엇이 바뀌고 왜 빨라지는지**를 설명할 수 있도록 정리한 자료이다.

---

## 1. 왜 "커널 재작성 없이" 가능한가

### 1-1. H2O의 기술 스택 문제

```
H2O AutoML 기술 스택:
┌─────────────────────────────┐
│  H2O Python API              │ ← Python에서 호출
├─────────────────────────────┤
│  H2O REST API                │ ← HTTP 통신
├─────────────────────────────┤
│  H2O Core (Java)             │ ← JVM 위에서 실행
├─────────────────────────────┤
│  H2O Algorithms (Java)       │ ← GBM, RF, GLM 등 Java 구현
├─────────────────────────────┤
│  H2O Frame (Java, 자체 구현) │ ← 데이터 구조도 Java
└─────────────────────────────┘
```

이것을 GPU로 바꾸려면 Java → CUDA C++ 재작성이 필요 → **커널 수준 재작성**.
H2O 팀이 아닌 우리가 할 수 있는 일이 아니다.

### 1-2. 하지만 RAPIDS 생태계가 이미 해놨다

```
RAPIDS 생태계:
┌─────────────────────────────┐
│  Python API (sklearn 호환)   │ ← 동일한 Python 인터페이스
├─────────────────────────────┤
│  cuML (GPU ML 알고리즘)      │ ← RF, GLM, KNN 등 GPU 구현 완료
├─────────────────────────────┤
│  XGBoost GPU (tree_method)   │ ← GBM 계열 GPU 구현 완료
├─────────────────────────────┤
│  cuDF (GPU DataFrame)        │ ← 데이터 구조도 GPU
├─────────────────────────────┤
│  CUDA / cuDNN                │ ← NVIDIA GPU 하드웨어
└─────────────────────────────┘
```

**핵심 인사이트:** 우리가 커널을 짤 필요가 없다. NVIDIA/RAPIDS가 이미 GPU 커널을 구현해놨다.
우리가 할 일은 **H2O의 설계 전략(순서, 앙상블 구조, OOF 로직)을 이 라이브러리 조합으로 재현**하는 것이다.

### 1-3. 구체적으로 "import만 바꾸면 되는" 부분

cuML은 **scikit-learn과 동일한 API**를 제공한다:

```python
# scikit-learn (CPU)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# cuML (GPU) — import만 변경
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression

# 사용법은 완전히 동일
model = RandomForestClassifier(n_estimators=100, max_depth=16)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

XGBoost는 파라미터 하나만 변경:

```python
# CPU
model = xgb.XGBClassifier(tree_method='hist')

# GPU — 파라미터 하나만 변경
model = xgb.XGBClassifier(tree_method='hist', device='cuda')
```

이것이 "커널 재작성 없이 적용 가능"하다는 의미이다.

---

## 2. Data Layer: H2O Frame → cuDF

### 2-1. H2O Frame의 한계

H2O Frame은 Java 기반이고 JVM 힙 메모리에 데이터를 저장한다.
GPU 알고리즘(XGBoost GPU 모드)을 쓸 때 다음 경로로 데이터가 이동한다:

```
[H2O Frame]       [Java-C++ 브릿지]       [XGBoost C++]       [GPU]
JVM 힙 메모리  →  JNI 호출 + 직렬화   →  C++ 메모리 복사   →  VRAM 전송
                  ~수 ms/MB              ~수 ms/MB            ~수 ms/MB

총 오버헤드: 대용량 데이터에서 수 초 ~ 수십 초
```

이 오버헤드가 **매 모델 훈련마다** 발생한다.
20개 모델 × 5-fold CV = 100번 훈련이면 → 오버헤드만 수 분.

H2O의 XGBoost GPU 모드가 native XGBoost 대비 **8~11배** 느린 이유가 이것이다.
([szilard/GBM-perf 벤치마크](https://github.com/szilard/GBM-perf) 기준, V100 GPU)

### 2-2. cuDF가 이 문제를 해결하는 방법

```
[cuDF DataFrame]   [cuML / XGBoost GPU]
GPU VRAM        →  직접 접근 (Zero-copy)

데이터 이동: 없음 (0 ms)
```

cuDF DataFrame은 처음부터 GPU VRAM에 있다.
cuML과 XGBoost GPU는 cuDF DataFrame을 직접 읽을 수 있다.
**데이터 복사가 발생하지 않는다.**

이것이 **Zero-copy 원칙**이다.

### 2-3. cuDF의 내부 구조

cuDF는 **Apache Arrow** 포맷을 기반으로 한다.

```
Apache Arrow 컬럼형 메모리 레이아웃:

열 "금액":    [100, 250, 50, 1000, 75, ...]    → GPU VRAM의 연속된 메모리
열 "시간":    [14, 3, 22, 8, 17, ...]           → GPU VRAM의 연속된 메모리
열 "카테고리": [1, 0, 2, 1, 0, ...]              → GPU VRAM의 연속된 메모리
```

**왜 컬럼형(columnar)이 GPU에 유리한가?**

GPU는 **같은 연산을 많은 데이터에 동시에** 적용하는 데 최적화되어 있다 (SIMD).
컬럼형이면 하나의 열에 대한 연산(평균, 합계, 정렬 등)이
**연속된 메모리 위에서 한 번에** 수행된다.

```
행 기반 (pandas):
  [100, 14, 1], [250, 3, 0], [50, 22, 2], ...
  → "금액" 평균 계산: 메모리에서 3칸 간격으로 점프하며 읽음 (느림)

컬럼 기반 (cuDF):
  금액: [100, 250, 50, 1000, 75, ...]
  → "금액" 평균 계산: 연속된 메모리를 한 번에 읽음 (빠름)
```

### 2-4. End-to-End GPU 파이프라인의 의미

```
CPU 기반 파이프라인:
  CSV → pandas (CPU RAM) → 전처리 (CPU) → numpy 변환 → GPU 전송 → 훈련 (GPU)
       → GPU에서 결과 → CPU로 복귀 → pandas로 정리 → 다시 GPU 전송 → 다음 모델
       
GPU End-to-End 파이프라인:
  CSV → cuDF (GPU VRAM) → 전처리 (GPU) → 훈련 (GPU) → 예측 (GPU)
       → cuDF로 정리 (GPU) → 다음 모델 (GPU) → ... → 최종 결과
       
       데이터가 GPU를 떠나지 않음
```

모델이 20개고 5-fold CV를 하면 100번 훈련이다.
매번 CPU↔GPU 전송이 일어나면 오버헤드가 누적되지만,
End-to-End GPU라면 이 오버헤드가 **0**이다.

---

## 3. Algorithm Pool: 전체 알고리즘 GPU 가속

### 3-1. H2O의 현실: XGBoost만 GPU

H2O AutoML에서 GPU를 쓸 수 있는 알고리즘은 **XGBoost 하나뿐**이다.

```
H2O AutoML (GPU 모드 켜도):
  XGBoost  → GPU ✅ (하지만 Java 브릿지 오버헤드)
  GBM      → CPU ❌
  RF       → CPU ❌
  GLM      → CPU ❌
  DNN      → CPU ❌
  XRT      → CPU ❌
```

이 상태에서 AutoML을 돌리면:
- XGBoost는 빠르게 끝남
- 나머지 5개 알고리즘은 CPU에서 느리게 진행
- **전체 파이프라인은 가장 느린 알고리즘의 속도에 지배됨**

### 3-2. GPU 재설계: 전체 알고리즘이 GPU에서 실행

```
GPU 재설계:
  XGBoost GPU (native)           → GPU ✅ (브릿지 없음)
  XGBoost hist+cuda (GBM 대체)   → GPU ✅
  cuML RandomForestClassifier    → GPU ✅
  cuML LogisticRegression        → GPU ✅
  PyTorch MLP (cuDNN)            → GPU ✅
  cuML RF (max_features 조정)    → GPU ⚠️ (완전 XRT 모방 불가)
```

**파이프라인 전체가 GPU에서 돌아간다.**

### 3-3. 각 대체의 구체적 근거

**H2O GBM → XGBoost `tree_method='hist', device='cuda'`**

H2O GBM과 XGBoost는 둘 다 gradient boosting 알고리즘이다.
H2O GBM은 Java로 자체 구현한 것이고, XGBoost는 C++ 네이티브 구현이다.
`tree_method='hist'` + `device='cuda'`는 GPU에서 히스토그램 기반 트리 분할을 수행한다.
(참고: 이전의 `tree_method='gpu_hist'`는 deprecated)

둘의 차이는 구현 디테일이지 알고리즘 원리는 동일하므로,
XGBoost GPU 모드로 H2O GBM의 역할을 완전히 대체할 수 있다.

**H2O XGBoost (Java wrapper) → XGBoost native GPU**

H2O의 XGBoost는 Java에서 JNI를 통해 C++ XGBoost를 호출하는 wrapper이다.
이 wrapper를 제거하고 Python에서 XGBoost를 직접 호출하면:
- JNI 오버헤드 제거
- Java 힙 → C++ → GPU의 이중 데이터 복사 제거
- cuDF DataFrame에서 직접 DMatrix 생성 가능

**H2O DRF → cuML RandomForestClassifier**

cuML의 RF는 GPU에서 트리를 병렬 구축한다.
scikit-learn RF와 동일한 API를 제공하므로 사용법이 같다.

주의점: cuML RF는 `max_depth`에 제한이 있을 수 있다 (기본 최대 16).
H2O의 RF는 깊이 제한이 없는데, 이 차이를 인지하고 설계해야 한다.

**H2O GLM → cuML LogisticRegression/LinearRegression**

GLM(Generalized Linear Model)은 선형 모델의 일반화이다.
- 분류: LogisticRegression
- 회귀: LinearRegression / Ridge / Lasso

cuML은 이 모든 변형을 GPU에서 지원한다.
Regularization (L1, L2, Elastic Net)도 동일하게 지원.

**H2O DNN → PyTorch MLP**

H2O의 DNN은 Java로 구현된 multi-layer perceptron이다.
PyTorch로 동일한 구조의 MLP를 만들면 cuDNN 가속을 받는다.

```
H2O DNN 기본 구조: 입력 → 200 → 200 → 출력
PyTorch 대체:      nn.Linear(n, 200) → ReLU → nn.Linear(200, 200) → ReLU → nn.Linear(200, out)
```

**H2O XRT → cuML RF (split 방식 변경)**

XRT(Extremely Randomized Trees)와 RF의 차이:
- RF: 각 분할에서 **최적의** threshold를 찾음
- XRT: 각 분할에서 **랜덤** threshold를 사용

cuML RF는 과거 `split_algo` 파라미터가 있었으나 현재 버전에서는 삭제되었으며,
XRT 스타일의 랜덤 분할을 직접 지원하지 않는다.
→ XRT는 cuML RF의 `max_features` 조정 + 깊은 트리로 유사한 다양성 효과를 노리거나, H2O의 알고리즘 풀에서 제외하는 것이 현실적이다.

---

## 4. Training Pipeline: GPU 병렬화의 구체적 방법

### 4-1. H2O의 순차 실행 문제

H2O는 모델을 **한 번에 하나씩** 훈련한다.

```
시간 →
[XGB-1]──────[XGB-2]──────[GBM-1]──────[RF-1]──────[GLM-1]──────
              ↑ XGB-1이 끝나야 시작
```

GPU 1개가 있어도, 한 모델이 GPU의 자원을 100% 활용하지 못하는 경우가 많다.
특히 데이터가 작거나, GLM처럼 단순한 모델은 GPU 자원의 10% 정도만 사용한다.

### 4-2. GPU 병렬화 방법 1: 모델 간 병렬 (Model Parallelism)

GPU VRAM이 충분하면 여러 모델을 **동시에** 훈련할 수 있다.

```
GPU 0 (VRAM 24GB):
시간 →
[XGB-1]──────[XGB-3]──────[GBM-2]──────
[XGB-2]──────[GBM-1]──────[RF-1]───────

VRAM 사용: XGB 1개 ~4GB + XGB 1개 ~4GB = 8GB (24GB 중)
나머지 VRAM은 데이터(cuDF)가 사용
```

Dask-CUDA의 `LocalCUDACluster`로 구현:

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster(n_workers=2)  # GPU 0에서 worker 2개
client = Client(cluster)

# 서로 다른 파라미터의 모델을 동시에 훈련
futures = [
    client.submit(train_xgboost, params_1),
    client.submit(train_xgboost, params_2),
]
results = client.gather(futures)
```

### 4-3. GPU 병렬화 방법 2: 멀티 GPU 병렬

GPU가 여러 개 있으면 각 GPU에 다른 모델을 배정한다.

```
GPU 0: [XGB-1]──────[XGB-3]──────[RF-1]──────
GPU 1: [XGB-2]──────[GBM-1]──────[GLM-1]─────
GPU 2: [GBM-2]──────[DNN-1]──────[XRT-1]─────

총 시간: 모델 수 / GPU 수 × 모델당 시간
```

### 4-4. GPU 병렬화 방법 3: Fold 병렬 (Fold Parallelism)

5-fold CV에서 각 fold를 다른 GPU에 배정한다.

```
모델 A에 대한 5-fold CV:

싱글 GPU:
[Fold1]──[Fold2]──[Fold3]──[Fold4]──[Fold5]──  총 5T

멀티 GPU (3개):
GPU 0: [Fold1]──────[Fold4]──────
GPU 1: [Fold2]──────[Fold5]──────              총 ~2T
GPU 2: [Fold3]──────
```

**이론적 속도 향상:** fold 수 / GPU 수

실제로는 데이터 분배, 결과 수집, 동기화 오버헤드가 있어서
이론치의 70~85% 정도 달성이 현실적이다.

### 4-5. Resource-Aware Scheduling

GPU 메모리는 유한하다. 동시에 너무 많은 모델을 돌리면 OOM(Out of Memory)이 발생한다.

**필요한 것:** 각 모델의 예상 GPU 메모리 사용량을 추정하고, 여유 VRAM 내에서만 병렬 실행.

```
예: GPU VRAM = 24GB, 데이터 = 4GB

가용 VRAM: 24 - 4 = 20GB

XGBoost 1개: ~4GB → 동시에 5개까지 가능
cuML RF 1개: ~6GB → 동시에 3개까지 가능
PyTorch MLP 1개: ~2GB → 동시에 10개까지 가능

스케줄러: 현재 가용 VRAM을 추적하며 모델을 배정
```

이것이 팀원 자료에서 언급한 **"Resource-aware scheduling"**이다.

---

## 5. Cross-Validation: GPU 가속의 최대 효과 구간

### 5-1. 왜 CV가 시간의 대부분을 차지하는가

AutoML에서의 시간 분배:

```
전체 시간을 100이라 하면:

단일 모델 훈련 시간:          1 (기준)
5-fold CV:                   5 (5배)
20개 모델 × 5-fold CV:      100 (100배)
Stacked Ensemble 구성:       1 (Meta learner 훈련)
Leaderboard + Explain:       2

→ 전체 시간의 ~95%가 "base model의 CV 훈련"
```

**GPU 가속이 가장 직접적으로 효과를 주는 구간이 바로 이것이다.**

### 5-2. 구체적 속도 향상 추정

벤치마크 기반 추정 (100만 행, 10 feature 기준):

| 알고리즘 | CPU 훈련 시간 | GPU 훈련 시간 | 가속비 |
|----------|--------------|--------------|--------|
| XGBoost | 60초 | 3초 | 20x |
| Random Forest | 45초 | 5초 | 9x |
| GLM | 10초 | 0.5초 | 20x |
| MLP (DNN) | 120초 | 8초 | 15x |

5-fold CV × 20개 모델 = 100번 훈련:

```
CPU 전체: ~100 × 평균 60초 = 약 100분
GPU 전체: ~100 × 평균 4초  = 약 7분

→ 전체 파이프라인 약 14배 가속
```

여기에 fold-level parallelism까지 적용하면 추가 가속.

### 5-3. Level-One Data 생성의 GPU 이점

OOF prediction을 GPU 메모리에서 바로 수집하는 이점:

```
CPU 파이프라인:
  모델A Fold1 예측 → CPU RAM에 저장
  모델A Fold2 예측 → CPU RAM에 저장
  ...
  전체 OOF 수집 → numpy array로 병합
  → GPU 전송 (Meta learner 훈련을 위해)

GPU End-to-End:
  모델A Fold1 예측 → GPU VRAM에 cuDF 컬럼으로 저장
  모델A Fold2 예측 → 같은 cuDF에 append
  ...
  전체 OOF → cuDF DataFrame으로 이미 완성
  → 바로 Meta learner 훈련 (GPU에서, 이동 없음)
```

---

## 6. Stacked Ensemble: GPU에서의 강화 포인트

### 6-1. Meta Learner: cuML GLM

H2O의 non-negative GLM meta learner를 cuML로 대체:

```
H2O:
  h2o.glm(non_negative=True, alpha=0.5)  → Java, CPU

GPU 대체:
  cuml.linear_model.LogisticRegression(
      penalty='elasticnet',
      l1_ratio=0.5,        # alpha에 해당
      C=1.0                # 정규화 강도 (lambda의 역수)
  )
  → GPU에서 실행
```

주의: cuML의 LogisticRegression은 `non_negative` 옵션을 직접 지원하지 않을 수 있다.
이 경우 학습 후 음수 계수를 0으로 클리핑하는 후처리가 필요하다.

```python
meta_model.fit(level_one_data, y_train)
# 음수 가중치 클리핑 (non-negative 제약 모방)
meta_model.coef_ = cupy.maximum(meta_model.coef_, 0)
```

### 6-2. GPU 메모리 관리: 현실적 제약

Level-One Data의 크기: N (샘플 수) × L (모델 수)

```
시나리오 1: N=100만, L=20 (적당한 규모)
  100만 × 20 × 4 bytes (float32) = 80 MB → 문제 없음

시나리오 2: N=100만, L=100 (대규모)
  100만 × 100 × 4 bytes = 400 MB → 아직 괜찮음

시나리오 3: N=1000만, L=100 (극대규모)
  1000만 × 100 × 4 bytes = 4 GB → GPU VRAM 압박
```

**Level-One Data 자체**는 대부분 GPU VRAM에 충분히 들어간다.
문제는 Level-One Data + 원본 데이터 + 모델 파라미터가 **동시에** VRAM에 있어야 한다는 것.

해결 전략 (우선순위순):
1. **Best of Family 사용** → L을 5~6개로 제한 (가장 실용적)
2. **float16 사용** → 메모리 절반으로 감소
3. **배치 처리** → OOF를 모델 그룹별로 나눠 생성
4. **Dask-cuDF 분산** → 멀티 GPU에 데이터 분산

### 6-3. GPU에서만 가능한 확장: 비선형 Meta Learner

H2O의 기본 meta learner는 GLM(선형)이다.
CPU에서는 이것이 합리적이었지만, GPU에서는 더 복잡한 meta learner를 시도할 여유가 있다.

| Meta Learner | 장점 | 단점 | GPU에서의 실용성 |
|---|---|---|---|
| GLM (기본) | 빠름, 안정, 해석 가능 | 비선형 관계 포착 못함 | 높음 (cuML) |
| XGBoost | 비선형 관계 포착 | 과적합 위험 | 높음 (native GPU) |
| MLP (2층) | 유연한 학습 | 튜닝 필요 | 중간 (PyTorch) |
| Stacking of Stacking | 이론적 최고 성능 | 복잡도 급증, 과적합 | 낮음 (비추천) |

**실전 추천:**
기본은 GLM으로 하되, GLM 성능이 포화되면 XGBoost meta learner를 시도.
GPU에서는 XGBoost meta learner 훈련이 수 초면 끝나므로 부담이 없다.

---

## 7. 재현성(Reproducibility) 문제

### 7-1. GPU 연산의 비결정성

GPU는 부동소수점 연산의 순서가 매번 달라질 수 있다.

```
CPU: 0.1 + 0.2 + 0.3 = 항상 같은 결과

GPU: 0.1 + 0.2 + 0.3
     → 실행 1: 스레드A가 (0.1+0.2)를 먼저 계산 → 0.30000000000000004 + 0.3
     → 실행 2: 스레드B가 (0.2+0.3)를 먼저 계산 → 0.1 + 0.5
     → 부동소수점 특성상 미세하게 다른 결과 가능
```

이것이 누적되면 같은 데이터 + 같은 파라미터인데 결과가 달라질 수 있다.

### 7-2. Deterministic 모드

XGBoost: `deterministic_histogram=True` 설정으로 재현 가능
cuML: `random_state` 파라미터로 시드 고정
PyTorch: `torch.use_deterministic_algorithms(True)` 설정

**트레이드오프:** Deterministic 모드는 GPU 활용 효율이 떨어져서 10~30% 느려질 수 있다.

연구/실험 시에는 deterministic 모드 사용, 프로덕션에서는 성능 우선.

---

## 8. 전체 비교 요약

### H2O AutoML vs GPU 재설계의 본질적 차이

| 측면 | H2O AutoML | GPU 재설계 |
|------|------------|-----------|
| 데이터 위치 | CPU RAM (JVM 힙) | GPU VRAM |
| GPU 가속 범위 | XGBoost만 | **전체 알고리즘** |
| 모델 훈련 방식 | 순차 (하나씩) | **병렬 (동시 다수)** |
| CV fold 처리 | 순차 | **fold-level 병렬** |
| 데이터 이동 | CPU↔GPU 빈번 | **Zero-copy (이동 없음)** |
| HPO 탐색 | 순차 Random Search | **병렬 Random Search** |
| Meta Learner | CPU GLM | **GPU GLM + 확장 가능** |
| SHAP 계산 | CPU TreeSHAP | **GPUTreeSHAP** |

### 바뀌지 않는 것 (H2O에서 그대로 가져가는 기법)

| 기법 | 이유 |
|------|------|
| 5-fold CV + OOF | 데이터 누수 방지의 표준 방법, GPU/CPU 무관 |
| Two-type Ensemble | 설계 전략이지 구현이 아님 |
| 훈련 순서 (baseline → diversity → random) | 전략적으로 유효, GPU에서는 더 빠르게 실행될 뿐 |
| 시간 기반 제어 (max_runtime_secs) | AutoML의 핵심 UX, 동일하게 적용 |
| Non-negative GLM meta learner | 안정성의 핵심, GPU GLM으로 동일 구현 |
| Sparse Ensemble (Lasso) | Regularization 기법, cuML에서 동일 지원 |
| Leaderboard 구조 | 분석/시각화 레이어, 독립적 |

---

## 9. 발표 시 예상 질문과 답변

**Q: "그러면 결국 H2O 안 쓰고 cuML로 새로 짜는 건데, H2O를 분석하는 이유가 뭔가요?"**

A: H2O가 10년간 검증한 **설계 전략**(훈련 순서, 앙상블 구조, OOF 방식, 시간 분배)을
그대로 가져가기 위해서이다. 이것을 처음부터 설계하려면 수많은 실험과 시행착오가 필요한데,
H2O가 이미 답을 내놓았다. 우리는 "무엇을 할지"는 H2O에서 배우고,
"어떻게 할지"만 GPU 라이브러리로 바꾸는 것이다.

**Q: "XGBoost만 GPU로 써도 충분하지 않나요?"**

A: 충분하지 않다. Stacked Ensemble의 성능은 model diversity에 달려 있는데,
XGBoost만 GPU로 빠르게 돌려도 RF, GLM, DNN이 CPU에서 병목이 되면
전체 파이프라인이 느려진다. **전체 알고리즘이 GPU에서 돌아야** H2O의
"빠른 인프라로 많이 탐색"이라는 철학이 실현된다.

**Q: "GPU 메모리가 부족하면 어떻게 하나요?"**

A: 세 가지 전략이 있다.
1. Best of Family Ensemble으로 모델 수를 5~6개로 제한 (가장 실용적)
2. Dask-cuDF로 멀티 GPU에 데이터 분산
3. 배치 처리로 한 번에 모든 데이터를 올리지 않음
실제로 대부분의 tabular 데이터는 16~24GB GPU에 충분히 들어간다.

**Q: "실제 성능 향상이 얼마나 되나요?"**

A: 벤치마크 사례를 보면,
- TPOT + RAPIDS: CPU 8시간 대비 GPU 1시간에 더 높은 정확도
- AutoGluon + RAPIDS: 훈련 40배, 추론 10배 가속
- cuML Stacking: 훈련 35배, 추론 350배 가속
전체 AutoML 파이프라인으로 보면 10~40배 가속이 현실적 기대치이다.
