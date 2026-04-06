# RAPIDS 기반 GPU AutoML 구현 전략

> 이전 문서(01~03)에서 "H2O 기법을 GPU로 옮길 수 있다"는 것을 확인했다.
> 이 문서는 그 다음 스텝: **RAPIDS 생태계를 활용해서 구체적으로 어떻게 구현하는가**에 대한 전략이다.

---

## 우리 팀의 위치와 목적

| 질문 | 답 |
|------|---|
| 우리가 만드는 것은? | **Memory-Aware GPU AutoML Framework** (RAPIDS 기반) |
| H2O에서 가져가는 것은? | Stacking/HPO의 **설계 전략** (방법론) |
| RAPIDS에서 활용하는 것은? | cuDF, cuML, Dask-CUDA, **rmm** (인프라) |
| 우리의 고유 기여는? | **GPU 메모리 최적화** 관점의 AutoML 스케줄링 |

H2O를 단순히 CPU→GPU로 포팅하는 것이 목적이 **아니다.**
RAPIDS LAB으로서 실무에서 활용하는 **메모리 최적화 역량**을 GPU AutoML이라는 도메인에 적용하고,
그 과정에서 메모리 최적화가 내재된 프레임워크를 설계하며 연구를 진행하는 것이 목적이다.

Dask-CUDA를 사용하는 이유도 단순히 "병렬화가 빨라서"가 아니라,
Dask-CUDA + rmm 조합이 **GPU 메모리 자원을 인지하고 관리하는 스케줄링 인프라**를 제공하기 때문이다.

---

## 1. RAPIDS 생태계에서 우리가 활용하는 것

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAPIDS 생태계                                │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  cuDF        │  cuML        │  Dask-CUDA   │  cuGraph / cuSpatial │
│  GPU 데이터  │  GPU ML      │  GPU 분산    │  (우리 범위 밖)      │
│  프레임      │  알고리즘    │  컴퓨팅      │                      │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│  rmm (RAPIDS Memory Manager) — GPU 메모리 풀 관리               │
├────────────────────────────────────────────────────────────────┤
│  CUDA / cuDNN / NCCL — NVIDIA GPU 하드웨어 레이어               │
└────────────────────────────────────────────────────────────────┘
```

| 컴포넌트 | 역할 | 우리 시스템에서의 위치 |
|----------|------|----------------------|
| **cuDF** | GPU DataFrame | Data Layer — 데이터 GPU 상주 |
| **Dask-cuDF** | 멀티 GPU 분산 DataFrame | VRAM 초과 데이터 처리 |
| **cuML** | GPU ML 알고리즘 (RF, GLM 등) | Algorithm Pool |
| **Dask-CUDA** | GPU worker 관리 + 스케줄링 | **메모리 인지 병렬 훈련의 핵심 인프라** |
| **rmm** | GPU 메모리 풀 관리 | **메모리 최적화 연구의 핵심 도구** |
| **XGBoost + Dask** | 분산 GPU XGBoost | GBM 계열 훈련 |

### Dask-CUDA + rmm 조합이 핵심인 이유

Dask-CUDA는 단순한 병렬화 프레임워크가 아니다.
GPU 환경에서 **메모리 자원을 인지하는 스케줄링**을 제공한다:

1. **`device_memory_limit`**: worker별 VRAM 사용 상한 → 초과 시 spill
2. **`rmm_pool_size`**: 메모리 풀 사전 할당 → 할당/해제 오버헤드 제거
3. **worker별 자원 추적**: 어떤 worker에 VRAM 여유가 있는지 실시간 파악
4. **spill-to-host**: VRAM 부족 시 자동으로 CPU RAM으로 데이터 이동

이 기능들이 우리의 메모리 최적화 연구와 직접적으로 연결된다.

---

## 2. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                 Memory-Aware GPU AutoML System                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Orchestrator (Python)                                     │  │
│  │  - 훈련 순서 관리 (H2O 전략 계승)                          │  │
│  │  - Leaderboard 관리                                        │  │
│  │  - 시간 기반 제어 (max_runtime_secs)                       │  │
│  │  - ★ 메모리 예산 기반 모델 스케줄링                        │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────────┐  │
│  │  Dask-CUDA Cluster + rmm                                   │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │  │
│  │  │ Worker 0 │ │ Worker 1 │ │ Worker 2 │  ...              │  │
│  │  │ (GPU 0)  │ │ (GPU 1)  │ │ (GPU 0)  │                   │  │
│  │  │ rmm pool │ │ rmm pool │ │ rmm pool │                   │  │
│  │  └──────────┘ └──────────┘ └──────────┘                   │  │
│  │                                                            │  │
│  │  - device_memory_limit로 worker별 VRAM 상한 관리           │  │
│  │  - rmm logging으로 메모리 사용 패턴 추적                   │  │
│  │  - spill-to-host로 OOM 안전장치 제공                       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Data Layer                                                │  │
│  │  - cuDF DataFrame (단일 GPU)                               │  │
│  │  - Dask-cuDF DataFrame (멀티 GPU 분산)                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 파이프라인 단계별 RAPIDS 활용

### 3-1. Dask-CUDA 클러스터 초기화 + rmm 설정

```python
import cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf

cluster = LocalCUDACluster(
    n_workers=2,                    # GPU 2개 사용
    device_memory_limit="12GB",     # worker당 VRAM 제한 (메모리 관리의 핵심)
    rmm_pool_size="10GB",           # rmm 메모리 풀 사전 할당
)
client = Client(cluster)
```

**`device_memory_limit`**: worker가 이 이상 메모리를 쓰면 spill(디스크로 밀어냄)  
**`rmm_pool_size`**: GPU 메모리를 미리 풀로 잡아두면 할당/해제 오버헤드가 사라짐

### 3-2. 데이터 로드

```python
# 단일 GPU에 들어가는 크기면 cuDF
train = cudf.read_csv("train.csv")

# 단일 GPU를 초과하면 Dask-cuDF로 분산
train_distributed = dask_cudf.read_csv("train.csv", chunksize="1GB")
```

### 3-3. K-Fold CV 분할 (GPU에서)

```python
import cupy as cp

def create_folds(df, n_folds=5, seed=42):
    n = len(df)
    cp.random.seed(seed)
    fold_ids = cp.random.randint(0, n_folds, size=n)
    df['fold'] = fold_ids
    return df

train = create_folds(train, n_folds=5)
```

### 3-4. 병렬 모델 훈련

#### 모델 간 병렬 — 서로 다른 모델을 동시에 실행

```python
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLR
import xgboost as xgb

def train_cuml_rf(X_train, y_train, X_valid, params):
    model = cuRF(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_valid)[:, 1]
    return preds

def train_xgboost_gpu(X_train, y_train, X_valid, params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid)
    params['tree_method'] = 'hist'
    params['device'] = 'cuda'
    model = xgb.train(params, dtrain)
    preds = model.predict(dvalid)
    return preds

# Dask가 가용 GPU worker에 자동 배정
futures = [
    client.submit(train_cuml_rf, X_tr, y_tr, X_val, rf_params),
    client.submit(train_xgboost_gpu, X_tr, y_tr, X_val, xgb_params),
    client.submit(train_cuml_lr, X_tr, y_tr, X_val, lr_params),
]
results = client.gather(futures)
```

#### Fold 병렬 — 같은 모델의 다른 fold를 동시에 실행

```python
def train_single_fold(model_fn, params, train_df, fold_id):
    valid_mask = train_df['fold'] == fold_id
    X_train = train_df[~valid_mask].drop(['target', 'fold'])
    y_train = train_df[~valid_mask]['target']
    X_valid = train_df[valid_mask].drop(['target', 'fold'])

    model = model_fn(**params)
    model.fit(X_train, y_train)
    oof_preds = model.predict_proba(X_valid)[:, 1]
    return fold_id, oof_preds

# 5-fold를 동시에 Dask worker에 제출
fold_futures = [
    client.submit(train_single_fold, cuRF, rf_params, train, fold_id=i)
    for i in range(5)
]
fold_results = client.gather(fold_futures)
```

#### XGBoost Dask 분산 훈련 — 멀티 GPU로 하나의 모델 훈련

```python
from xgboost import dask as dxgb

dtrain = dxgb.DaskDMatrix(client, X_train_dask, y_train_dask)

output = dxgb.train(
    client,
    params={'tree_method': 'hist', 'device': 'cuda', 'objective': 'binary:logistic'},
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dvalid, 'valid')],
    early_stopping_rounds=10,
)
```

### 3-5. OOF 수집 → Level-One Data 생성

```python
import cudf
import cupy as cp

def collect_oof_predictions(fold_results, n_samples):
    oof = cp.zeros(n_samples)
    for fold_id, preds in fold_results:
        fold_mask = (train['fold'] == fold_id).values
        oof[fold_mask] = preds
    return oof

# Level-One Data를 cuDF DataFrame으로 구성 (GPU 메모리에서)
level_one = cudf.DataFrame({
    'rf_pred': collect_oof_predictions(rf_fold_results, len(train)),
    'xgb_pred': collect_oof_predictions(xgb_fold_results, len(train)),
    'glm_pred': collect_oof_predictions(glm_fold_results, len(train)),
})
```

### 3-6. Stacked Ensemble — Meta Learner 훈련

```python
from cuml.linear_model import LogisticRegression

meta_model = LogisticRegression(
    penalty='l1',           # Lasso → sparse ensemble 유도
    C=1.0,
    max_iter=1000,
)
meta_model.fit(level_one, y_train)

# Non-negative 제약 (H2O 방식 모방)
import cupy as cp
meta_model.coef_ = cp.maximum(meta_model.coef_, 0)
```

### 3-7. Leaderboard 생성

```python
import pandas as pd

leaderboard = pd.DataFrame([
    {'model': 'StackedEnsemble_AllModels', 'auc': 0.9823, 'train_time_sec': 45.2, 'peak_vram_gb': 8.3},
    {'model': 'StackedEnsemble_BestOfFamily', 'auc': 0.9819, 'train_time_sec': 12.1, 'peak_vram_gb': 3.1},
    {'model': 'XGBoost_gpu_1', 'auc': 0.9801, 'train_time_sec': 8.3, 'peak_vram_gb': 4.2},
    {'model': 'cuML_RF_1', 'auc': 0.9795, 'train_time_sec': 5.1, 'peak_vram_gb': 5.8},
    {'model': 'cuML_GLM_1', 'auc': 0.9650, 'train_time_sec': 0.8, 'peak_vram_gb': 0.9},
]).sort_values('auc', ascending=False)

leaderboard['rank'] = range(1, len(leaderboard) + 1)
```

기존 H2O Leaderboard에 없는 **`peak_vram_gb`** 컬럼을 추가한다.
이것이 메모리 최적화 연구에서 중요한 메트릭이 된다.

---

## 4. 구현 로드맵

### Phase 1: 기반 구축 (MVP)

| 순서 | 작업 | RAPIDS 컴포넌트 | 목표 |
|------|------|-----------------|------|
| 1-1 | cuDF로 데이터 로드 + 전처리 | cuDF | E2E GPU 데이터 파이프라인 |
| 1-2 | cuML RF + XGBoost GPU 단일 모델 훈련 | cuML, XGBoost | GPU 모델 훈련 확인 |
| 1-3 | 5-fold CV + OOF 생성 (순차, 단일 GPU) | cuML, cuDF | Stacking 기초 동작 확인 |
| 1-4 | cuML GLM Meta Learner + 앙상블 예측 | cuML | Stacked Ensemble 완성 |
| 1-5 | Leaderboard 출력 | pandas | 결과 확인 |

**Phase 1 완료 = "GPU에서 H2O와 동일한 Stacking이 돌아간다"**

### Phase 2: Dask-CUDA 병렬화 + 메모리 인프라

| 순서 | 작업 | RAPIDS 컴포넌트 | 목표 |
|------|------|-----------------|------|
| 2-1 | LocalCUDACluster + rmm 풀 세팅 | Dask-CUDA, rmm | GPU 클러스터 인프라 |
| 2-2 | Fold 병렬화 (5-fold 동시 실행) | Dask-CUDA | CV 시간 단축 |
| 2-3 | 모델 간 병렬화 (RF, XGB, GLM 동시) | Dask-CUDA | 전체 훈련 시간 단축 |
| 2-4 | rmm logging으로 메모리 프로파일링 | rmm | 메모리 사용 패턴 수집 |

**Phase 2 완료 = "병렬 훈련 + 메모리 사용 데이터 확보"**

### Phase 3: 메모리 최적화 연구 (→ 05번 문서에서 상세)

| 순서 | 작업 | 목표 |
|------|------|------|
| 3-1 | 메모리 프로파일링 분석 | 병목 구간 식별 |
| 3-2 | Memory-Aware Scheduling 설계 | 메모리 예산 기반 모델 배정 |
| 3-3 | Stacking 메모리 최적화 | Level-One Data 관리 전략 |
| 3-4 | 벤치마크 + 비교 실험 | 최적화 효과 정량 검증 |

**Phase 3 완료 = "메모리 최적화가 내재된 GPU AutoML"**

### Phase 4: H2O 전략 고도화 + 확장

| 순서 | 작업 | 목표 |
|------|------|------|
| 4-1 | 훈련 순서 전략 (baseline → diversity → random) | H2O Phase 설계 재현 |
| 4-2 | 시간 기반 제어 (max_runtime_secs) | 시간 예산 관리 |
| 4-3 | Two-type Ensemble (All Models + Best of Family) | 앙상블 전략 완성 |
| 4-4 | Explainability (GPUTreeSHAP) | 모델 설명 기능 |

---

## 5. 기술 스택 요약

```
┌────────────────────────────────────────────────┐
│  Memory-Aware GPU AutoML 기술 스택              │
├────────────────────────────────────────────────┤
│  Python API (sklearn 호환 인터페이스)           │
├────────────────────────────────────────────────┤
│  Orchestrator                                   │
│  - H2O 전략 로직 (훈련 순서/시간 제어)         │
│  - ★ Memory-Aware Scheduling                   │
├────────────────────────────────────────────────┤
│  Dask-CUDA Cluster                              │
│  - GPU worker 관리                              │
│  - 메모리 인지 Task 스케줄링                    │
│  - spill-to-host 안전장치                       │
├────────────────────────────────────────────────┤
│  cuML          │  XGBoost GPU  │  PyTorch MLP  │
│  RF, GLM, ...  │  GBM 계열    │  DNN          │
├────────────────┴───────────────┴───────────────┤
│  cuDF / Dask-cuDF (GPU DataFrame)              │
├────────────────────────────────────────────────┤
│  rmm (RAPIDS Memory Manager)                   │
│  - 메모리 풀 관리                               │
│  - 할당/해제 프로파일링                         │
├────────────────────────────────────────────────┤
│  CUDA / cuDNN / NCCL                           │
└────────────────────────────────────────────────┘

우리가 직접 구현: Orchestrator + Memory-Aware Scheduling
RAPIDS가 제공:    나머지 전부
```
