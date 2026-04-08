# Ray 프로세스 수준 페이징: 블랙박스 GPU 메모리 관리의 현실적 해법

> **요약**: cuML, XGBoost 등 블랙박스 GPU 라이브러리는 내부 CUDA 할당을 가로챌 수 없어
> vLLM 스타일의 바이트 수준 페이징이 불가능하다.
> Ray는 **프로세스 단위 격리**와 **Object Store 자동 Swap**으로
> 이 문제를 우회하며, AutoML 파이프라인에 가장 현실적인 GPU 메모리 관리 계층을 제공한다.

---

## 목차

1. [문제 정의: 블랙박스 라이브러리의 한계](#1-문제-정의-블랙박스-라이브러리의-한계)
2. [Ray 프레임워크 개요](#2-ray-프레임워크-개요)
3. [Actor/Task 격리 — 프로세스 단위 메모리 관리](#3-actortask-격리--프로세스-단위-메모리-관리)
4. [Fractional GPU — 하나의 GPU를 여러 Actor가 공유](#4-fractional-gpu--하나의-gpu를-여러-actor가-공유)
5. [Object Store — GPU <-> CPU 간 자동 Swap](#5-object-store--gpu--cpu-간-자동-swap)
6. [Ray Data — 스트리밍 블록 처리와 백프레셔](#6-ray-data--스트리밍-블록-처리와-백프레셔)
7. [Placement Groups — 리소스 배치 전략](#7-placement-groups--리소스-배치-전략)
8. [종합 아키텍처: Ray 기반 PagedAutoML 설계](#8-종합-아키텍처-ray-기반-pagedautoml-설계)
9. [vLLM 페이징 vs Ray 프로세스 페이징 비교](#9-vllm-페이징-vs-ray-프로세스-페이징-비교)
10. [References](#10-references)

---

## 1. 문제 정의: 블랙박스 라이브러리의 한계

### 1.1 현재 PagedAutoML의 근본적 제약

현재 프로젝트의 `paged_manager.py`는 vLLM의 PagedAttention에서 영감을 받아
블록 단위 GPU 메모리 관리를 시도하였다. 그러나 핵심적인 차이가 존재한다.

```
vLLM의 경우:
  ┌─────────────────────────────────────────────┐
  │  Attention 커널을 직접 작성                    │
  │  → 비연속 물리 블록을 Page Table로 참조         │
  │  → 연산 도중에도 메모리 재배치 가능              │
  └─────────────────────────────────────────────┘

PagedAutoML의 경우:
  ┌─────────────────────────────────────────────┐
  │  cuML RF, XGBoost = 블랙박스 라이브러리         │
  │  → 내부 CUDA malloc을 가로챌 수 없음            │
  │  → 훈련 도중 메모리 재배치 불가능                │
  │  → "연산 사이"에만 관리 가능 (coarse-grained)    │
  └─────────────────────────────────────────────┘
```

### 1.2 구체적 실패 시나리오

```
[시나리오] Higgs 5M rows, 8GB VRAM GPU

1. rmm pool이 4GB를 선점
2. cuML RF 훈련 시작 → 내부적으로 ~ 3GB 추가 할당 시도
3. 남은 VRAM 부족 → CUDA OOM
4. Memory-Aware Scheduler가 모든 모델을 skip
5. AutoML 전체가 실패
```

**문제의 핵심**: cuML이 내부적으로 `cudaMalloc`을 호출할 때, 외부에서 이를 가로채거나
중간에 다른 블록을 evict할 방법이 없다. rmm의 `device_memory_resource`를 커스텀해도
cuML이 자체적으로 생성하는 임시 버퍼까지 제어하기는 극히 어렵다.

### 1.3 발상의 전환: 바이트에서 프로세스로

```
바이트 수준 제어가 불가능하다면,
프로세스 수준에서 VRAM을 격리하고 관리하면 된다.

  Before (불가능):                   After (가능):
  ┌──────────────┐                 ┌──────────────┐
  │ GPU VRAM     │                 │ GPU VRAM     │
  │ ┌──┬──┬──┐  │                 │ ┌──────────┐ │
  │ │B1│B2│B3│  │  ← 블록 단위     │ │ Process A│ │  ← 프로세스 단위
  │ └──┴──┴──┘  │    관리 시도      │ ├──────────┤ │    격리
  │ cuML이 내부  │    (실패)        │ │ Process B│ │    (성공)
  │ 할당을 가로챌 │                 │ └──────────┘ │
  │ 수 없음      │                 │ CUDA context │
  └──────────────┘                 │ 가 분리됨     │
                                   └──────────────┘
```

**Ray**는 바로 이 "프로세스 수준 격리"를 체계적으로 제공하는 프레임워크이다.

---

## 2. Ray 프레임워크 개요

### 2.1 Ray란 무엇인가

Ray는 UC Berkeley의 RISELab에서 탄생한 분산 컴퓨팅 프레임워크로,
Python 함수와 클래스를 분산 태스크(Task)와 액터(Actor)로 변환하여
클러스터 전체에서 병렬 실행할 수 있게 한다.

### 2.2 핵심 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      Ray Cluster                            │
│                                                             │
│  ┌──────────── Head Node ──────────────┐                    │
│  │                                     │                    │
│  │   ┌─────────────────────┐           │                    │
│  │   │  Global Control     │           │                    │
│  │   │  Service (GCS)      │           │                    │
│  │   │  ┌───────────────┐  │           │                    │
│  │   │  │ Cluster 메타   │  │           │                    │
│  │   │  │ Actor 위치     │  │           │                    │
│  │   │  │ Resource 상태  │  │           │                    │
│  │   │  └───────────────┘  │           │                    │
│  │   └────────┬────────────┘           │                    │
│  │            │ gRPC                   │                    │
│  │   ┌────────▼────────────┐           │                    │
│  │   │     Raylet          │           │                    │
│  │   │  ┌───────────────┐  │           │                    │
│  │   │  │ Scheduler     │  │           │                    │
│  │   │  │ Resource Mgr  │  │           │                    │
│  │   │  └───────┬───────┘  │           │                    │
│  │   │  ┌───────▼───────┐  │           │                    │
│  │   │  │ Object Store  │  │  ← /dev/shm (shared memory)   │
│  │   │  │ (Plasma)      │  │                                │
│  │   │  └───────────────┘  │           │                    │
│  │   └─────────────────────┘           │                    │
│  │                                     │                    │
│  │   ┌────────┐ ┌────────┐ ┌────────┐ │                    │
│  │   │Worker 0│ │Worker 1│ │Worker 2│ │                    │
│  │   │(Task)  │ │(Actor) │ │(Actor) │ │                    │
│  │   └────────┘ └────────┘ └────────┘ │                    │
│  └─────────────────────────────────────┘                    │
│                                                             │
│  ┌──────────── Worker Node ────────────┐                    │
│  │   ┌─────────────────────┐           │                    │
│  │   │     Raylet          │           │                    │
│  │   │  ┌───────────────┐  │           │                    │
│  │   │  │ Scheduler     │  │           │                    │
│  │   │  │ Resource Mgr  │  │           │                    │
│  │   │  └───────┬───────┘  │           │                    │
│  │   │  ┌───────▼───────┐  │           │                    │
│  │   │  │ Object Store  │  │           │                    │
│  │   │  │ (Plasma)      │  │           │                    │
│  │   │  └───────────────┘  │           │                    │
│  │   └─────────────────────┘           │                    │
│  │                                     │                    │
│  │   ┌────────┐ ┌────────┐            │                    │
│  │   │Worker 3│ │Worker 4│            │                    │
│  │   │(Actor) │ │(Task)  │            │                    │
│  │   └────────┘ └────────┘            │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 핵심 컴포넌트 상세

#### Global Control Service (GCS)

GCS는 Ray의 중앙 집중식 메타데이터 저장소이자 제어 평면(control plane)이다.

| 항목 | 설명 |
|:-----|:-----|
| 역할 | 클러스터 수준 상태 관리, 분산 작업 조율 |
| 저장 내용 | 시스템 메타데이터, Actor 위치, 리소스 상태 |
| 통신 | Raylet/Worker와 gRPC로 통신 |
| 스토리지 백엔드 | 인메모리 (기본) 또는 Redis (레거시) |
| 위치 | Head 노드에 위치 (단일 장애점) |

#### Raylet (노드당 데몬)

각 노드에서 실행되는 데몬으로, 로컬 리소스 관리와 태스크 스케줄링을 담당한다.

- **스케줄링**: 하이브리드 방식 — 부하 분산 vs 데이터 지역성 균형
- **리소스 인식**: CPU, GPU, 메모리 수량을 자동 감지
- **핫스팟 방지**: 무작위 top-K 선택으로 특정 노드 과부하 예방
- **Spillback**: 로컬에서 실행 불가능한 태스크를 다른 노드로 전달

#### Object Store (Plasma)

Apache Arrow Plasma 기반의 공유 메모리 객체 저장소이다.

| 설정 | 기본값 |
|:-----|:-------|
| 용량 | 가용 메모리의 30% 또는 최대 200GB |
| 최소 크기 | 75MB |
| 위치 | `/dev/shm` (Linux 공유 메모리) |
| 설정 방법 | `--object-store-memory` 또는 `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES` |

#### Worker 프로세스

실제 연산을 수행하는 프로세스로, 두 가지 유형이 있다.

- **Stateless Worker**: 임의의 `@ray.remote` 함수를 실행
- **Actor Worker**: 특정 `@ray.remote` 클래스의 메서드만 실행 (상태 유지)
- 모든 Worker는 내부에 **CoreWorker** 인스턴스를 포함
- 로컬 Raylet과 IPC 통신, 다른 Worker와는 gRPC 통신

---

## 3. Actor/Task 격리 — 프로세스 단위 메모리 관리

### 3.1 CUDA_VISIBLE_DEVICES 자동 관리

Ray는 GPU를 요청하는 태스크나 액터를 스케줄링할 때
**자동으로 `CUDA_VISIBLE_DEVICES` 환경변수를 설정**한다.
이를 통해 각 프로세스가 자신에게 할당된 GPU만 볼 수 있도록 격리한다.

```python
import ray

ray.init()

@ray.remote(num_gpus=1)
class CuMLTrainer:
    """각 Actor는 독립된 프로세스 + 독립된 CUDA context를 갖는다."""

    def __init__(self, model_type: str):
        import cuml
        import os
        # Ray가 자동으로 CUDA_VISIBLE_DEVICES를 설정
        print(f"Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Ray assigned GPUs: {ray.get_gpu_ids()}")
        self.model_type = model_type

    def train(self, X, y):
        if self.model_type == "rf":
            from cuml.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100)
        elif self.model_type == "xgb":
            import xgboost as xgb
            model = xgb.XGBClassifier(tree_method="gpu_hist")

        model.fit(X, y)
        return model

# 2-GPU 시스템: 각 Actor가 별도의 GPU에서 실행
trainer_rf  = CuMLTrainer.remote(model_type="rf")   # GPU 0
trainer_xgb = CuMLTrainer.remote(model_type="xgb")  # GPU 1

# 병렬 훈련 — 각각 독립된 CUDA context
futures = [
    trainer_rf.train.remote(X_ref, y_ref),
    trainer_xgb.train.remote(X_ref, y_ref),
]
models = ray.get(futures)
```

### 3.2 프로세스 격리의 메커니즘

```
┌─────────────── 물리 GPU 0 ────────────────┐
│                                            │
│  ┌─── Actor Process (PID 1234) ─────────┐ │
│  │  CUDA_VISIBLE_DEVICES="0"            │ │
│  │  ┌──────────────────────────┐        │ │
│  │  │ CUDA Context (독립)      │        │ │
│  │  │  - cuML RF 할당          │        │ │
│  │  │  - 내부 임시 버퍼         │        │ │
│  │  │  - rmm pool (격리됨)     │        │ │
│  │  └──────────────────────────┘        │ │
│  └──────────────────────────────────────┘ │
│                                            │
└────────────────────────────────────────────┘

┌─────────────── 물리 GPU 1 ────────────────┐
│                                            │
│  ┌─── Actor Process (PID 5678) ─────────┐ │
│  │  CUDA_VISIBLE_DEVICES="1"            │ │
│  │  ┌──────────────────────────┐        │ │
│  │  │ CUDA Context (독립)      │        │ │
│  │  │  - XGBoost 할당          │        │ │
│  │  │  - 내부 임시 버퍼         │        │ │
│  │  │  - rmm pool (격리됨)     │        │ │
│  │  └──────────────────────────┘        │ │
│  └──────────────────────────────────────┘ │
│                                            │
└────────────────────────────────────────────┘
```

핵심 원리:

1. **프로세스 격리**: 각 Actor는 독립된 OS 프로세스로 실행된다
2. **CUDA Context 분리**: 프로세스가 다르면 CUDA context도 별도로 생성된다
3. **환경변수 제한**: `CUDA_VISIBLE_DEVICES`로 접근 가능한 GPU를 물리적으로 제한한다
4. **블랙박스 호환**: cuML/XGBoost가 내부적으로 `cudaMalloc`을 호출해도,
   해당 프로세스에 할당된 GPU 범위 내에서만 동작한다

### 3.3 Task vs Actor 선택 기준

| 특성 | Task (`@ray.remote` 함수) | Actor (`@ray.remote` 클래스) |
|:-----|:--------------------------|:-----------------------------|
| 상태 | Stateless (매번 새 프로세스 가능) | Stateful (프로세스 유지) |
| CUDA 오버헤드 | 높음 (매번 context 초기화) | 낮음 (context 재사용) |
| GPU 메모리 | 매번 할당/해제 | 프로세스 수명 동안 유지 |
| 적합 용도 | 단일 추론, 전처리 | 모델 훈련, 반복 학습 |
| AutoML에서 | 데이터 분할, 결과 수집 | 각 알고리즘 훈련기 |

AutoML 파이프라인에서는 **모델 훈련기를 Actor로, 데이터 전처리를 Task로** 분리하는 것이 최적이다.

```python
# Task: 상태 불필요, 짧은 실행 시간
@ray.remote
def preprocess(data_ref):
    """CPU에서 데이터 전처리 (GPU 불필요)"""
    import cudf
    df = cudf.DataFrame(data_ref)
    return df.dropna().reset_index(drop=True)

# Actor: 상태 유지, GPU context 재사용
@ray.remote(num_gpus=1)
class ModelTrainer:
    """GPU Actor — CUDA context를 유지하며 반복 훈련"""

    def __init__(self):
        import rmm
        # Actor 생성 시 한 번만 rmm pool 초기화
        rmm.reinitialize(pool_allocator=True, initial_pool_size=2 * 1024**3)

    def train_rf(self, X, y, params):
        from cuml.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
        model.fit(X, y)
        return model

    def train_xgb(self, X, y, params):
        import xgboost as xgb
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(params, dtrain)
        return model

    def cleanup(self):
        """명시적 VRAM 해제"""
        import gc
        import cupy as cp
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
```

---

## 4. Fractional GPU — 하나의 GPU를 여러 Actor가 공유

### 4.1 Fractional GPU란

Ray는 `num_gpus`에 소수 값을 지정하여 **하나의 물리 GPU를 여러 Actor/Task가 공유**할 수 있게 한다.
이는 모델 훈련이 GPU VRAM을 100% 사용하지 않는 경우에 활용률을 높이는 핵심 기법이다.

```python
# 하나의 GPU를 2개 Actor가 공유
@ray.remote(num_gpus=0.5)
class LightModel:
    """VRAM을 적게 쓰는 모델 — GPU의 절반만 논리적으로 할당"""
    def __init__(self, model_name):
        self.model_name = model_name

    def train(self, X, y):
        if self.model_name == "lr":
            from cuml.linear_model import LogisticRegression
            model = LogisticRegression()
        elif self.model_name == "rf_small":
            from cuml.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, max_depth=8)
        model.fit(X, y)
        return model

# 두 Actor가 동일한 물리 GPU를 공유
actor_lr = LightModel.remote("lr")       # GPU 0의 0.5
actor_rf = LightModel.remote("rf_small") # GPU 0의 나머지 0.5
```

### 4.2 Fractional GPU의 동작 원리

```
┌─────────────── 물리 GPU 0 (8GB VRAM) ────────────────┐
│                                                        │
│  CUDA_VISIBLE_DEVICES="0"  (두 프로세스 모두 동일)       │
│                                                        │
│  ┌─── Actor A (num_gpus=0.5) ────┐                    │
│  │  PID 1234                      │                    │
│  │  CUDA Context A                │                    │
│  │  실제 VRAM 사용: ~ 3GB          │ ← Ray가 보장하는   │
│  └────────────────────────────────┘    것은 "스케줄링   │
│                                        논리"이지,       │
│  ┌─── Actor B (num_gpus=0.5) ────┐    물리적 VRAM      │
│  │  PID 5678                      │    파티션이 아님     │
│  │  CUDA Context B                │                    │
│  │  실제 VRAM 사용: ~ 2GB          │                    │
│  └────────────────────────────────┘                    │
│                                                        │
│  남은 VRAM: ~ 3GB (8 - 3 - 2)                          │
└────────────────────────────────────────────────────────┘
```

**중요한 이해**: `num_gpus=0.5`는 "VRAM의 50%를 하드 파티션"하는 것이 **아니다**.
Ray의 스케줄러가 "이 GPU에 논리적으로 0.5 단위가 남아있으므로 여기에 스케줄링한다"는
**논리적 리소스 관리**이다. 실제 VRAM 사용량은 각 프로세스의 실행 내용에 따라 달라진다.

### 4.3 AutoML에서의 Fractional GPU 전략

```python
import ray

ray.init()

# GPU 리소스를 모델 무거움에 따라 차등 배분
RESOURCE_MAP = {
    "xgboost":           0.5,   # XGBoost GPU — VRAM 사용량 중간
    "random_forest":     0.3,   # cuML RF — 상대적으로 가벼움
    "logistic_regression": 0.2, # cuML LR — 매우 가벼움
}

@ray.remote
class AutoMLOrchestrator:
    def __init__(self, gpu_budget=1.0):
        self.gpu_budget = gpu_budget

    def create_trainer(self, model_type):
        gpu_fraction = RESOURCE_MAP.get(model_type, 0.5)

        @ray.remote(num_gpus=gpu_fraction)
        class DynamicTrainer:
            def train(self, X, y, params):
                # 모델별 훈련 로직
                pass

        return DynamicTrainer.remote()

    def run_pipeline(self, X_ref, y_ref):
        """
        XGBoost(0.5) + LR(0.2) + RF(0.3) = 1.0 GPU
        → 세 모델이 하나의 GPU에서 동시 훈련 가능
        """
        trainers = {}
        for model_type in RESOURCE_MAP:
            trainers[model_type] = self.create_trainer(model_type)

        futures = {
            name: trainer.train.remote(X_ref, y_ref, {})
            for name, trainer in trainers.items()
        }
        return {name: ray.get(f) for name, f in futures.items()}
```

### 4.4 Fractional GPU 제약 사항

| 제약 | 설명 |
|:-----|:-----|
| 논리적 관리만 제공 | VRAM의 물리적 파티션이 아님 — OOM 위험은 여전히 존재 |
| 정수 초과 불가 | `num_gpus=1.5`처럼 1을 초과하는 소수는 불가 |
| CUDA MPS 비권장 | NVIDIA MPS와 병행 시 예측 불가능한 동작 가능 |
| 메모리 초과 보호 없음 | Actor가 할당분 이상의 VRAM을 사용해도 Ray가 막지 않음 |

따라서 Fractional GPU를 사용할 때는 **각 모델의 VRAM 사용량을 사전에 프로파일링**하고,
합계가 물리 VRAM을 초과하지 않도록 `RESOURCE_MAP`을 설계해야 한다.

---

## 5. Object Store — GPU <-> CPU 간 자동 Swap

### 5.1 Object Store의 역할

Ray Object Store는 Apache Arrow Plasma 기반의 공유 메모리 저장소로,
**노드 내 프로세스 간 데이터 공유**와 **자동 디스크 스필링**을 제공한다.
이것이 AutoML에서 "CPU <-> GPU 간 자동 Swap"의 핵심 메커니즘이 된다.

```
┌─────────────────────────────────────────────────────────┐
│                    메모리 계층 구조                        │
│                                                          │
│  ┌─────────┐    zero-copy    ┌──────────────────────┐   │
│  │ Worker 0 │ ◄────────────► │                      │   │
│  └─────────┘                 │   Object Store       │   │
│  ┌─────────┐    zero-copy    │   (/dev/shm)         │   │
│  │ Worker 1 │ ◄────────────► │                      │   │
│  └─────────┘                 │   기본: 가용 RAM의 30% │   │
│  ┌─────────┐    zero-copy    │                      │   │
│  │ Worker 2 │ ◄────────────► │                      │   │
│  └─────────┘                 └──────────┬───────────┘   │
│                                         │                │
│                               자동 spill │ (용량 초과 시)  │
│                                         ▼                │
│                              ┌──────────────────────┐   │
│                              │   Local Disk          │   │
│                              │   /tmp/ray/session_*  │   │
│                              │   /spill/             │   │
│                              └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Zero-Copy 읽기

Object Store의 핵심 장점은 **같은 노드의 Worker 간 데이터 복사 없이 공유**할 수 있다는 것이다.

```python
import ray
import numpy as np

ray.init()

# 대용량 데이터를 Object Store에 저장
X_train = np.random.randn(1_000_000, 100).astype(np.float32)  # ~ 400MB
X_ref = ray.put(X_train)  # Object Store에 저장 → ObjectRef 반환

@ray.remote(num_gpus=0.5)
class TrainerA:
    def train(self, X_ref):
        # X_ref에서 데이터를 가져올 때 같은 노드면 zero-copy
        X = ray.get(X_ref)  # 메모리 복사 없이 numpy 배열 참조
        import cudf
        gdf = cudf.DataFrame(X)  # GPU로 전송
        # ... 훈련 ...
        return model

@ray.remote(num_gpus=0.5)
class TrainerB:
    def train(self, X_ref):
        # 동일한 ObjectRef → 동일한 공유 메모리 참조 (zero-copy)
        X = ray.get(X_ref)  # 추가 메모리 할당 없음
        import cudf
        gdf = cudf.DataFrame(X)
        # ... 훈련 ...
        return model

# 두 Trainer가 같은 데이터를 복사 없이 공유
a = TrainerA.remote()
b = TrainerB.remote()
results = ray.get([a.train.remote(X_ref), b.train.remote(X_ref)])
```

### 5.3 자동 디스크 스필링 (Object Spilling)

Ray 1.3+ 부터 Object Store가 가득 차면 **자동으로 디스크에 스필링**한다.
이것이 AutoML의 GPU <-> CPU <-> Disk 간 데이터 이동의 핵심이다.

```python
import ray

# Object Store 크기와 스필링 디렉토리 설정
ray.init(
    object_store_memory=4 * 1024**3,  # 4GB
    _system_config={
        "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": "/fast-ssd/ray_spill"}}'
    }
)

# 또는 CLI에서:
# ray start --object-store-memory=4000000000
#            --object-spilling-directory=/fast-ssd/ray_spill
```

스필링 동작 흐름:

```
1. Object Store에 새 객체 저장 요청
2. 용량 확인:
   ├── 여유 있음 → 즉시 저장
   └── 용량 부족 → 스필링 시작
       ├── 참조 카운트가 낮은 객체를 디스크로 이동
       ├── 공간 확보 후 새 객체 저장
       └── 스필된 객체가 다시 필요하면 디스크에서 복원

모니터링:
  $ ray memory --stats-only
  ======== Object store ========
  Plasma memory usage 3.8 GiB (95.0% full)
  Objects spilled: 1.2 GiB (avg write 450 MiB/s)
  Objects restored: 800 MiB (avg read 600 MiB/s)
```

### 5.4 AutoML에서의 Swap 패턴

```python
import ray

@ray.remote(num_gpus=1)
class SequentialTrainer:
    """
    하나의 GPU에서 여러 모델을 순차 훈련.
    훈련 완료된 모델은 Object Store로 이동 → VRAM 해제.
    Object Store가 가득 차면 디스크로 자동 스필.
    """

    def train_and_store(self, X_ref, y_ref, model_configs):
        model_refs = []

        for config in model_configs:
            # 1) GPU에서 훈련
            model = self._train_model(config, X_ref, y_ref)

            # 2) 훈련된 모델을 Object Store로 이동 (CPU 메모리)
            #    → GPU VRAM 해제
            model_ref = ray.put(model)
            model_refs.append(model_ref)

            # 3) GPU 메모리 명시적 정리
            del model
            self._cleanup_gpu()

            # 이 시점에서 GPU VRAM은 비어 있고,
            # 모델은 Object Store (CPU)에 안전하게 보관됨.
            # Object Store가 가득 차면 디스크로 자동 스필.

        return model_refs

    def _train_model(self, config, X_ref, y_ref):
        X, y = ray.get(X_ref), ray.get(y_ref)
        # ... cuML/XGBoost 훈련 ...
        return model

    def _cleanup_gpu(self):
        import gc, cupy as cp
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
```

이 패턴으로 **8GB VRAM GPU에서도 수십 개 모델을 순차 훈련**할 수 있다.
각 모델은 훈련 후 Object Store → 디스크로 자동 이동하므로
VRAM은 항상 "현재 훈련 중인 모델" 하나만 차지한다.

---

## 6. Ray Data — 스트리밍 블록 처리와 백프레셔

### 6.1 Ray Data 개요

Ray Data는 대규모 데이터셋을 **블록 단위 스트리밍**으로 처리하는 라이브러리이다.
전체 데이터를 메모리에 올리지 않고, 블록을 하나씩 처리하며
**백프레셔(backpressure)** 메커니즘으로 메모리 초과를 방지한다.

### 6.2 블록 기반 처리 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                  Ray Data 파이프라인                       │
│                                                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────┐ │
│  │  Read    │───►│ Transform│───►│  Train  │───►│ Write │ │
│  │ Operator │    │ Operator │    │ Operator│    │Operator│ │
│  └────┬────┘    └────┬────┘    └────┬────┘    └───┬───┘ │
│       │              │              │              │      │
│  ┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌───▼───┐ │
│  │ Block 1 │    │ Block 1'│    │ Block 1"│    │Result1│ │
│  │ Block 2 │    │ Block 2'│    │ Block 2"│    │Result2│ │
│  │ Block 3 │    │ Block 3'│    │ Block 3"│    │Result3│ │
│  │   ...   │    │   ...   │    │   ...   │    │  ...  │ │
│  └─────────┘    └─────────┘    └─────────┘    └───────┘ │
│                                                           │
│  블록 크기: 1 ~ 128 MiB (기본)                             │
│  128 MiB 초과 시: 동적 블록 분할 (dynamic block splitting)  │
│  블록 형식: Arrow Table 또는 pandas DataFrame               │
└──────────────────────────────────────────────────────────┘
```

### 6.3 백프레셔 메커니즘

Ray Data의 스트리밍 실행기는 **세 가지 백프레셔** 조건을 확인하여
메모리 초과를 방지한다.

```
┌────────────────────────────────────────────────────────────┐
│              백프레셔 판단 흐름                               │
│                                                             │
│  Operator가 새 입력을 받으려면 다음 세 조건을 모두 충족해야 함:  │
│                                                             │
│  1. ┌─────────────────────────────────────┐                 │
│     │ 처리할 입력이 존재하는가?              │                 │
│     │ (입력 큐에 블록이 있는가?)             │                 │
│     └──────────────┬──────────────────────┘                 │
│                    │ Yes                                     │
│  2. ┌──────────────▼──────────────────────┐                 │
│     │ 클러스터 리소스가 충분한가?            │                 │
│     │ (CPU/GPU/메모리 여유가 있는가?)        │                 │
│     └──────────────┬──────────────────────┘                 │
│                    │ Yes                                     │
│  3. ┌──────────────▼──────────────────────┐                 │
│     │ 다운스트림 Operator가 막혀있지 않은가?  │                 │
│     │ (출력 큐가 가득 차지 않았는가?)         │                 │
│     └──────────────┬──────────────────────┘                 │
│                    │ Yes                                     │
│                    ▼                                         │
│              새 태스크 제출                                   │
│                                                             │
│  여러 Operator가 조건을 충족하면:                              │
│  → 출력 큐가 가장 작은 Operator를 우선 스케줄링                 │
│  → 다운스트림 병목 해소 우선                                   │
└────────────────────────────────────────────────────────────┘
```

### 6.4 AutoML 데이터 파이프라인 적용

```python
import ray
import ray.data

# 대용량 데이터셋을 스트리밍으로 처리
ds = ray.data.read_parquet("s3://bucket/higgs_11m.parquet")

# 전처리 파이프라인 — 블록 단위로 스트리밍 처리
preprocessed = (
    ds
    .map_batches(
        lambda batch: batch.dropna(),
        batch_format="pandas",
        # 한 번에 처리할 블록 크기 제한
        batch_size=100_000,
    )
    .map_batches(
        lambda batch: normalize(batch),
        batch_format="pandas",
        batch_size=100_000,
    )
)

# GPU 훈련을 위한 블록 반복
for batch in preprocessed.iter_batches(
    batch_size=500_000,
    batch_format="numpy",
    # 프리페치 블록 수 제한 → 메모리 제어
    prefetch_batches=2,
):
    X_batch = batch["features"]
    y_batch = batch["label"]
    # GPU로 전송하여 incremental 훈련
    # ...
```

### 6.5 Ray Data의 메모리 관리 전략

| 전략 | 설명 |
|:-----|:-----|
| Object Spilling | 블록이 Object Store 용량을 초과하면 자동으로 디스크로 이동 |
| Locality Scheduling | 데이터 블록이 존재하는 노드에서 태스크를 실행하여 네트워크 전송 최소화 |
| Reference Counting | 더 이상 참조되지 않는 블록은 자동 해제 |
| Dynamic Block Splitting | 192 MiB 초과 블록을 자동으로 분할하여 메모리 압력 분산 |
| Backpressure | 다운스트림 병목 시 업스트림 생산 속도를 자동 조절 |

---

## 7. Placement Groups — 리소스 배치 전략

### 7.1 Placement Group이란

Placement Group은 여러 Actor/Task가 **어떤 노드에 배치되어야 하는지**를 제어하는 메커니즘이다.
AutoML에서는 데이터 지역성, GPU 집중 배치, 또는 균등 분산을 전략적으로 선택할 수 있다.

### 7.2 네 가지 전략

```
┌─────────────────────────────────────────────────────────────────┐
│                     Placement Group 전략                          │
│                                                                  │
│  STRICT_PACK                    PACK                             │
│  ┌──────────────────┐           ┌──────────────────┐             │
│  │ Node A           │           │ Node A           │             │
│  │ ┌──┐ ┌──┐ ┌──┐  │           │ ┌──┐ ┌──┐ ┌──┐  │             │
│  │ │B1│ │B2│ │B3│  │           │ │B1│ │B2│ │B3│  │             │
│  │ └──┘ └──┘ └──┘  │           │ └──┘ └──┘ └──┘  │             │
│  └──────────────────┘           └──────────────────┘             │
│  모든 번들이 반드시 하나의       가능한 한 하나의 노드에 집중.       │
│  노드에 배치. 실패 시 전체 실패.  불가능하면 다른 노드도 사용.       │
│                                                                  │
│  STRICT_SPREAD                  SPREAD                           │
│  ┌────────┐┌────────┐┌────────┐ ┌────────┐┌────────┐┌────────┐  │
│  │ Node A ││ Node B ││ Node C │ │ Node A ││ Node B ││ Node C │  │
│  │ ┌──┐   ││ ┌──┐   ││ ┌──┐   │ │ ┌──┐   ││ ┌──┐   ││ ┌──┐   │  │
│  │ │B1│   ││ │B2│   ││ │B3│   │ │ │B1│   ││ │B2│   ││ │B3│   │  │
│  │ └──┘   ││ └──┘   ││ └──┘   │ │ └──┘   ││ └──┘   ││ └──┘   │  │
│  └────────┘└────────┘└────────┘ └────────┘└────────┘└────────┘  │
│  각 번들이 반드시 서로 다른       가능한 한 분산 배치.              │
│  노드에 배치. 실패 시 전체 실패.  불가능하면 겹침 허용.             │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 전략별 상세 비교

| 전략 | 동작 | 실패 조건 | AutoML 적합 시나리오 |
|:-----|:-----|:---------|:------------------|
| `STRICT_PACK` | 모든 번들을 단일 노드에 강제 배치 | 한 노드에 모든 리소스가 부족하면 실패 | 데이터 지역성이 최우선인 단일 GPU 훈련 |
| `PACK` | 가능한 한 한 노드에 집중, 불가능하면 분산 | 거의 실패하지 않음 | 일반적인 AutoML 파이프라인 (권장) |
| `SPREAD` | 가능한 한 서로 다른 노드에 분산 | 거의 실패하지 않음 | 다중 GPU 병렬 훈련, 내결함성 |
| `STRICT_SPREAD` | 각 번들을 반드시 다른 노드에 배치 | 노드 수 < 번들 수이면 실패 | 노드 장애 격리가 필수인 경우 |

### 7.4 AutoML에서의 Placement Group 활용

```python
import ray
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy

ray.init()

# 시나리오: 3개 모델을 2-GPU 노드에서 훈련
# XGBoost(0.5 GPU) + RF(0.3 GPU) + LR(0.2 GPU) = 1.0 GPU
# → 하나의 노드에 PACK하여 데이터 지역성 극대화

# 번들 정의: 각 번들은 리소스 요구사항
bundles = [
    {"GPU": 0.5, "CPU": 2},  # XGBoost
    {"GPU": 0.3, "CPU": 2},  # Random Forest
    {"GPU": 0.2, "CPU": 1},  # Logistic Regression
]

# Placement Group 생성 (PACK 전략)
pg = placement_group(bundles, strategy="PACK")
ray.get(pg.ready())  # 리소스 확보 완료까지 대기

# 각 Actor를 Placement Group 내 특정 번들에 배치
@ray.remote(num_gpus=0.5, num_cpus=2)
class XGBTrainer:
    def train(self, X_ref, y_ref): ...

@ray.remote(num_gpus=0.3, num_cpus=2)
class RFTrainer:
    def train(self, X_ref, y_ref): ...

@ray.remote(num_gpus=0.2, num_cpus=1)
class LRTrainer:
    def train(self, X_ref, y_ref): ...

# Placement Group 스케줄링 전략 적용
xgb_actor = XGBTrainer.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
).remote()

rf_actor = RFTrainer.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=1,
    )
).remote()

lr_actor = LRTrainer.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=2,
    )
).remote()

# 동일 노드에서 병렬 훈련 → Object Store zero-copy 활용
X_ref = ray.put(X_train)
y_ref = ray.put(y_train)

futures = [
    xgb_actor.train.remote(X_ref, y_ref),
    rf_actor.train.remote(X_ref, y_ref),
    lr_actor.train.remote(X_ref, y_ref),
]
models = ray.get(futures)
```

---

## 8. 종합 아키텍처: Ray 기반 PagedAutoML 설계

### 8.1 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                 Ray-based PagedAutoML Architecture                │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    Driver (Orchestrator)                   │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │    │
│  │  │ HPO Scheduler│  │ Memory Monitor│  │ Model Registry│  │    │
│  │  │ (Ray Tune)   │  │ (VRAM Tracker)│  │ (Object Store)│  │    │
│  │  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │    │
│  └─────────┼─────────────────┼──────────────────┼───────────┘    │
│            │                 │                   │                 │
│  ┌─────────▼─────────────────▼──────────────────▼───────────┐    │
│  │                   Ray Object Store                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │    │
│  │  │ X_train  │  │ y_train  │  │ model_rf │  │model_xgb │ │    │
│  │  │ (shared) │  │ (shared) │  │ (trained)│  │(trained) │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │    │
│  │                                                           │    │
│  │  용량 초과 시 → 자동 디스크 스필링 (/fast-ssd/ray_spill)    │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─── GPU 0 ────────────────────┐  ┌─── GPU 1 ───────────────┐  │
│  │  Placement Group (PACK)      │  │  Placement Group (PACK)  │  │
│  │                              │  │                          │  │
│  │  ┌────────────────────────┐  │  │  ┌──────────────────┐   │  │
│  │  │ XGBTrainer Actor       │  │  │  │ StackTrainer     │   │  │
│  │  │ num_gpus=0.5           │  │  │  │ Actor            │   │  │
│  │  │ CUDA_VISIBLE_DEVICES=0 │  │  │  │ num_gpus=1.0     │   │  │
│  │  └────────────────────────┘  │  │  │ CUDA_VIS..=1     │   │  │
│  │  ┌────────────────────────┐  │  │  │                  │   │  │
│  │  │ RFTrainer Actor        │  │  │  │ Meta Learner     │   │  │
│  │  │ num_gpus=0.3           │  │  │  │ 훈련 + OOF 수집   │   │  │
│  │  │ CUDA_VISIBLE_DEVICES=0 │  │  │  └──────────────────┘   │  │
│  │  └────────────────────────┘  │  │                          │  │
│  │  ┌────────────────────────┐  │  │                          │  │
│  │  │ LRTrainer Actor        │  │  │                          │  │
│  │  │ num_gpus=0.2           │  │  │                          │  │
│  │  │ CUDA_VISIBLE_DEVICES=0 │  │  │                          │  │
│  │  └────────────────────────┘  │  │                          │  │
│  └──────────────────────────────┘  └──────────────────────────┘  │
│                                                                   │
│  ┌────────── CPU Workers (데이터 전처리) ──────────────────────┐  │
│  │  @ray.remote Task: preprocess, feature_engineering, ...    │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 핵심 구현

```python
import ray
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy
from dataclasses import dataclass
from typing import Dict, List, Optional
import time

# ──────────────────────────────────────────────────────────
# 1. 설정
# ──────────────────────────────────────────────────────────

@dataclass
class AutoMLConfig:
    max_runtime_secs: int = 300
    max_models: int = 20
    gpu_memory_gb: float = 8.0
    models: List[str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = ["xgboost", "random_forest", "logistic_regression"]

# GPU 리소스 프로파일 (사전 측정 필요)
GPU_PROFILES = {
    "xgboost":              {"num_gpus": 0.5, "num_cpus": 2, "vram_est_gb": 3.0},
    "random_forest":        {"num_gpus": 0.3, "num_cpus": 2, "vram_est_gb": 2.0},
    "logistic_regression":  {"num_gpus": 0.2, "num_cpus": 1, "vram_est_gb": 1.0},
}

# ──────────────────────────────────────────────────────────
# 2. GPU Trainer Actor (프로세스 격리)
# ──────────────────────────────────────────────────────────

@ray.remote
class GPUTrainer:
    """
    각 인스턴스가 독립된 프로세스 + 독립된 CUDA context에서 실행.
    num_gpus는 동적으로 .options()에서 설정.
    """

    def __init__(self, model_type: str, pool_size_gb: float = 2.0):
        self.model_type = model_type
        self._init_gpu(pool_size_gb)

    def _init_gpu(self, pool_size_gb: float):
        """rmm pool 초기화 — Actor 생성 시 한 번만 실행"""
        import rmm
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=int(pool_size_gb * 1024**3),
        )

    def train(self, X_ref, y_ref, params: dict):
        """
        블랙박스 라이브러리 호출 — 내부 CUDA 할당을 제어하지 않음.
        프로세스 격리에 의해 다른 Actor의 VRAM에 영향 없음.
        """
        import cudf
        import numpy as np

        X = ray.get(X_ref)  # Object Store에서 zero-copy 읽기
        y = ray.get(y_ref)

        if self.model_type == "xgboost":
            import xgboost as xgb
            dtrain = xgb.DMatrix(X, label=y)
            model = xgb.train(params, dtrain, num_boost_round=100)

        elif self.model_type == "random_forest":
            from cuml.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params)
            model.fit(X, y)

        elif self.model_type == "logistic_regression":
            from cuml.linear_model import LogisticRegression
            model = LogisticRegression(**params)
            model.fit(X, y)

        return model

    def predict(self, X_ref):
        """추론 — OOF 예측 등에 사용"""
        X = ray.get(X_ref)
        return self.model.predict(X)

    def cleanup(self):
        """명시적 GPU 메모리 해제"""
        import gc, cupy as cp
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

# ──────────────────────────────────────────────────────────
# 3. AutoML Orchestrator
# ──────────────────────────────────────────────────────────

@ray.remote
class RayAutoML:
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.model_registry: Dict[str, ray.ObjectRef] = {}

    def run(self, X_ref, y_ref):
        start_time = time.time()

        # Phase 1: Base 모델 동시 훈련 (Placement Group으로 GPU 공유)
        base_models = self._train_base_models(X_ref, y_ref)

        # Phase 2: 시간 예산 확인
        elapsed = time.time() - start_time
        if elapsed > self.config.max_runtime_secs * 0.8:
            return base_models

        # Phase 3: Stacked Ensemble (별도 GPU에서)
        ensemble = self._train_ensemble(X_ref, y_ref, base_models)

        return ensemble

    def _train_base_models(self, X_ref, y_ref):
        """Fractional GPU + Placement Group으로 Base 모델 병렬 훈련"""

        # 번들 정의
        bundles = []
        for model_type in self.config.models:
            profile = GPU_PROFILES[model_type]
            bundles.append({
                "GPU": profile["num_gpus"],
                "CPU": profile["num_cpus"],
            })

        # Placement Group 생성 (PACK: 같은 노드에 집중)
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

        # 각 모델의 Trainer Actor 생성
        trainers = {}
        for i, model_type in enumerate(self.config.models):
            profile = GPU_PROFILES[model_type]
            trainer = GPUTrainer.options(
                num_gpus=profile["num_gpus"],
                num_cpus=profile["num_cpus"],
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote(model_type=model_type)
            trainers[model_type] = trainer

        # 병렬 훈련 시작
        futures = {
            name: trainer.train.remote(X_ref, y_ref, {})
            for name, trainer in trainers.items()
        }

        # 결과 수집 → Object Store에 저장
        results = {}
        for name, future in futures.items():
            model = ray.get(future)
            model_ref = ray.put(model)  # Object Store로 이동
            self.model_registry[name] = model_ref
            results[name] = model_ref

        # GPU 정리
        for trainer in trainers.values():
            ray.get(trainer.cleanup.remote())

        return results

    def _train_ensemble(self, X_ref, y_ref, base_model_refs):
        """Stacked Ensemble 훈련 — 별도 GPU Actor에서"""

        @ray.remote(num_gpus=1)
        class EnsembleTrainer:
            def train(self, X_ref, y_ref, base_model_refs):
                # OOF 예측 수집 + Meta Learner 훈련
                # ...
                pass

        trainer = EnsembleTrainer.remote()
        ensemble = ray.get(trainer.train.remote(X_ref, y_ref, base_model_refs))
        return ensemble
```

### 8.3 메모리 흐름 시나리오

8GB VRAM GPU 1개에서 3개 모델을 훈련하는 시나리오:

```
시간 ──────────────────────────────────────────────────────►

GPU VRAM 사용량 (8GB 중):

Phase 1: Base 모델 동시 훈련 (Fractional GPU)
┌───────────────────────────────────────────────────────┐
│ XGB(0.5) + RF(0.3) + LR(0.2) = 1.0 GPU               │
│                                                        │
│ VRAM:  [===XGB 3GB===][==RF 2GB==][LR 1GB]  (6/8 GB)  │
│                                                        │
│ 각 프로세스가 독립 CUDA context                          │
│ OOM 시 해당 프로세스만 실패 (다른 모델은 계속)              │
└───────────────────────────────────────────────────────┘
         │
         ▼  훈련 완료 → 모델을 Object Store로 이동
┌───────────────────────────────────────────────────────┐
│ Object Store (CPU 공유 메모리):                         │
│ [model_xgb] [model_rf] [model_lr] [X_train] [y_train] │
│                                                        │
│ GPU VRAM:  [비어있음]  (0/8 GB)                         │
└───────────────────────────────────────────────────────┘
         │
         ▼  Phase 2: Ensemble 훈련 (전체 GPU 사용)
┌───────────────────────────────────────────────────────┐
│ StackTrainer(1.0 GPU)                                  │
│                                                        │
│ VRAM:  [=========Ensemble 5GB=========]  (5/8 GB)      │
│                                                        │
│ Object Store에서 base 모델을 필요 시 로드               │
│ 불필요한 모델은 디스크로 자동 스필                        │
└───────────────────────────────────────────────────────┘
```

### 8.4 Ray + RAPIDS (cuML/XGBoost) 통합

```python
import ray
import numpy as np

ray.init()

# ──────────────────────────────────────────────
# XGBoost-Ray 통합 (공식 지원)
# ──────────────────────────────────────────────

from xgboost_ray import RayDMatrix, RayParams, train as xgb_ray_train

def train_xgboost_distributed(X_ref, y_ref):
    """Ray 위에서 분산 XGBoost 훈련"""

    X, y = ray.get(X_ref), ray.get(y_ref)
    dtrain = RayDMatrix(X, label=y)

    config = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "gpu_hist",
        "max_depth": 6,
    }

    # 4개 GPU Actor에서 병렬 훈련
    bst = xgb_ray_train(
        config,
        dtrain,
        num_boost_round=100,
        ray_params=RayParams(
            num_actors=4,
            gpus_per_actor=1,
            # 내결함성: Actor 실패 시 자동 재시작
            max_actor_restarts=2,
        ),
    )
    return bst

# ──────────────────────────────────────────────
# cuML Actor (프로세스 격리)
# ──────────────────────────────────────────────

@ray.remote(num_gpus=1)
class CuMLActor:
    """cuML 모델을 Ray Actor로 래핑 — GPU 격리 + 상태 유지"""

    def __init__(self):
        import rmm
        rmm.reinitialize(pool_allocator=True, initial_pool_size=2 * 1024**3)

    def train_random_forest(self, X_ref, y_ref, params):
        from cuml.ensemble import RandomForestClassifier
        X, y = ray.get(X_ref), ray.get(y_ref)

        import cudf
        X_gpu = cudf.DataFrame(X)
        y_gpu = cudf.Series(y)

        model = RandomForestClassifier(**params)
        model.fit(X_gpu, y_gpu)
        return model

    def train_logistic_regression(self, X_ref, y_ref, params):
        from cuml.linear_model import LogisticRegression
        X, y = ray.get(X_ref), ray.get(y_ref)
        model = LogisticRegression(**params)
        model.fit(X, y)
        return model
```

---

## 9. vLLM 페이징 vs Ray 프로세스 페이징 비교

### 9.1 근본적 차이

```
┌─────────────── vLLM PagedAttention ────────────────┐
│                                                     │
│  관리 대상:  KV Cache (Attention 연산의 중간 결과)    │
│  관리 단위:  블록 (16 tokens ≈ 수 KB)                │
│  관리 시점:  연산 도중 (커널 내부)                     │
│  구현 방법:  커스텀 CUDA 커널 + Page Table             │
│  전제 조건:  관리 대상의 메모리 접근 패턴을 완전히 제어   │
│                                                     │
│  ┌──────── GPU VRAM ─────────┐                      │
│  │ ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐ │                      │
│  │ │P0││P1││P2││P3││P4││P5│ │  Physical Pages       │
│  │ └──┘└──┘└──┘└──┘└──┘└──┘ │                      │
│  │   ▲       ▲    ▲         │                      │
│  │   │       │    │         │                      │
│  │ ┌─┴──┬──┬─┴──┬─┴──┐     │  Page Table           │
│  │ │ L0 │L1│ L2 │ L3 │     │  (logical→physical)   │
│  │ └────┴──┴────┴────┘     │                      │
│  └──────────────────────────┘                      │
└─────────────────────────────────────────────────────┘

┌─────────────── Ray Process Paging ─────────────────┐
│                                                     │
│  관리 대상:  프로세스 (Actor/Task)                    │
│  관리 단위:  프로세스 전체 (수 GB)                     │
│  관리 시점:  연산 사이 (스케줄링 시점)                  │
│  구현 방법:  CUDA_VISIBLE_DEVICES + Object Store     │
│  전제 조건:  없음 (블랙박스 라이브러리 호환)             │
│                                                     │
│  ┌──────── GPU VRAM ─────────┐                      │
│  │ ┌──────────────────────┐  │                      │
│  │ │   Process A          │  │  독립 CUDA Context    │
│  │ │   (cuML RF 전체)     │  │                      │
│  │ ├──────────────────────┤  │                      │
│  │ │   Process B          │  │  독립 CUDA Context    │
│  │ │   (XGBoost 전체)     │  │                      │
│  │ └──────────────────────┘  │                      │
│  └──────────────────────────┘                      │
│         │                                           │
│         │ Object Store                              │
│         ▼ (자동 swap)                                │
│  ┌──────────────────────────┐                      │
│  │   CPU 공유 메모리          │                      │
│  │   훈련 완료된 모델 보관     │                      │
│  └──────────┬───────────────┘                      │
│             │ 자동 spill                             │
│             ▼                                       │
│  ┌──────────────────────────┐                      │
│  │   Disk                    │                      │
│  └──────────────────────────┘                      │
└─────────────────────────────────────────────────────┘
```

### 9.2 상세 비교표

| 비교 항목 | vLLM PagedAttention | Ray Process Paging |
|:----------|:-------------------|:------------------|
| **관리 단위** | 블록 (16 tokens ~ 수 KB) | 프로세스 (수 GB) |
| **관리 시점** | 연산 **도중** (커널 내부) | 연산 **사이** (스케줄링 시점) |
| **입도(Granularity)** | Fine-grained (KB 수준) | Coarse-grained (GB 수준) |
| **구현 복잡도** | 매우 높음 (커스텀 CUDA 커널) | 낮음 (Python API) |
| **블랙박스 호환** | 불가 (내부 접근 필수) | **가능** (핵심 장점) |
| **메모리 활용률** | ~ 96% (논문 검증) | ~ 70 ~ 85% (추정, 프로세스 오버헤드) |
| **Swap 메커니즘** | GPU ↔ CPU KV Cache 전송 | Object Store 자동 스필링 |
| **Swap 단위** | 블록 단위 (수 KB) | 객체 단위 (수 MB ~ GB) |
| **적용 대상** | LLM 추론 (Transformer) | 범용 GPU 연산 (ML, AutoML) |
| **장애 격리** | 프로세스 내 공유 | 프로세스 간 격리 (Actor 실패 독립) |
| **확장성** | 단일 모델 최적화 | 다중 모델 병렬 실행 |
| **데이터 공유** | 불필요 (단일 모델) | Object Store zero-copy (다중 모델) |

### 9.3 PagedAutoML의 위치

```
입도(Granularity) ◄───────────────────────────────────────► 블랙박스 호환성

Fine-grained                                    Coarse-grained
(KB 단위)                                       (GB 단위)
  │                                                    │
  │  vLLM              rmm Custom        Ray           │
  │  PagedAttention    Allocator         Process       │
  │                                      Paging        │
  │  ┌────┐            ┌────┐            ┌────┐        │
  │  │ 96%│ 활용률     │ 85%│ 추정       │ 75%│ 추정    │
  │  └────┘            └────┘            └────┘        │
  │                                                    │
  │  블랙박스 불가      부분 호환          완전 호환      │
  │  (커널 직접 작성)   (rmm 레벨만)      (any library)  │
  │                                                    │
  ▼                                                    ▼
  높은 최적화                              높은 범용성

  현재 PagedAutoML의 한계:
  ┌──────────────────────────────────────────────────┐
  │ vLLM 스타일을 시도했으나 블랙박스 제약으로 실패      │
  │ → Ray Process Paging이 현실적 대안                │
  └──────────────────────────────────────────────────┘
```

### 9.4 왜 Ray가 AutoML에 더 적합한가

1. **블랙박스 호환**: cuML, XGBoost, LightGBM 등 어떤 라이브러리든 Actor로 래핑만 하면 된다.
   내부 CUDA 할당을 가로챌 필요가 없다.

2. **장애 격리**: 하나의 모델이 OOM으로 죽어도 다른 Actor는 영향받지 않는다.
   현재 PagedAutoML에서 "모든 모델이 skip되는" 문제를 방지한다.

3. **자동 Swap**: Object Store의 자동 스필링으로 CPU 메모리와 디스크까지
   3단계 메모리 계층을 투명하게 활용한다.

4. **스케일 아웃**: 단일 GPU에서 다중 GPU, 나아가 다중 노드로의 확장이
   코드 변경 최소화로 가능하다.

5. **HPO 통합**: Ray Tune과 자연스럽게 연동되어 하이퍼파라미터 최적화를
   분산 환경에서 실행할 수 있다.

6. **데이터 파이프라인**: Ray Data의 스트리밍 처리로 메모리에 들어가지 않는
   대용량 데이터셋도 블록 단위로 처리 가능하다.

### 9.5 트레이드오프

Ray Process Paging의 한계도 명확히 인식해야 한다.

| 한계 | 설명 |
|:-----|:-----|
| 메모리 오버헤드 | 프로세스마다 CUDA context가 ~ 300MB ~ 1GB 추가 소비 |
| 입도 제한 | "모델 훈련 도중"의 메모리 재배치는 여전히 불가능 |
| VRAM 파티션 없음 | Fractional GPU는 논리적 관리이며, 물리적 OOM 보호는 없음 |
| 프로세스 생성 비용 | Actor 시작 시 CUDA context 초기화에 2 ~ 5초 소요 |
| 직렬화 비용 | GPU 객체를 Object Store로 이동 시 GPU → CPU 복사 발생 |

---

## 10. References

### Ray 공식 문서

1. [Ray Core System Design: A Deep Dive](https://nkkarpov.github.io/blog/ray-core-system-design/) — GCS, Raylet, Object Store 아키텍처 상세
2. [Memory Management — Ray Docs](https://docs.ray.io/en/latest/ray-core/scheduling/memory-management.html) — Object Store, 메모리 계층, 설정 파라미터
3. [Object Spilling — Ray Docs](https://docs.ray.io/en/latest/ray-core/objects/object-spilling.html) — 자동 디스크 스필링 설정 및 모니터링
4. [GPU Support — Ray Docs](https://docs.ray.io/en/releases-2.7.1/ray-core/tasks/using-ray-with-gpus.html) — CUDA_VISIBLE_DEVICES 관리, Fractional GPU
5. [Accelerator Support — Ray Docs](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html) — GPU/TPU 리소스 스케줄링
6. [Ray Data Internals — Ray Docs](https://docs.ray.io/en/latest/data/data-internals.html) — 스트리밍 실행, 블록 처리, 백프레셔
7. [Placement Groups — Ray Docs](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) — PACK, SPREAD 등 배치 전략
8. [Fractional GPU Serving — Ray Docs](https://docs.ray.io/en/latest/serve/llm/user-guides/fractional-gpu.html) — Fractional GPU 활용 가이드
9. [Resources — Ray Docs](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html) — 리소스 요구사항 및 스케줄링

### Ray + RAPIDS / XGBoost

10. [Distributed XGBoost with Ray — XGBoost Docs](https://xgboost.readthedocs.io/en/stable/tutorials/ray.html) — xgboost_ray 공식 튜토리얼
11. [Distributed XGBoost Training with Ray — Anyscale Blog](https://www.anyscale.com/blog/distributed-xgboost-training-with-ray) — RayDMatrix, RayParams 활용
12. [xgboost_ray — GitHub](https://github.com/ray-project/xgboost_ray) — XGBoost-Ray 라이브러리
13. [RAPIDS Cloud ML Examples — GitHub](https://github.com/rapidsai/cloud-ml-examples) — cuML + Ray/Dask 통합 예시

### vLLM 및 메모리 관리

14. [Efficient Memory Management for LLM Serving with PagedAttention (arXiv:2309.06180)](https://arxiv.org/pdf/2309.06180) — vLLM 원 논문
15. [Paged Attention — vLLM Docs](https://docs.vllm.ai/en/stable/design/paged_attention/) — PagedAttention 설계 문서
16. [The Architecture Behind vLLM — Medium](https://medium.com/@mandeep0405/the-architecture-behind-vllm-how-pagedattention-improves-memory-utilization-2f9b25272110) — PagedAttention 아키텍처 해설

### Ray 아키텍처 해설

17. [Ray: Distributed Computing Framework — Medium (Juniper AI/ML)](https://medium.com/juniper-team/ray-distributed-computing-framework-for-ai-ml-applications-4b40617be4a3) — Ray 아키텍처 개요
18. [Ray Core Internals — DS 5110 Lecture](https://tddg.github.io/ds5110-cs5501-spring24/assets/docs/lec6b-ray-internals.pdf) — Ray 내부 구조 강의 자료
19. [Ray DeepWiki](https://deepwiki.com/ray-project/ray) — Ray 프로젝트 위키
20. [Plasma In-Memory Object Store — Ray Blog](https://ray-project.github.io/2017/08/08/plasma-in-memory-object-store.html) — Plasma 설계 원리
