# GPU AutoML 메모리 최적화 연구 방향

> 04번 문서의 Phase 3에 해당하는 내용을 상세화한 문서이다.
> GPU AutoML 파이프라인에서 메모리가 어디서, 왜, 얼마나 문제가 되는지를 분석하고,
> RAPIDS(rmm, Dask-CUDA)를 활용한 최적화 전략을 정리한다.

---

## 1. 문제 인식: GPU AutoML에서 메모리가 왜 병목인가

### 1-1. CPU vs GPU 메모리 환경의 근본적 차이

```
CPU AutoML 환경:
  시스템 RAM: 64~256GB (보통 충분)
  → 메모리가 실질적 제약이 되는 경우가 드물다
  → H2O AutoML은 이 환경을 전제로 설계되었다

GPU AutoML 환경:
  GPU VRAM: 16~24GB (항상 부족)
  → AutoML이 하는 일을 이 안에서 돌려야 한다
```

### 1-2. AutoML 파이프라인에서 VRAM을 소비하는 것들

```
[동시에 VRAM에 있어야 하는 것]

1. 원본 데이터 (cuDF DataFrame)
   100만 행 × 50 feature × float32 = ~200MB
   1000만 행 × 50 feature × float32 = ~2GB

2. 현재 훈련 중인 모델의 내부 데이터
   XGBoost: 히스토그램 버퍼, 트리 구조 → ~2~6GB (데이터 크기에 비례)
   cuML RF: 트리 노드 배열 → ~3~8GB
   cuML GLM: 행렬 연산 임시 버퍼 → ~0.5~2GB

3. OOF 예측값 (Level-One Data 생성 중)
   진행 중인 모델들의 OOF → 모델 수에 비례하여 증가

4. cuDF/cuML 임시 버퍼
   정렬, 인덱싱, 행렬 연산 등의 중간 결과물
```

### 1-3. VRAM 사용량 스케일링

| 요소 | 스케일링 | 예시 (100만 행, 50 features) |
|------|----------|------------------------------|
| 원본 데이터 | O(N × F) | ~200MB |
| 모델 훈련 (XGBoost 1개) | O(N × depth) | ~3GB |
| 모델 훈련 (cuML RF 1개) | O(N × trees) | ~5GB |
| Level-One Data | O(N × L) | 모델 20개 → ~80MB |
| **동시 실행 시 peak** | **데이터 + 활성 모델들** | **데이터 0.2 + 모델 2개 ~8 = ~8.2GB** |

문제는 **동시 실행 모델 수가 늘어날 때**이다:
- 모델 1개 동시: 0.2 + 5 = 5.2GB → 문제 없음
- 모델 3개 동시: 0.2 + 5 + 3 + 2 = 10.2GB → 16GB GPU에서 위험
- 모델 5개 동시: 0.2 + 5 + 3 + 5 + 3 + 2 = 18.2GB → 24GB GPU에서도 위험

**병렬화를 많이 할수록 빨라지지만, 메모리 제약에 부딪힌다.**
이것이 "성능 vs 메모리" 트레이드오프이며, 우리 연구의 핵심 문제이다.

---

## 2. 메모리 프로파일링: rmm logging 활용

### 2-1. rmm logging allocator

rmm은 GPU 메모리의 모든 할당/해제를 로깅하는 기능을 제공한다.
이것으로 AutoML 파이프라인의 **메모리 사용 패턴을 정밀 측정**할 수 있다.

```python
import rmm

# 로깅 활성화
rmm.reinitialize(logging=True, log_file="automl_memory_trace.log")

# 이후 cuDF, cuML의 모든 GPU 메모리 연산이 기록됨
# → 어떤 시점에 얼마나 할당/해제되는지 추적 가능
```

### 2-2. 측정해야 할 것

**단계별 peak VRAM:**
```
파이프라인 단계           peak VRAM    설명
──────────────────────   ──────────   ──────────────────
데이터 로드              ???MB        cudf.read_csv 후
전처리 + fold 분할       ???MB        임시 버퍼 포함
XGBoost 훈련 (1 fold)   ???MB        히스토그램 + 트리
cuML RF 훈련 (1 fold)   ???MB        트리 노드 배열
cuML GLM 훈련 (1 fold)  ???MB        행렬 연산 버퍼
OOF 수집 (모델 10개)    ???MB        Level-One Data
Meta Learner 훈련       ???MB        Level-One + GLM
```

**모델별 메모리 footprint:**
```
모델 종류      데이터 10만행   100만행   1000만행
────────────   ───────────   ────────   ─────────
XGBoost        ???MB         ???MB      ???MB
cuML RF        ???MB         ???MB      ???MB
cuML GLM       ???MB         ???MB      ???MB
PyTorch MLP    ???MB         ???MB      ???MB
```

이 `???`를 실제로 채우는 것이 **Phase 2의 핵심 산출물**이다.
이 데이터가 있어야 Memory-Aware Scheduling을 설계할 수 있다.

### 2-3. 프로파일링 결과의 시각화

```
VRAM 사용 타임라인 (예상):

24GB ┤
     │
20GB ┤                              ┌─────┐
     │              ┌──────┐        │OOF  │
16GB ┤              │RF    │   ┌──┐ │수집 │
     │  ┌──────┐    │훈련  │   │XG│ │+Meta│  ← peak 구간 식별
12GB ┤  │XGB   │    │      │   │B │ │    │
     │  │훈련  │    │      │   │  │ │    │
 8GB ┤  │      │    │      │   │  │ │    │
     │  │      │    │      │   │  │ └────┘
 4GB ┤──┤데이터├────┤데이터├───┤  ├──────── ← 데이터 상주
     │  │      │    │      │   │  │
 0GB ┼──┴──────┴────┴──────┴───┴──┴──────→ 시간
```

---

## 3. Memory-Aware Scheduling 설계

### 3-1. 기존 접근 (Memory-Naive)의 문제

```
Naive 스케줄러:
  "가용 worker가 있으면 바로 task 배정"
  
  → 작은 모델(GLM, 0.8GB)도 큰 모델(RF, 5GB)도 같은 방식으로 배정
  → 큰 모델 2개가 같은 GPU에 배정되면 OOM
  → OOM 발생 후 재시도 → 시간 낭비
```

### 3-2. Memory-Aware 접근

```
Memory-Aware 스케줄러:
  Step 1: 모델별 예상 VRAM 사용량 추정 (프로파일 데이터 기반)
  Step 2: 각 GPU worker의 현재 가용 VRAM 확인
  Step 3: 가용 VRAM 내에서 실행 가능한 조합을 선택
  Step 4: 실행 중 실제 VRAM을 모니터링하며 동적 조정
```

구체적인 스케줄링 예시:

```
GPU 0 (VRAM 24GB):
  데이터: 2GB 사용 중
  가용: 22GB

  스케줄러 판단:
  - XGBoost(~3GB) + RF(~5GB) + GLM(~1GB) = 9GB → ✅ 동시 실행
  - XGBoost(~3GB) + RF(~5GB) + RF(~5GB)  = 13GB → ✅ 동시 실행
  - XGBoost × 3 + RF × 2 = 19GB → ⚠️ 여유 3GB (위험)
  - XGBoost × 3 + RF × 3 = 24GB → ❌ 대기열에 추가

GPU 1 (VRAM 16GB):
  데이터: 2GB 사용 중
  가용: 14GB

  스케줄러 판단:
  - GLM은 여기로 (가벼워서 16GB GPU에 적합)
  - RF는 GPU 0으로 (무거워서 24GB GPU에 배정)
```

### 3-3. Dask-CUDA에서의 구현 접점

Dask-CUDA의 `device_memory_limit`이 기본적인 메모리 보호를 제공하지만,
**task 단위의 세밀한 스케줄링**은 우리가 Orchestrator 레벨에서 구현해야 한다.

```python
# 방법 1: Dask resource annotation 활용
# worker에 커스텀 리소스를 명시하고, task 제출 시 요구 리소스를 지정

cluster = LocalCUDACluster(
    n_workers=2,
    resources={"VRAM": 24_000},  # 각 worker에 VRAM 24GB 리소스 태깅
)

# task 제출 시 예상 VRAM 사용량을 리소스 요구사항으로 지정
future = client.submit(
    train_xgboost_gpu, X, y, params,
    resources={"VRAM": 3_000},   # 이 task는 ~3GB 필요
)

# Dask 스케줄러가 VRAM 여유가 있는 worker에만 배정
```

```python
# 방법 2: 수동 메모리 체크 후 제출
import pynvml

def get_free_vram(device_id=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free / (1024**3)  # GB

# 스케줄러 로직
estimated_vram = estimate_model_vram("xgboost", data_size=len(train))
free_vram = get_free_vram(device_id=0)

if free_vram > estimated_vram * 1.2:  # 20% 여유 확보
    client.submit(train_xgboost_gpu, ...)
else:
    # 대기하거나 다른 GPU에 배정
    pass
```

---

## 4. Stacking에서의 메모리 최적화

### 4-1. Level-One Data 메모리 문제

AutoML은 모델을 계속 추가한다. Level-One Data 크기는 모델 수에 비례:

```
모델 수(L)   Level-One 크기 (N=100만)   누적 VRAM
──────────   ────────────────────────   ──────────
10개         100만 × 10 × 4B = 40MB     +40MB
50개         100만 × 50 × 4B = 200MB    +200MB
100개        100만 × 100 × 4B = 400MB   +400MB
```

Level-One Data 자체는 크지 않지만, **원본 데이터 + 활성 모델 + Level-One Data**가
동시에 VRAM에 있어야 하는 시점이 문제이다.

### 4-2. 최적화 전략 A: Streaming OOF

모든 모델의 OOF를 동시에 VRAM에 유지하지 않는다.

```
기존 (Memory-Naive):
  모델A OOF 생성 → VRAM에 유지
  모델B OOF 생성 → VRAM에 유지
  ...
  모델L OOF 생성 → VRAM에 유지
  전체 Level-One Data가 VRAM에 → Meta Learner 훈련

최적화 (Streaming):
  모델A OOF 생성 → Host RAM으로 이동 (cudf.to_pandas() or cupy.asnumpy())
  모델B OOF 생성 → Host RAM으로 이동
  ...
  Meta Learner 훈련 시 → 필요한 만큼만 VRAM에 로드
```

VRAM이 부족한 환경에서 모델 수를 늘릴 수 있는 전략이다.
트레이드오프: Host↔Device 전송 비용이 추가되지만, Level-One Data는 크기가 작아서 영향 미미.

### 4-3. 최적화 전략 B: Adaptive Model Pool

메모리 예산에 따라 앙상블에 포함할 모델 수를 **동적으로 결정**한다.

```
VRAM 여유 상태 확인:
  충분 (>8GB 가용) → All Models Ensemble (모델 전체 포함)
  보통 (4~8GB)     → Top-K Models (상위 K개만 포함)
  부족 (<4GB)      → Best of Family (알고리즘별 1개씩, 5~6개)

H2O의 Two-type Ensemble이 사실은 메모리 관점의 전략이기도 하다:
  All Models     = 메모리 여유 있을 때 최고 성능
  Best of Family = 메모리 제약 하에서 최적의 성능/메모리 균형
```

이것을 "수동 선택"에서 "메모리 예산 기반 자동 선택"으로 일반화하면
그 자체가 연구 기여가 된다.

### 4-4. 최적화 전략 C: Mixed Precision

Level-One Data와 일부 모델 훈련에 **float16** 또는 **bfloat16**을 사용:

```
float32: N × L × 4 bytes
float16: N × L × 2 bytes → 메모리 50% 절약

실험 질문:
  Meta Learner(GLM)의 정확도에 float16 Level-One Data가 미치는 영향은?
  → "거의 차이 없다"를 실험적으로 보이면 그 자체로 유의미한 결과
```

XGBoost의 `enable_categorical`이나 cuML의 `dtype` 파라미터로 구현 가능.

---

## 5. rmm 풀 전략 연구

### 5-1. 풀 전략 비교

| 전략 | 설정 | 장점 | 단점 | 적합한 상황 |
|------|------|------|------|-------------|
| No pool | `pool_allocator=False` | 설정 불필요 | 할당/해제 느림 | 테스트용 |
| Fixed pool | `pool_allocator=True, initial_pool_size=10GB` | 할당 빠름, 예측 가능 | 사전 크기 결정 필요 | 데이터 크기가 알려진 경우 |
| Managed memory | `managed_memory=True` | VRAM 초과 가능 | 성능 저하 | VRAM 부족 환경 |

### 5-2. AutoML에 최적화된 풀 전략 연구

AutoML은 다른 ML 워크로드와 메모리 사용 패턴이 다르다:
- **반복적**: 같은 크기의 데이터로 모델을 수십~수백 번 훈련
- **증가하는**: 모델이 추가될수록 OOF 데이터가 누적
- **다양한**: XGBoost, RF, GLM 등 모델마다 메모리 패턴이 다름

이 패턴에 맞춘 **Adaptive Pool** 전략을 연구할 수 있다:

```
AutoML 시작:
  Phase 1 (Baseline): 작은 풀로 시작 (모델이 작으므로)
  Phase 2 (Diversity): 풀 확대 (다양한 모델 동시 실행)
  Phase 3 (Random Search): 풀 크기 유지 + spill 활성화
  Phase 4 (Stacking): Level-One Data용 영역 확보

→ 파이프라인 단계에 따라 rmm 풀 설정을 동적으로 조정
```

### 5-3. 실험 매트릭스

```
변수:
  Pool 전략:    No pool / Fixed 8GB / Fixed 16GB / Managed / Adaptive
  데이터 크기:  10만 행 / 100만 행 / 1000만 행
  모델 수:      5개 / 20개 / 50개
  GPU:          16GB / 24GB

측정값:
  - 전체 파이프라인 소요 시간
  - Peak VRAM 사용량
  - OOM 발생 여부
  - 메모리 할당/해제 횟수 및 소요 시간
  - 최종 모델 정확도 (메모리 전략이 정확도에 영향을 주는지)
```

---

## 6. 기존 GPU AutoML과의 차별점

| | AutoGluon + RAPIDS | TPOT + RAPIDS | 우리 시스템 |
|---|---|---|---|
| RAPIDS 활용 범위 | 일부 모델만 cuML 대체 | 일부 전처리 + cuML | **전 파이프라인 RAPIDS** |
| 분산 전략 | Ray 기반 | 없음 (싱글) | **Dask-CUDA 네이티브** |
| 메모리 관리 | 프레임워크 기본값 | 없음 | **rmm 풀 + Memory-Aware Scheduling** |
| 메모리 프로파일링 | 없음 | 없음 | **rmm logging 기반 정밀 측정** |
| Stacking 메모리 최적화 | 없음 | 없음 | **Streaming OOF + Adaptive Model Pool** |
| Stacking 전략 | 자체 구현 | 없음 | **H2O 전략 계승** |

**기존 GPU AutoML 프레임워크들은 "속도"에만 집중하고 "메모리"는 무시한다.**
우리는 메모리를 1등 시민(first-class citizen)으로 다루는 GPU AutoML을 만든다.

---

## 7. 연구 로드맵 요약

```
Phase 1 — 측정 (현상 파악)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  rmm logging으로 GPU AutoML 파이프라인의 메모리 사용 패턴 정밀 측정
  → 모델별, 단계별, 데이터 크기별 프로파일 확보
  → 이 데이터가 이후 모든 최적화의 근거

Phase 2 — 분석 (병목 식별)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  프로파일 데이터에서 패턴 추출
  → 어떤 모델이 메모리를 많이 쓰는가
  → 어떤 단계에서 peak가 발생하는가
  → 메모리 부족이 정확도에 미치는 영향은?

Phase 3 — 최적화 (핵심 연구)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Memory-Aware Scheduling 구현
  Stacking 메모리 최적화 (Streaming OOF, Adaptive Model Pool)
  rmm 풀 전략 최적화 (Adaptive Pool)

Phase 4 — 검증 (벤치마크)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Memory-Naive vs Memory-Aware 비교 실험
  다양한 데이터셋 + GPU 환경에서 재현성 확인
  → 같은 VRAM에서 더 많은 모델 탐색 가능 / OOM 발생률 감소 / 처리량 향상
```
