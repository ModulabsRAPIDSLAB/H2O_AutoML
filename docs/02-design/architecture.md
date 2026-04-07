# Architecture: Research-to-Code Mapping

이 문서는 Phase 1(연구)에서 분석한 H2O 개념이 `paged_automl/` 코드의 어느 모듈로 구현되었는지를 매핑합니다.

## H2O Concept → paged_automl/ Module

| H2O 개념 | 연구 문서 | paged_automl/ 모듈 | GPU 구현 방식 |
|----------|----------|-----------------|--------------|
| Stacked Ensemble (All Models + BoF) | [01_stacked_ensemble](../01-research/01_stacked_ensemble_deep_dive.md) | `ensemble/stacking.py` | cuML GLM meta learner로 OOF 기반 2-type 앙상블 |
| Non-negative GLM Meta Learner | [01_stacked_ensemble](../01-research/01_stacked_ensemble_deep_dive.md) | `ensemble/meta_learner.py` | Post-hoc clipping + renorm (GPU), scipy NNLS (fallback) |
| HPO: Baseline → Diversity → Random Search | [02_hyperparameter](../01-research/02_hyperparameter_tuning_deep_dive.md) | `orchestrator.py`, `hpo/presets.py`, `hpo/random_search.py` | 동일 전략, cuML/XGBoost GPU 위에서 실행 |
| 5-fold CV + OOF 수집 | [01_stacked_ensemble](../01-research/01_stacked_ensemble_deep_dive.md) | `data/cv.py` | cuDF 인덱싱, streaming OOF로 VRAM 절약 |
| H2O Frame (데이터 구조) | [03_gpu_native](../01-research/03_gpu_native_application.md) | `data/loader.py` | cuDF DataFrame (GPU-native) |
| H2O RF / GBM / GLM | [03_gpu_native](../01-research/03_gpu_native_application.md) | `models/cuml_rf.py`, `models/xgboost_gpu.py`, `models/cuml_glm.py` | cuML RF, XGBoost `device='cuda'`, cuML GLM |
| 시간 기반 제어 (`max_runtime_secs`) | [02_hyperparameter](../01-research/02_hyperparameter_tuning_deep_dive.md) | `orchestrator.py` | `time.perf_counter()` 기반 budget 관리 |
| **Memory-Aware Scheduling** (새로운 기여) | [05_memory](../01-research/05_memory_optimization_research.md) | `scheduler.py`, `memory/estimator.py` | VRAM 추정 → 가용 메모리 체크 → 초과 시 skip |
| rmm 풀 관리 | [05_memory](../01-research/05_memory_optimization_research.md) | `memory/pool.py` | 4가지 전략 (None/Fixed/Managed/Adaptive) |
| VRAM 프로파일링 | [05_memory](../01-research/05_memory_optimization_research.md) | `memory/profiler.py` | pynvml 기반 실시간 VRAM 모니터링 |

## Module Dependency Flow

```
GPUAutoML (automl.py)
  │
  ├── _init_gpu_resources()
  │     ├── MemoryProfiler (memory/profiler.py)    ── pynvml
  │     ├── RMMPoolManager (memory/pool.py)         ── rmm
  │     └── LocalCUDACluster (dask-cuda, optional)
  │
  ├── _prepare_data() → Preprocessor (data/preprocessor.py) ── cuDF
  │
  └── Orchestrator (orchestrator.py)
        │
        ├── VRAMEstimator (memory/estimator.py)   ← VRAM 추정
        ├── CrossValidator (data/cv.py)            ← 5-fold CV + OOF
        │     └── BaseModel.fit() / predict()
        │           ├── XGBoostGPU (models/xgboost_gpu.py)
        │           ├── CuMLRandomForest (models/cuml_rf.py)
        │           └── CuMLGLM (models/cuml_glm.py)
        │
        ├── StackedEnsemble (ensemble/stacking.py)
        │     └── MetaLearner (ensemble/meta_learner.py)
        │
        └── Leaderboard (reporting/leaderboard.py)
              └── MemoryReport (reporting/memory_report.py)
```

## Key Design Decisions

### 1. Non-negative Meta Learner

**문제**: cuML LogisticRegression은 non-negative weight constraint를 네이티브로 지원하지 않음.
H2O의 Non-negative GLM은 앙상블 안정성의 핵심.

| 순위 | 방안 | 선택 | 이유 |
|:----:|------|:----:|------|
| 1 | Post-hoc clipping + renormalization | **Primary** | GPU에서 완결, 구현 간단 |
| 2 | CuPy NNLS 직접 구현 | - | 구현 복잡도 대비 이점 불분명 |
| 3 | scipy.optimize.nnls CPU fallback | **Fallback** | Level-One Data가 작아 CPU 전송 영향 미미 |

### 2. cuML RF GPU max_depth 제한

**문제**: cuML RF는 GPU에서 `max_depth` 최대 16.
**대응**: 초과 시 16에서 자동 cap + warning 로그. 대부분의 tabular 태스크에서 depth 16이면 충분.

### 3. Memory-Aware vs Memory-Naive

| 모드 | 동작 | 용도 |
|------|------|------|
| `memory_aware=True` (기본) | 훈련 전 VRAM 추정, 부족 시 skip | OOM 방지 |
| `memory_aware=False` | VRAM 확인 없이 즉시 실행 | 벤치마크 baseline |

Orchestrator가 매 모델 훈련 전에 `VRAMEstimator.estimate()` → `profiler.get_free_vram_gb()` 비교를 수행.
실제 사용량은 `estimator.record_actual()`로 기록되어 추후 회귀 모델 학습에 사용.

---

## 실증 결과: 설계가 실제로 작동하는가?

Kaggle Credit Card Fraud (1,296,675 rows)로 검증한 결과, 위 설계가 모두 유효하게 동작합니다.

| 설계 결정 | 실증 결과 |
|----------|----------|
| H2O 3-Phase 전략 | Diversity Phase에서 Baseline 대비 AUC +0.27% 향상 (0.9953 → 0.9980) |
| Non-negative Meta Learner (clipping) | 실패 모델(GLM, AUC 0.52) 가중치를 0으로 자동 제거 |
| Two-Type Ensemble | All Models(0.9961) + Best of Family(0.9973) 모두 생성 성공 |
| Memory-Aware Scheduling | 10개 모델 모두 VRAM 체크 통과, OOM 0건 |
| cuML RF max_depth 16 cap | RF 모델 4개 모두 정상 훈련 (depth 8 ~ 16) |

상세 해석: [docs/03-results/benchmark_interpretation.md](../03-results/benchmark_interpretation.md)

---

## vLLM-inspired Paged Memory (Coarse-grained Paging)

### 왜 vLLM인가?

vLLM은 LLM 추론에서 KV Cache를 Page 단위로 관리하여 GPU 메모리 활용률을 96%+로 끌어올렸다.
AutoML도 동일한 문제를 갖고 있다 — 수십 개 모델이 제한된 GPU VRAM을 경쟁한다.

### vLLM → AutoML 매핑

| vLLM | AutoML (paged_automl) | 구현 파일 |
|------|---------------------|----------|
| KV Cache Block (16 tokens) | MemoryBlock (64MB) | `memory/paged_manager.py` |
| Block Table (논리 → 물리) | task_block_map (model → blocks) | `memory/paged_manager.py` |
| Block Manager | PagedMemoryManager | `memory/paged_manager.py` |
| Continuous Batching | ContinuousScheduler | `scheduler.py` |
| Swap GPU ↔ CPU | GPU ↔ Host pinned memory | `memory/paged_manager.py` |
| LRU Eviction | LRU 기반 블록 회수 | `memory/paged_manager.py` |
| Pre-allocated Pool | rmm Pool을 Block 단위 분할 | `memory/paged_manager.py` |

### 커스텀 CUDA 커널이 필요 없는 이유

```
vLLM:    연산 도중 (Attention 커널 안에서) Page Table 참조
         → 커스텀 CUDA 커널 필수 (비연속 메모리 직접 접근)

AutoML:  연산 사이 (task 경계에서) 블록 할당/회수
         → 커스텀 커널 불필요 (rmm + cupy로 충분)
```

vLLM은 token-level **fine-grained** paging이고,
AutoML은 task-level **coarse-grained** paging이다.
cuML/XGBoost는 블랙박스 라이브러리이므로 훈련 도중 메모리를 제어할 수 없다.
대신 **훈련 시작 전 블록 할당, 완료 후 즉시 회수**하는 방식으로
커스텀 커널 없이 동일한 효과를 달성한다.

### 동작 흐름

```
1. 초기화: GPU free VRAM → 64MB 블록으로 분할 (예: 2GB → 32 blocks)
2. Task 요청: 모델 훈련에 필요한 블록 수 예측 (VRAMEstimator)
3. 할당: free queue에서 블록 pop → task에 배정
4. 부족 시: LRU eviction → 가장 오래된 task의 블록을 Host로 swap
5. 실행: cuML/XGBoost가 할당된 블록 안에서 훈련
6. 완료: 블록 즉시 회수 → free queue로 반환
7. 다음 task: 회수된 블록으로 즉시 다음 모델 시작 (Continuous)
```

---

[Phase 2 README로 돌아가기](README.md) | [Main README](../../README.md)
