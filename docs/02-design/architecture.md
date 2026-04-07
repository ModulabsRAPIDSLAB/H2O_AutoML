# Architecture: Research-to-Code Mapping

이 문서는 Phase 1(연구)에서 분석한 H2O 개념이 `gpu_automl/` 코드의 어느 모듈로 구현되었는지를 매핑합니다.

## H2O Concept → gpu_automl/ Module

| H2O 개념 | 연구 문서 | gpu_automl/ 모듈 | GPU 구현 방식 |
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

[Phase 2 README로 돌아가기](README.md) | [Main README](../../README.md)
