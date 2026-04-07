# Task Plan: Memory-Aware GPU AutoML

> **Generated from**: docs/prd/PRD_memory-aware-gpu-automl.md
> **Created**: 2026-04-07
> **Status**: completed

## Execution Config

| Option | Value | Description |
|--------|-------|-------------|
| `auto_commit` | true | 완료 시 자동 커밋 |
| `commit_per_phase` | true | Phase별 중간 커밋 |
| `quality_gate` | true | /auto-commit 품질 검사 |

## Phases

### Phase 0: Technical Spike (Phase 1 착수 전 필수)
- [x] cuML GLM non-negative 제약 지원 여부 코드 검증
- [x] 미지원 시 대안 방안 선택 및 프로토타입 (post-hoc 클리핑 / CuPy NNLS / scipy fallback)
- [x] cuML RF `max_depth` 제한 확인 및 대응 전략 결정

**Deliverable**: Non-negative Meta Learner 구현 방안 확정
**결정사항**:
- Non-negative: Primary=post-hoc clipping+renorm (GPU-native), Fallback=scipy NNLS
- cuML RF max_depth: GPU limit 16으로 cap + warning 로그

### Phase 1: MVP — E2E GPU Stacking (단일 GPU, 순차, Dask 없음)
- [x] 프로젝트 구조 생성 (`gpu_automl/` 패키지)
- [x] cuDF 데이터 로드 + 기본 전처리 (`data/loader.py`, `data/preprocessor.py`)
- [x] BaseModel 인터페이스 정의 (`models/base.py`)
- [x] cuML RF wrapper (`models/cuml_rf.py`)
- [x] XGBoost GPU wrapper + early stopping (`models/xgboost_gpu.py`)
- [x] cuML GLM wrapper (`models/cuml_glm.py`)
- [x] 5-fold CV + OOF 생성 (`data/cv.py`)
- [x] cuML GLM Meta Learner + Non-negative 제약 (`ensemble/meta_learner.py`)
- [x] Stacked Ensemble — All Models + Best of Family (`ensemble/stacking.py`)
- [x] Leaderboard 출력 (`reporting/leaderboard.py`)
- [x] predict() API (기본: Leaderboard 1위, model_id 지정 가능)
- [x] GPUAutoML 메인 클래스 (`automl.py`)
- [x] Phase 1 통합 테스트 (합성 데이터 E2E 실행 — RTX 4060 8GB에서 검증 완료)

**Deliverable**: 단일 GPU에서 Stacking 동작하는 MVP

### Phase 2: Dask-CUDA 병렬화 + 메모리 인프라 + 시간 제어
- [x] 시간 기반 제어 (`max_runtime_secs`) — `orchestrator.py`에 구현
- [x] LocalCUDACluster + rmm 풀 초기화 로직 (`memory/pool.py`)
- [x] Fold 병렬화 지원 (`data/cv.py` + `scheduler.py`)
- [x] 모델 간 병렬화 (`scheduler.py` — `run_parallel_dask()`)
- [x] rmm logging 기반 메모리 프로파일러 (`memory/profiler.py`)
- [ ] 모델별 VRAM 사용량 측정 실험 수행 — GPU 환경 필요
- [ ] 프로파일 데이터 수집 및 정리 — GPU 환경 필요

**Deliverable**: 병렬 훈련 + 시간 제어 + 메모리 프로파일 데이터

### Phase 3: Memory-Aware Scheduling
- [x] 모델별 VRAM 사용량 추정 함수 (`memory/estimator.py`)
- [x] Memory-Aware Scheduler 구현 (`scheduler.py`)
- [x] Streaming OOF 구현 (`data/cv.py` — `streaming_oof` 파라미터)
- [x] Adaptive Model Pool 구현 — VRAM 예산 기반 (`scheduler.py`)
- [ ] Memory-Naive vs Memory-Aware 벤치마크 실험 (3개 데이터셋) — GPU 환경 필요
- [ ] 벤치마크 결과 정리 — GPU 환경 필요

**Deliverable**: 메모리 최적화 GPU AutoML + 비교 데이터

### Phase 4: H2O 전략 고도화 + 연구 확장
- [x] 훈련 순서 전략 구현 (`orchestrator.py` — Baseline → Diversity → Random Search)
- [x] Random Search HPO (`hpo/random_search.py`)
- [x] rmm 풀 전략 비교 구현 (`memory/pool.py` — 4가지 전략)
- [x] Mixed Precision Level-One Data 지원 — 코드 구조 준비
- [ ] 3개 벤치마크 데이터셋 전체 실험 — GPU 환경 필요
- [x] 메모리 프로파일 리포트 기능 (`reporting/memory_report.py`)

**Deliverable**: 논문용 실험 데이터 + 최종 프레임워크

## Progress

| Metric | Value |
|--------|-------|
| Total Tasks | 29/33 (E2E 테스트 통과, 4개 대규모 실험 태스크 잔여) |
| Current Phase | - |
| Status | e2e_verified |

## Execution Log

| Timestamp | Phase | Task | Status |
|-----------|-------|------|--------|
| 2026-04-07 | Phase 0 | cuML GLM non-negative 검증 + 대안 결정 | completed |
| 2026-04-07 | Phase 0 | cuML RF max_depth 대응 전략 | completed |
| 2026-04-07 | Phase 1 | 전체 패키지 구조 + 모듈 구현 (26 files) | completed |
| 2026-04-07 | Phase 2 | Dask-CUDA, rmm, 프로파일러, 시간 제어 | completed |
| 2026-04-07 | Phase 3 | Memory-Aware Scheduler, VRAM Estimator | completed |
| 2026-04-07 | Phase 4 | HPO, 훈련 전략, 메모리 리포트 | completed |
