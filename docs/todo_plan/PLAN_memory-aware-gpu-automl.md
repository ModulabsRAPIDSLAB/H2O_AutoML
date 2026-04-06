# Task Plan: Memory-Aware GPU AutoML

> **Generated from**: docs/prd/PRD_memory-aware-gpu-automl.md
> **Created**: 2026-04-07
> **Status**: pending

## Execution Config

| Option | Value | Description |
|--------|-------|-------------|
| `auto_commit` | true | 완료 시 자동 커밋 |
| `commit_per_phase` | true | Phase별 중간 커밋 |
| `quality_gate` | true | /auto-commit 품질 검사 |

## Phases

### Phase 0: Technical Spike (Phase 1 착수 전 필수)
- [ ] cuML GLM non-negative 제약 지원 여부 코드 검증
- [ ] 미지원 시 대안 방안 선택 및 프로토타입 (post-hoc 클리핑 / CuPy NNLS / scipy fallback)
- [ ] cuML RF `max_depth` 제한 확인 및 대응 전략 결정

**Deliverable**: Non-negative Meta Learner 구현 방안 확정

### Phase 1: MVP — E2E GPU Stacking (단일 GPU, 순차, Dask 없음)
- [ ] 프로젝트 구조 생성 (`gpu_automl/` 패키지)
- [ ] cuDF 데이터 로드 + 기본 전처리 (`data/loader.py`)
- [ ] BaseModel 인터페이스 정의 (`models/base.py`)
- [ ] cuML RF wrapper (`models/cuml_rf.py`)
- [ ] XGBoost GPU wrapper + early stopping (`models/xgboost_gpu.py`)
- [ ] cuML GLM wrapper (`models/cuml_glm.py`)
- [ ] 5-fold CV + OOF 생성 (`data/cv.py`)
- [ ] cuML GLM Meta Learner + Non-negative 제약 (`ensemble/meta_learner.py`)
- [ ] Stacked Ensemble — All Models + Best of Family (`ensemble/stacking.py`)
- [ ] Leaderboard 출력 (`reporting/leaderboard.py`)
- [ ] predict() API (기본: Leaderboard 1위, model_id 지정 가능)
- [ ] GPUAutoML 메인 클래스 (`automl.py`)
- [ ] Phase 1 통합 테스트 (Credit Card Fraud 데이터셋으로 E2E 실행)

**Deliverable**: 단일 GPU에서 Stacking 동작하는 MVP

### Phase 2: Dask-CUDA 병렬화 + 메모리 인프라 + 시간 제어
- [ ] 시간 기반 제어 (`max_runtime_secs`) — 벤치마크 전제 조건
- [ ] LocalCUDACluster + rmm 풀 초기화 로직
- [ ] Fold 병렬화 (Dask worker에 fold 분배)
- [ ] 모델 간 병렬화 (서로 다른 모델 동시 실행)
- [ ] rmm logging 기반 메모리 프로파일러 (`memory/profiler.py`)
- [ ] 모델별 VRAM 사용량 측정 실험 수행
- [ ] 프로파일 데이터 수집 및 정리

**Deliverable**: 병렬 훈련 + 시간 제어 + 메모리 프로파일 데이터

### Phase 3: Memory-Aware Scheduling
- [ ] 모델별 VRAM 사용량 추정 함수 (`memory/estimator.py`) — 프로파일 데이터 기반 회귀 모델
- [ ] Memory-Aware Scheduler 구현 (`scheduler.py`)
- [ ] Streaming OOF 구현 (Level-One Data GPU↔Host 관리)
- [ ] Adaptive Model Pool 구현 (VRAM 예산 기반)
- [ ] Memory-Naive vs Memory-Aware 벤치마크 실험 (3개 데이터셋)
- [ ] 벤치마크 결과 정리

**Deliverable**: 메모리 최적화 GPU AutoML + 비교 데이터

### Phase 4: H2O 전략 고도화 + 연구 확장
- [ ] 훈련 순서 전략 구현 (`orchestrator.py`)
- [ ] Random Search HPO (`hpo/random_search.py`)
- [ ] rmm 풀 전략 비교 실험 (No pool / Fixed / Managed / Adaptive)
- [ ] Mixed Precision Level-One Data 실험
- [ ] 3개 벤치마크 데이터셋 전체 실험 (Credit Card Fraud, Higgs Boson, Airline Delays)
- [ ] 메모리 프로파일 리포트 기능 (`reporting/memory_report.py`)

**Deliverable**: 논문용 실험 데이터 + 최종 프레임워크

## Progress

| Metric | Value |
|--------|-------|
| Total Tasks | 0/33 |
| Current Phase | - |
| Status | pending |

## Execution Log

| Timestamp | Phase | Task | Status |
|-----------|-------|------|--------|
| - | - | - | - |
