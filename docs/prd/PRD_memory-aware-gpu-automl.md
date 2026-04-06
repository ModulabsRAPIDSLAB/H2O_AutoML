# Memory-Aware GPU AutoML Framework PRD

> **Version**: 1.0
> **Created**: 2026-04-07
> **Status**: Draft
> **Scale Grade**: Startup (소규모 연구)

## 1. Overview

### 1.1 Problem Statement

GPU AutoML은 CPU 대비 10~40배 빠른 모델 탐색이 가능하지만, **GPU VRAM(16~24GB)이라는 메모리 제약**이 실질적 병목이다. 기존 GPU AutoML 프레임워크(AutoGluon+RAPIDS, TPOT+RAPIDS)는 "속도"에만 집중하고 메모리 관리는 "OOM 나면 줄이세요" 수준에 머물러 있다.

H2O AutoML은 10년간 검증된 Stacking/HPO 전략을 갖고 있지만 Java/CPU 기반이라 GPU 환경에서 직접 사용할 수 없다.

### 1.2 Goals

1. H2O AutoML의 검증된 설계 전략(Stacked Ensemble, HPO 순서, Two-type Ensemble)을 RAPIDS 생태계로 재구현
2. GPU 메모리 최적화(Memory-Aware Scheduling, rmm 풀 관리)를 프레임워크의 핵심 기능으로 내재화
3. Dask-CUDA 기반 병렬 훈련으로 멀티 GPU 환경 지원
4. 메모리 프로파일링 데이터 수집 및 최적화 효과 정량 검증

### 1.3 Non-Goals (Out of Scope)

- H2O Java 코드의 직접 포팅 또는 재사용
- CUDA C++ 커널 수준의 재작성
- 프로덕션급 웹 UI/대시보드
- 분산 클러스터(멀티 노드) 지원 (싱글 노드 멀티 GPU까지만)
- AutoML 외 도메인 (NLP, CV 등) 지원

### 1.4 Scope

| 포함 | 제외 |
|------|------|
| cuDF 기반 E2E GPU 데이터 파이프라인 | 전면적 CPU fallback 모드 (cuML 미지원 기능에 한해 제한적 fallback 허용) |
| cuML RF/GLM + XGBoost GPU 알고리즘 풀 | LightGBM, CatBoost 등 추가 알고리즘 |
| 5-fold CV + OOF 기반 Stacked Ensemble | 사용자 정의 CV 전략 |
| Dask-CUDA 병렬 훈련 + Memory-Aware Scheduling | Ray/Spark 연동 |
| rmm 기반 메모리 프로파일링 및 풀 최적화 | GPU 간 RDMA 통신 최적화 |
| Leaderboard + peak VRAM 메트릭 | 웹 기반 시각화 대시보드 |
| Tabular 데이터 (분류/회귀) | 시계열, 이미지, 텍스트 |

---

## 2. User Stories

### 2.1 연구자 (Primary User)

**US-001**: As a RAPIDS LAB 연구자, I want to GPU에서 H2O 수준의 Stacked Ensemble AutoML을 실행할 수 있어야 한다, so that CPU 대비 대폭 빠른 모델 탐색이 가능하다.

**US-002**: As a 연구자, I want to AutoML 파이프라인의 각 단계별 VRAM 사용량을 확인할 수 있어야 한다, so that 메모리 병목을 식별하고 최적화할 수 있다.

**US-003**: As a 연구자, I want to VRAM이 부족한 GPU(16GB)에서도 OOM 없이 AutoML을 완료할 수 있어야 한다, so that 다양한 GPU 환경에서 실험할 수 있다.

**US-004**: As a 연구자, I want to Memory-Naive vs Memory-Aware의 성능 차이를 벤치마크로 비교할 수 있어야 한다, so that 메모리 최적화의 효과를 논문에 정량적으로 제시할 수 있다.

### 2.2 Acceptance Criteria (Gherkin)

```
Scenario: E2E GPU AutoML 실행
  Given cuDF DataFrame으로 로드된 학습 데이터
  When AutoML(max_runtime_secs=300)을 실행하면
  Then 최소 3개 알고리즘(XGBoost, RF, GLM)의 base model이 훈련되고
  And Stacked Ensemble(All Models + Best of Family)이 생성되고
  And Leaderboard에 성능 메트릭 + peak_vram_gb이 표시된다

Scenario: Memory-Aware Scheduling
  Given VRAM 16GB GPU에서 100만 행 데이터로 AutoML 실행
  When 동시 실행 모델의 예상 VRAM 합이 가용 VRAM을 초과하면
  Then 스케줄러가 일부 모델을 대기열에 넣고 OOM 없이 완료된다
  And Memory-Naive 대비 OOM 발생률이 0%이다

Scenario: 메모리 프로파일링
  Given rmm logging이 활성화된 상태에서 AutoML 실행
  When 파이프라인이 완료되면
  Then 단계별(데이터 로드, CV 훈련, OOF 수집, Meta Learner) peak VRAM이 기록되고
  And 모델별 메모리 footprint가 수집된다
```

---

## 3. Functional Requirements

| ID | Requirement | Priority | Dependencies |
|----|------------|----------|--------------|
| **Data Layer** | | | |
| FR-001 | cuDF DataFrame으로 CSV/Parquet 데이터 로드 | P0 (Must) | - |
| FR-002 | GPU 메모리 위에서 K-fold CV 분할 수행 | P0 (Must) | FR-001 |
| FR-003 | Dask-cuDF로 VRAM 초과 데이터 분산 로드 | P2 (Could) | FR-001 |
| **Algorithm Pool** | | | |
| FR-010 | cuML RandomForestClassifier/Regressor 훈련 | P0 (Must) | FR-001 |
| FR-011 | XGBoost GPU (`tree_method='hist', device='cuda'`) 훈련 | P0 (Must) | FR-001 |
| FR-012 | cuML LogisticRegression/LinearRegression 훈련 | P0 (Must) | FR-001 |
| FR-013 | PyTorch MLP 기반 DNN 훈련 | P2 (Could) | FR-001 |
| **Cross-Validation & OOF** | | | |
| FR-020 | 각 base model에 대해 5-fold CV 수행 + OOF 예측 생성 | P0 (Must) | FR-010~012 |
| FR-021 | OOF 예측을 cuDF DataFrame으로 Level-One Data 구성 | P0 (Must) | FR-020 |
| **Stacked Ensemble** | | | |
| FR-030 | cuML GLM Meta Learner로 All Models Ensemble 생성 | P0 (Must) | FR-021 |
| FR-031 | Best of Family Ensemble 생성 (알고리즘별 best 1개씩) | P0 (Must) | FR-021 |
| FR-032 | Non-negative 가중치 제약 + L1 정규화 (sparse ensemble) — 아래 기술 스파이크 참고 | P0 (Must) | FR-030 |
| **Data Preprocessing** | | | |
| FR-004 | 기본 전처리 (결측치 처리, 카테고리 인코딩) | P1 (Should) | FR-001 |
| **Training Pipeline** | | | |
| FR-040 | H2O 훈련 순서 전략 (Baseline → Diversity → Random Search) | P1 (Should) | FR-010~012 |
| FR-041 | 시간 기반 제어 (`max_runtime_secs`) — 벤치마크 전제 조건 | P1 (Should) | - |
| FR-042 | Random Search 하이퍼파라미터 탐색 | P1 (Should) | FR-040 |
| FR-043 | XGBoost early stopping 지원 | P0 (Must) | FR-011 |
| FR-044 | cuML RF early stopping 대안 전략 (n_estimators 탐색 대체) | P1 (Should) | FR-010 |
| **Dask-CUDA 병렬화** | | | |
| FR-050 | LocalCUDACluster로 멀티 GPU worker 관리 | P0 (Must) | - |
| FR-051 | Fold 병렬화 (5-fold를 다수 worker에서 동시 실행) | P1 (Should) | FR-050 |
| FR-052 | 모델 간 병렬화 (서로 다른 모델을 동시 실행) | P1 (Should) | FR-050 |
| FR-053 | XGBoost Dask 분산 훈련 (멀티 GPU 단일 모델) | P2 (Could) | FR-050 |
| **Memory-Aware Scheduling** | | | |
| FR-060 | rmm logging으로 메모리 할당/해제 프로파일링 | P0 (Must) | FR-050 |
| FR-061 | 모델별 예상 VRAM 사용량 추정 함수 (Phase 2 프로파일 데이터 기반 회귀 모델, 입력: 행 수/특성 수/HP, 정확도 목표: 실제 +-20%) | P0 (Must) | FR-060 |
| FR-062 | 가용 VRAM 기반 task 스케줄링 (OOM 방지) | P0 (Must) | FR-061 |
| FR-063 | `device_memory_limit` 기반 spill-to-host 안전장치 | P1 (Should) | FR-050 |
| FR-064 | Streaming OOF (Level-One Data 메모리 최적화) | P1 (Should) | FR-021 |
| FR-065 | Adaptive Model Pool (VRAM 예산 기반 모델 수 동적 결정) | P2 (Could) | FR-062 |
| FR-066 | Mixed Precision Level-One Data (float16) | P2 (Could) | FR-021 |
| **Leaderboard & Metrics** | | | |
| FR-070 | 모델별 성능 메트릭 (AUC/RMSE, 훈련 시간) Leaderboard | P0 (Must) | FR-020 |
| FR-071 | 모델별 peak VRAM 사용량 Leaderboard에 포함 | P0 (Must) | FR-060 |
| FR-072 | 메모리 프로파일링 리포트 출력 (단계별 peak VRAM) | P1 (Should) | FR-060 |
| FR-073 | GPUTreeSHAP 기반 모델 설명 기능 | P2 (Could) | FR-010~012 |
| FR-074 | predict() API: 기본 Leaderboard 1위 모델 예측, model_id로 특정 모델 지정 가능 | P0 (Must) | FR-030 |

---

## 4. Non-Functional Requirements

### 4.0 Scale Grade

**Startup (소규모 연구)** — RAPIDS LAB 내부 연구 프로젝트, 2~3명 팀, 논문 발표 목표.

### 4.1 Performance

| 지표 | 목표값 | 비고 |
|------|--------|------|
| E2E 파이프라인 속도 | CPU H2O 대비 10배 이상 가속 | non-deterministic 기준 (deterministic 시 ~20% 저하 허용) |
| 최종 모델 정확도 | CPU H2O AutoML 대비 동등 이상 | 동일 시간 예산 기준 |
| 메모리 프로파일링 오버헤드 | 측정 필요 (rmm 공식 문서: "significant impact", 정량 수치 미공개) | rmm logging 활성화 시 |

### 4.2 Compatibility

| 항목 | 요구사항 |
|------|----------|
| Python | 3.9~3.11 |
| CUDA | 11.8+ |
| GPU | NVIDIA (Compute Capability 7.0+, VRAM 16GB+) |
| RAPIDS | 24.02+ (cuDF, cuML, Dask-CUDA, rmm) |
| XGBoost | 2.0+ (GPU 지원) |
| OS | Linux (Ubuntu 22.04+) |

### 4.3 Data Requirements

| 항목 | 값 |
|------|-----|
| 지원 데이터 형식 | CSV, Parquet |
| 지원 데이터 크기 | 단일 GPU VRAM 내 (기본), Dask-cuDF로 확장 가능 |
| 지원 태스크 | Binary Classification, Multiclass Classification, Regression |

### 4.4 Reproducibility

| 항목 | 요구사항 |
|------|----------|
| Random seed | 모든 알고리즘에 seed 고정 지원 |
| Deterministic mode | XGBoost `deterministic_histogram`, cuML `random_state` |
| 실험 기록 | Leaderboard + 메모리 프로파일을 파일로 저장 |

---

## 4.5 Benchmark Datasets

논문 실험의 재현성과 비교 가능성을 위해 사전 선정:

| Dataset | Rows | Features | Task | 비고 |
|---------|------|----------|------|------|
| Kaggle Credit Card Fraud | 1.29M | 23 | Binary Classification | 프로젝트에서 이미 사용 중 |
| Higgs Boson (UCI) | 11M | 28 | Binary Classification | 대규모, GPU 가속 효과 극대화 |
| Airline Delays (OpenML) | 5.8M | 13 | Binary Classification | 중규모, 카테고리 특성 포함 |

### 4.6 Terminology

| 용어 | 정의 |
|------|------|
| Memory-Naive Scheduling | VRAM 잔여량을 고려하지 않는 기본 스케줄링 (Dask 기본 동작) |
| Memory-Aware Scheduling | VRAM 예산 기반으로 task를 배정하는 스케줄링 (우리의 핵심 기여) |
| Level-One Data | base model들의 OOF 예측을 모은 N x L 행렬 (Stacking 입력) |
| OOF (Out-of-Fold) | 모델이 보지 못한 fold에 대한 예측값 |

---

## 5. Technical Design

### 5.0 Technical Spike: cuML GLM Non-negative 제약

**배경**: H2O Stacking의 Meta Learner는 non-negative GLM을 사용한다. 이는 앙상블 안정성의 핵심이다. cuML LogisticRegression이 `non_negative` 옵션을 직접 지원하지 않을 수 있으므로, **Phase 1 착수 전에 반드시 검증**해야 한다.

**검증 항목**:
1. `cuml.linear_model.LogisticRegression`에 non-negative constraint 파라미터 존재 여부
2. `cuml.linear_model.LinearRegression`의 `fit_intercept=False` + positive 제약 가능 여부

**대안 전략 (우선순위순)**:

| 순위 | 방안 | 장점 | 단점 |
|------|------|------|------|
| 1 | cuML GLM 학습 후 post-hoc 클리핑 + 재정규화 | GPU에서 완결, 구현 간단 | 수학적으로 제약 최적화와 다름 |
| 2 | CuPy 기반 NNLS (Non-Negative Least Squares) 직접 구현 | GPU 네이티브, 정확한 제약 최적화 | 구현 복잡도 |
| 3 | scipy.optimize.nnls CPU fallback | 정확한 NNLS | Level-One Data를 CPU로 전송해야 함 (크기 작아 영향 미미) |

**Acceptance Criteria**: 선택된 방안으로 H2O GLM meta learner와 동등한 가중치 분배 및 앙상블 정확도를 달성해야 한다.

### 5.1 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                 Memory-Aware GPU AutoML System                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Python API (sklearn 호환)                                 │  │
│  │  automl = GPUAutoML(max_runtime_secs=300)                  │  │
│  │  automl.fit(train, y='target')                             │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────────┐  │
│  │  Orchestrator                                              │  │
│  │  - H2O 전략: Baseline → Diversity → Random Search          │  │
│  │  - 시간 제어: max_runtime_secs                              │  │
│  │  - Memory-Aware Scheduler: VRAM 기반 task 배정             │  │
│  │  - Leaderboard 관리                                        │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────────┐  │
│  │  Dask-CUDA Cluster + rmm                                   │  │
│  │  - LocalCUDACluster (멀티 GPU worker)                      │  │
│  │  - device_memory_limit (spill 안전장치)                    │  │
│  │  - rmm pool + logging (메모리 프로파일링)                  │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────────┐  │
│  │  Algorithm Pool                                            │  │
│  │  cuML RF │ cuML GLM │ XGBoost GPU │ (PyTorch MLP)         │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
│  ┌────────────────────▼───────────────────────────────────────┐  │
│  │  Data Layer                                                │  │
│  │  cuDF DataFrame (단일 GPU) / Dask-cuDF (멀티 GPU)          │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Core API Design

```python
# 사용자 인터페이스 (sklearn 호환)
from gpu_automl import GPUAutoML

automl = GPUAutoML(
    max_runtime_secs=300,       # 시간 예산
    max_models=20,              # 최대 모델 수
    nfolds=5,                   # CV fold 수
    seed=42,                    # 재현성
    memory_aware=True,          # Memory-Aware Scheduling 활성화
    memory_profile=True,        # rmm 프로파일링 활성화
)

automl.fit(X_train, y_train)           # cuDF DataFrame 입력
lb = automl.leaderboard()              # 성능 + peak_vram 포함
profile = automl.memory_profile()      # 단계별 VRAM 리포트
preds = automl.predict(X_test)         # 앙상블 예측
```

### 5.3 Key Module Structure

```
gpu_automl/
├── __init__.py
├── automl.py              # GPUAutoML 메인 클래스
├── orchestrator.py        # 훈련 순서 / 시간 제어 / 스케줄링
├── scheduler.py           # Memory-Aware Scheduler
├── data/
│   ├── loader.py          # cuDF/Dask-cuDF 데이터 로드
│   └── cv.py              # K-fold CV 분할 + OOF 수집
├── models/
│   ├── base.py            # BaseModel 인터페이스
│   ├── xgboost_gpu.py     # XGBoost GPU wrapper
│   ├── cuml_rf.py         # cuML RF wrapper
│   ├── cuml_glm.py        # cuML GLM wrapper
│   └── pytorch_mlp.py     # PyTorch MLP wrapper (P2)
├── ensemble/
│   ├── stacking.py        # Stacked Ensemble (All Models + Best of Family)
│   └── meta_learner.py    # cuML GLM Meta Learner
├── memory/
│   ├── profiler.py        # rmm logging 기반 프로파일러
│   ├── estimator.py       # 모델별 VRAM 사용량 추정
│   └── pool.py            # rmm 풀 관리
├── hpo/
│   ├── random_search.py   # Random Search
│   └── presets.py         # Pre-specified 파라미터 셋
└── reporting/
    ├── leaderboard.py     # Leaderboard 생성
    └── memory_report.py   # 메모리 프로파일 리포트
```

---

## 6. Implementation Phases

### Phase 0: Technical Spike (Phase 1 착수 전 필수)

- [ ] cuML GLM non-negative 제약 지원 여부 코드 검증
- [ ] 미지원 시 대안 방안 선택 및 프로토타입 (Section 5.0 참고)
- [ ] cuML RF `max_depth` 제한 확인 및 대응 전략 결정

**Deliverable**: Non-negative Meta Learner 구현 방안 확정

### Phase 1: MVP — E2E GPU Stacking (단일 GPU, 순차, Dask 없음)

- [ ] cuDF 데이터 로드 + 기본 전처리 (`data/loader.py`)
- [ ] cuML RF, XGBoost GPU (early stopping 포함), cuML GLM 모델 wrapper (`models/`)
- [ ] 5-fold CV + OOF 생성 — 순차, 단일 GPU (`data/cv.py`)
- [ ] cuML GLM Meta Learner + Non-negative 제약 + Stacked Ensemble (`ensemble/`)
- [ ] Leaderboard 출력 (`reporting/leaderboard.py`)
- [ ] GPUAutoML 메인 클래스 (`automl.py`)

**Deliverable**: 단일 GPU에서 H2O와 동일한 Stacking이 돌아가는 MVP

### Phase 2: Dask-CUDA 병렬화 + 메모리 인프라 + 시간 제어

- [ ] LocalCUDACluster + rmm 풀 초기화
- [ ] 시간 기반 제어 (`max_runtime_secs`) — 벤치마크 전제 조건
- [ ] Fold 병렬화 (5-fold 동시 실행)
- [ ] 모델 간 병렬화 (RF, XGB, GLM 동시)
- [ ] rmm logging 기반 메모리 프로파일러 (`memory/profiler.py`)
- [ ] 모델별 VRAM 사용량 측정 + 프로파일 데이터 수집

**Deliverable**: 멀티 GPU 병렬 훈련 + 시간 제어 + 메모리 프로파일 데이터

### Phase 3: Memory-Aware Scheduling

- [ ] 모델별 VRAM 사용량 추정 함수 (`memory/estimator.py`)
- [ ] 가용 VRAM 기반 task 스케줄링 로직 (`scheduler.py`)
- [ ] Streaming OOF (Level-One Data GPU↔Host 이동 최적화)
- [ ] Adaptive Model Pool (VRAM 예산 기반 모델 수 자동 결정)
- [ ] Memory-Naive vs Memory-Aware 벤치마크

**Deliverable**: 메모리 최적화가 내재된 GPU AutoML + 비교 벤치마크 데이터

### Phase 4: H2O 전략 고도화 + 연구 확장

- [ ] 훈련 순서 전략 (Baseline → Diversity → Random Search)
- [ ] 시간 기반 제어 (`max_runtime_secs`)
- [ ] rmm 풀 전략 비교 실험 (No pool / Fixed / Managed / Adaptive)
- [ ] Mixed Precision Level-One Data 실험
- [ ] 다양한 데이터셋 벤치마크 (Kaggle 데이터셋 3~5개)

**Deliverable**: 논문용 실험 데이터 + 최종 프레임워크

---

## 7. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| E2E 속도 향상 | CPU H2O 대비 10x+ | 동일 데이터, 동일 모델 수, wall-clock 시간 비교 |
| 모델 정확도 | H2O 동등 이상 | 동일 시간 예산에서 best model AUC/RMSE 비교 |
| OOM 발생률 (Memory-Aware) | 0% | 16GB GPU에서 100만 행 데이터, 모델 20개 기준 |
| OOM 발생률 (Memory-Naive) | 측정 | 동일 조건에서 baseline 측정 |
| 메모리 프로파일 정밀도 | 실제 VRAM +-10% | rmm logging vs nvidia-smi 비교 |
| 지원 알고리즘 수 (Phase 1) | 3개 이상 | XGBoost, RF, GLM |
| Stacking 성능 | 개별 모델 대비 향상 | Ensemble AUC > Best single model AUC |

---

## 8. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| cuML API가 H2O 기능을 완전히 커버하지 못함 | 중간 | 중간 | cuML 문서 사전 조사, 부족 시 sklearn fallback |
| rmm 프로파일링 오버헤드가 예상보다 클 수 있음 | 낮음 | 낮음 | 프로파일링 on/off 옵션, 샘플링 모드 |
| Dask-CUDA worker 간 데이터 전송 오버헤드 | 중간 | 중간 | 데이터를 worker에 사전 배포, 전송 최소화 |
| GPU 환경 세팅 복잡성 (CUDA, RAPIDS 버전 호환) | 높음 | 높음 | Docker 이미지 제공, conda 환경 고정 |
| 메모리 추정 정확도가 낮을 수 있음 | 중간 | 중간 | 프로파일 데이터 기반 회귀 모델, 20% 마진 확보 |

---

## 9. Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| RAPIDS cuDF | 24.02+ | GPU DataFrame |
| RAPIDS cuML | 24.02+ | GPU ML 알고리즘 |
| RAPIDS Dask-CUDA | 24.02+ | GPU 분산 컴퓨팅 |
| RAPIDS rmm | 24.02+ | GPU 메모리 관리 |
| XGBoost | 2.0+ | Gradient Boosting (GPU) |
| PyTorch | 2.0+ | MLP (Phase 4, optional) |
| pandas | 1.3+ | Leaderboard 출력 |
| pynvml | 11.0+ | GPU 메모리 조회 |

---

## 10. References

- H2O AutoML 논문: "H2O AutoML: Scalable Automatic Machine Learning" (LeDell & Poirier, 2020)
- RAPIDS 공식 문서: https://docs.rapids.ai/
- Bergstra & Bengio (2012): "Random Search for Hyper-Parameter Optimization"
- 팀 분석 문서: `docs/assignment/01~05`
