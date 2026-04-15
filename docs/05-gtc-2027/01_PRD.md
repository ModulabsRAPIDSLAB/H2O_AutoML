# PRD: GPU-Native End-to-End AutoML on RAPIDS

> **프로젝트 코드명**: `paged-automl` (working title, 재명명 검토 대상)
> **문서 버전**: v1.0 (2026-04-14)
> **문서 상태**: Draft → Review
> **작성자**: 프로젝트 리드
> **검토 필요**: 박사급 공동연구자 2명
> **상위 문서**: [00_strategic_direction.md](00_strategic_direction.md)

---

## 0. 문서 메타데이터

| 항목 | 내용 |
|:-----|:-----|
| 프로젝트 시작 | 2026-04-14 |
| 목표 완료 | 2026-07-14 (3개월) |
| 최종 산출물 | GTC 2027 정규세션 talk + 기술 백서 + 오픈소스 코드 |
| 현재 코드베이스 | `src/paged_automl/` (기존 PagedAutoML 실험) |
| 대상 하드웨어 | NVIDIA H100 80GB × 5 (확보 예정) |
| 개발 언어 | Python 3.11+, CUDA 12.x |
| 핵심 의존성 | RAPIDS (cuDF 24.x, cuML 24.x), XGBoost GPU, Ray 2.x, ONNX Runtime GPU 또는 Triton Inference Server (추론 경로 설계 후 확정) |

---

## 1. Executive Summary

본 프로젝트는 NVIDIA RAPIDS 생태계에 현존하지 않는 **end-to-end 데이터 전처리 자동화**를 구현하여, GPU에 완전히 상주하는(GPU-resident) AutoML 파이프라인을 구축한다. 기존 AutoML 프레임워크(H2O, AutoGluon, TPOT, LightAutoML)는 풍부한 preprocessing 자동화를 제공하지만 CPU 중심이며, 이들이 GPU 가속 모델 훈련과 결합될 때 필연적으로 발생하는 CPU↔GPU 데이터 이동 비용이 전체 파이프라인의 병목이 된다.

본 프로젝트는 다음 세 가지를 결합한다.

1. **GPU-Native Preprocessing Operator 카탈로그** — cuDF/cuML 커널을 조합하여 결측치 처리, 인코딩, 스케일링, 피처 선택, 피처 엔지니어링을 모두 GPU에서 수행.
2. **프로세스 수준 메모리 관리** — Ray Actor 기반 격리로 블랙박스 GPU 라이브러리의 VRAM 충돌을 해소.
3. **ASHA 기반 OOM-Free HPO** — 나쁜 trial을 조기 종료하여 동일 자원에서 탐색 밀도를 극대화.

최종적으로 **최소 5개 공개 데이터셋**에서 baseline 4종 대비 E2E wall-clock 3배 이상 단축, 동등 이상의 모델 품질, H100 1→4 scaling efficiency 70% 이상을 달성하는 것을 목표로 한다.

---

## 2. Background & Motivation

### 2-1. 현재 AutoML 생태계의 지형

AutoML 프레임워크는 크게 두 진영으로 나뉜다.

**Type A — CPU 중심 E2E 프레임워크** (H2O AutoML, AutoGluon, TPOT, LightAutoML)
- 장점: 성숙한 preprocessing 파이프라인, 넓은 데이터 타입 지원, 풍부한 피처 엔지니어링
- 한계: GPU 가속이 부분적이거나(AutoGluon의 일부 모델), 없음(TPOT). 데이터가 커질수록 CPU 전처리가 병목.

**Type B — GPU 커널 라이브러리** (RAPIDS: cuDF, cuML)
- 장점: 개별 연산의 압도적 속도 (10~100배 가속)
- 한계: **AutoML 계층이 없음**. 사용자가 수동으로 파이프라인을 조립해야 함. 자동 전처리 없음.

### 2-2. 공백(Gap)의 정량화

RAPIDS 사용자가 실제로 AutoML을 쓰려면 다음 중 하나를 선택해야 한다.

1. **H2O/AutoGluon을 쓰고 GPU 일부 활용**: 전처리를 CPU에서 수행 → `pandas.DataFrame`을 `cudf.DataFrame`으로 전환 → GPU 모델 훈련. **매 HPO iteration마다 CPU↔GPU 전환 비용 발생.**
2. **수동 파이프라인 구축**: cuDF로 전처리를 직접 작성 → cuML로 훈련. **자동화 부재.**

두 경로 모두 "GPU를 쓰는 AutoML"이라는 사용자 기대를 충족하지 못한다.

### 2-3. 기술적 Gap의 구체적 예시

| 자동 전처리 기능 | H2O | AutoGluon | TPOT | LightAutoML | RAPIDS |
|:-----------------|:---:|:---------:|:----:|:-----------:|:------:|
| Missing value imputation (mean/median/mode) | ✓ | ✓ | ✓ | ✓ | 부분 (cuDF `.fillna`만) |
| Target encoding | ✓ | ✓ | 부분 | ✓ | **없음** |
| One-hot encoding | ✓ | ✓ | ✓ | ✓ | cuML preprocessing 존재 |
| Frequency encoding | ✓ | ✓ | ✗ | ✓ | **없음** (수동 구현) |
| Feature interaction (poly, cross) | 부분 | ✓ | ✓ | ✓ | **없음** |
| Datetime feature extraction | 부분 | ✓ | ✗ | ✓ | 부분 (cuDF dt accessor) |
| Outlier detection + handling | ✗ | ✓ | 부분 | ✓ | **없음** |
| Feature selection (mutual info, SHAP) | ✓ | 부분 | ✓ | ✓ | **없음** |
| Text-lite features (length, token count) | ✗ | ✓ | ✗ | ✓ | 부분 |
| Data type inference | ✓ | ✓ | 부분 | ✓ | **없음** |
| Automatic pipeline composition | ✓ | ✓ | ✓ | ✓ | **없음** |

> **상기 매트릭스는 초기 추정이며, §6-1 Month 1 Week 1-2 작업으로 정확한 해부를 수행하여 대체된다.**

### 2-4. Why Now

- **H100 NVLink 환경의 보급**: Multi-GPU 간 대역폭이 전처리 파이프라인을 분산 실행 가능한 수준으로 향상.
- **RAPIDS 24.x의 안정화**: cuDF가 pandas API 호환성을 대부분 확보하여 operator 포팅 비용 감소.
- **Ray 2.x의 성숙**: Fractional GPU, Object Store spilling이 프로덕션 수준으로 안정화.
- **AutoML 수요의 확장**: 금융·의료·제조 도메인에서 대용량 tabular 데이터에 대한 AutoML 수요 증가.

---

## 3. Goals

### 3-1. Primary Goal

**GTC 2027 정규세션(45분 talk) 제출 및 수락.**

수락 판단은 NVIDIA GTC reviewer의 "novel contribution + industry impact + RAPIDS ecosystem fit" 기준에 따른다.

### 3-2. 기술적 Goals

**G1. GPU-Native Preprocessing Operator 카탈로그 완성**
- 최소 12종의 자동 전처리 operator를 cuDF/cuML 커널로 구현
- 각 operator는 fit/transform 인터페이스를 따르며, cuDF DataFrame을 입출력
- CPU 경로 없이 GPU-resident 상태로 chain 가능

**G2. 자동 파이프라인 합성(Composition)**
- 입력 데이터셋의 타입·카디널리티·결측 패턴을 감지하여 operator sequence를 자동 생성
- 사용자 개입 없이 raw 데이터 → 학습 가능한 tensor 생성

**G3. E2E Throughput 목표**
- 5개 이상 공개 데이터셋에서 4종 baseline 대비 wall-clock **3배 이상 단축**
- 최종 모델 품질(primary metric) **동등 이상** 유지

**G4. Multi-GPU Scaling**
- H100 1→4 scaling efficiency **70% 이상**
- Ray Actor + Object Store 기반 데이터 공유로 zero-copy 활용

**G5. 재현 가능성**
- 모든 실험이 단일 커맨드로 재현 가능
- Docker 이미지 + 벤치마크 스크립트 공개

**G6. Inference Serving Interface (Framework 완결성)**
- 학습된 파이프라인(전처리 + 모델)을 **배포 가능한 아티팩트**로 직렬화
- Dual Runtime 설계: 학습은 GPU-resident cuDF/cuML, 추론은 배포 환경에 맞춰 ONNX 또는 Triton Inference Server 경로 선택 가능
- 전처리 operator의 추론 경로 보장 — `fit`에서 학습한 파라미터가 ONNX 그래프 또는 Triton Python backend에서 동일하게 재현
- 단일 샘플 및 배치 추론 latency 측정 포함

### 3-3. Non-Goals (명시적 제외)

- **Neural Network AutoML (NAS)**: 본 프로젝트는 tabular AutoML에 한정. NN 아키텍처 탐색은 범위 외.
- **Time-series / NLP / Vision AutoML**: 별도 데이터 특성이 필요하므로 제외. 단, text-lite 피처(길이, 토큰 수 등)는 tabular 보조 피처로 포함.
- **Reinforcement Learning 기반 AutoML**: 계산 비용 대비 기여도 불명확.
- **Custom CUDA 커널 작성**: cuDF/cuML이 제공하는 기존 커널의 조합만으로 operator를 구성. 신규 커널은 개발 비용 초과.
- **End-user GUI**: CLI + Python API에 한정.
- **Online / streaming AutoML**: 배치 학습만 지원. 추론은 batch + single-sample 동기식만 지원, streaming inference는 범위 외.
- **Sub-10ms latency 최적화**: 서빙 인터페이스는 제공하되, 초저지연 튜닝(TensorRT 최적화, kernel fusion 등)은 범위 외.

---

## 4. Target Audience & Success Criteria

### 4-1. Primary Audience

**NVIDIA GTC 2027 Reviewer 및 참석자**
- 관심사: RAPIDS 생태계 확장, 산업 적용 사례, GPU 활용률 극대화, 재현 가능한 벤치마크
- 판단 기준: novelty, technical depth, measurable impact, reproducibility

### 4-2. Secondary Audience

- **RAPIDS 오픈소스 커뮤니티**: cuDF/cuML 기여자, 사용자
- **기업 ML 엔지니어**: AutoGluon/H2O 사용자 중 GPU 전환을 고려하는 그룹
- **학계 연구자**: AutoML, HPO, 시스템 논문 저자

### 4-3. Success Metrics

| 계층 | 지표 | 목표 |
|:-----|:-----|:-----|
| 발표 | GTC 정규세션 수락 | 수락 |
| 기술 | E2E wall-clock 단축률 (vs best baseline) | ≥ 3× |
| 기술 | 최종 모델 품질 (primary metric) | baseline 대비 동등 이상 |
| 기술 | H100 1→4 scaling efficiency | ≥ 70% |
| 기술 | OOM-free rate (HPO 전체 trial 중) | ≥ 95% |
| 기술 | GPU 활용률 (pipeline 실행 중 평균) | ≥ 80% |
| 커뮤니티 | GitHub stars (공개 후 3개월) | ≥ 500 |
| 커뮤니티 | 재현 가능성 (외부 사용자 벤치마크 재현) | ≥ 1건 검증 |

---

## 5. Research Questions

PRD는 다음 연구 질문에 대한 답을 산출해야 한다.

**RQ1.** RAPIDS 커널을 조합하여 기존 AutoML 프레임워크의 자동 전처리 기능을 어느 수준까지 동등하게 구현할 수 있는가? (coverage + quality)

**RQ2.** GPU-resident 전처리 파이프라인이 CPU-GPU 하이브리드 대비 실제로 얻는 throughput 이득은 얼마이며, 그 이득이 데이터 크기·타입에 따라 어떻게 변하는가?

**RQ3.** 블랙박스 GPU 라이브러리(cuML, XGBoost GPU)의 내부 메모리 할당을 제어할 수 없는 제약 하에서, 프로세스 수준 격리가 제공할 수 있는 실효 메모리 활용률의 이론적·실증적 상한은 무엇인가?

**RQ4.** ASHA 기반 early stopping이 tabular AutoML 환경에서 최종 모델 품질에 미치는 영향은 무엇이며, Phase-wise 전략(Diversity vs Random Search)에 따라 그 영향이 어떻게 달라지는가?

**RQ5.** H100 NVLink 환경에서 preprocessing DAG를 멀티 GPU로 분산할 때 얻는 scaling efficiency와 병목 지점은 무엇인가?

---

## 6. Scope

### 6-1. In-Scope

**S1. Baseline 해부 (Month 1)**
- H2O AutoML, AutoGluon, TPOT, LightAutoML의 자동 전처리 파이프라인 소스 코드 분석
- 각 프레임워크가 수행하는 전처리 연산의 정확한 알고리즘·하이퍼파라미터 추출
- RAPIDS 커널로의 치환 가능성 매핑표 작성

**S2. GPU-Native Preprocessing Operator 구현**
- 최소 12종 operator (§8-2 상세)
- fit/transform 인터페이스 통일
- cuDF `DataFrame` / `Series` 입출력
- 직렬화(pickle + CUDA IPC) 지원

**S3. 자동 파이프라인 합성기**
- 데이터 특성 자동 감지 (column type, cardinality, missing rate, skewness)
- Rule-based composition (초기) → ML-based composition (확장 여유 시)
- Pipeline을 Ray Object Store에 저장하여 재사용 가능

**S4. 모델링 + HPO 통합**
- 모델: **cuML RF, cuML GLM, XGBoost GPU** (3종으로 확정. LightGBM/CatBoost GPU는 제외 — Ray 통합 복잡도 대비 이득 불명확)
- HPO: Ray Tune + ASHA
- Phase-wise 전략 (Baseline → Diversity → Random Search → Ensemble)

**S5. 메모리 관리 인프라**
- Ray Actor 프로세스 격리
- Fractional GPU 리소스 할당 (실측 프로파일 기반)
- Object Store 자동 spilling

**S6. 벤치마크 스위트**
- 최소 5개 공개 데이터셋 (§9-1 상세)
- 4종 baseline 대비 E2E 측정
- Ablation + Scaling study

**S7. Inference Serving Layer**
- 학습된 파이프라인을 직렬화하는 `Exporter` 인터페이스
- ONNX 경로: 지원 가능한 operator + 모델을 ONNX 그래프로 내보내기 (XGBoost, 선형 모델, 단순 tree 기반 operator)
- Triton 경로: ONNX 미지원 operator(예: TargetEncoder의 런타임 lookup)를 Triton Python backend로 서빙
- 추론 클라이언트 SDK (Python) + REST/gRPC 엔드포인트 예제
- Batch + single-sample 추론 latency 벤치마크

**S8. 문서화 및 발표 자료**
- 기술 백서 (**arXiv 제출 — 필수 deliverable**)
- GTC talk outline + 데모
- 오픈소스 리포지토리 (README, 튜토리얼, API 문서)

### 6-2. Out-of-Scope

- Type-B 데이터 (이미지, 텍스트 장문, 시계열)
- Neural Architecture Search
- Federated Learning
- Edge 배포 (모바일, 임베디드)
- Streaming / continuous 추론 파이프라인 (batch + single-sample 동기 추론만 지원)
- GUI / Web dashboard
- 다국어 지원 (영문만)

### 6-3. Future Work (범위 외로 명시)

- NN 기반 tabular 모델(TabNet, FT-Transformer) 통합
- 시계열 전용 operator 확장
- Cloud 배포 자동화 (AWS/GCP/Azure)
- Continual learning 지원

---

## 7. Baseline Systems — 상세 선정 근거

### 7-1. H2O AutoML

**선정 이유**
- 가장 널리 쓰이는 tabular AutoML. 산업 적용 레퍼런스가 풍부.
- Stacked Ensemble 전략의 원조.
- 현재 프로젝트의 출발점(`H2O_AutoML` 리포 명칭의 근원).

**분석 대상**
- `h2o.automl.H2OAutoML`의 preprocessing 파이프라인
- Missing value handling, categorical encoding (특히 target/frequency encoding)
- Column type inference

**벤치마크 기준 설정**
- H2O AutoML은 CPU-only로 실행 (GPU 미지원 기본)
- 동일 시간 예산(예: 10분)에서 비교

### 7-2. AutoGluon

**선정 이유**
- Amazon이 유지하는 최신 대형 AutoML 프레임워크
- Tabular에서 state-of-the-art 성능을 자주 기록
- 부분적 GPU 지원 (모델링 계층만)

**분석 대상**
- `autogluon.tabular.TabularPredictor`의 `feature_generator` 모듈
- Text/datetime 자동 처리 로직
- Stacking + bagging 전략

**벤치마크 기준 설정**
- AutoGluon을 CPU 전처리 + GPU 모델링(가능한 경우) 모드로 실행
- 동일 시간 예산

### 7-3. TPOT (Tree-based Pipeline Optimization Tool)

**선정 이유**
- 유전 알고리즘 기반 파이프라인 최적화의 대표 주자
- 학술적 인용도가 높음
- 파이프라인 composition의 baseline

**분석 대상**
- GP 기반 pipeline 구성 방식
- sklearn operator 합성 규칙
- 피처 선택 자동화

**벤치마크 기준 설정**
- TPOT은 CPU-only
- 유전 알고리즘 특성상 시간 예산을 동일하게 부여하되 generation 수 제한

### 7-4. LightAutoML

**선정 이유**
- Sberbank가 개발한 러시아권 AutoML, 비영어권 베이스라인
- 매우 풍부한 자동 피처 엔지니어링 (특히 datetime, categorical)
- 최근 활발히 유지되는 오픈소스

**분석 대상**
- `lightautoml.transformers` 모듈의 각 transformer 클래스
- Advanced categorical handling (특히 Target Encoding 변종)
- Automatic feature selection

**벤치마크 기준 설정**
- LightAutoML CPU-only 기본 모드
- 동일 시간 예산

### 7-5. 비교 매트릭스 템플릿 (Month 1 완성 대상)

각 baseline에 대해 다음 매트릭스를 채운다. (현재는 예시)

| 기능 카테고리 | H2O | AutoGluon | TPOT | LightAutoML | RAPIDS 치환 가능성 | 우리 구현 우선순위 |
|:-------------|:---:|:---------:|:----:|:-----------:|:-------------------|:-------------------|
| Numerical imputation | | | | | | |
| Categorical imputation | | | | | | |
| One-hot encoding | | | | | | |
| Target encoding | | | | | | |
| Frequency encoding | | | | | | |
| Ordinal encoding | | | | | | |
| Scaling (standard/minmax/robust) | | | | | | |
| Outlier detection | | | | | | |
| Datetime features | | | | | | |
| Text-lite features | | | | | | |
| Polynomial features | | | | | | |
| Feature interaction | | | | | | |
| Feature selection (filter) | | | | | | |
| Feature selection (wrapper) | | | | | | |
| Column type inference | | | | | | |
| Class imbalance handling | | | | | | |
| Pipeline composition rule | | | | | | |

---

## 8. Technical Approach

### 8-1. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    paged-automl E2E Pipeline                       │
│                                                                    │
│  ┌──────────────┐                                                 │
│  │ Raw Dataset   │  (Parquet/CSV on disk or S3)                   │
│  └──────┬───────┘                                                 │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  Layer 1: Data Ingestion (cuDF / Ray Data)          │         │
│  │  - GPU-native Parquet reader                         │         │
│  │  - Column type auto-inference                        │         │
│  │  - Partition planning                                │         │
│  └──────┬───────────────────────────────────────────────┘         │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  Layer 2: Auto Preprocessing (★ 핵심 기여)           │         │
│  │  ┌──────────────┐  ┌──────────────┐                  │         │
│  │  │ DataProfiler │─▶│ PipelineComposer│              │         │
│  │  │ (col type,   │  │ (rule-based    │              │         │
│  │  │  cardinality,│  │  operator      │              │         │
│  │  │  missing)    │  │  sequence)     │              │         │
│  │  └──────────────┘  └──────┬───────┘                  │         │
│  │                           ▼                           │         │
│  │  ┌────────────────────────────────────────────┐     │         │
│  │  │  GPU-Native Operator Catalog                │     │         │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │     │         │
│  │  │  │ Imputer  │ │ Encoder  │ │ Scaler   │   │     │         │
│  │  │  └──────────┘ └──────────┘ └──────────┘   │     │         │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │     │         │
│  │  │  │ Outlier  │ │ Datetime │ │ FeatSel  │   │     │         │
│  │  │  └──────────┘ └──────────┘ └──────────┘   │     │         │
│  │  └────────────────────────────────────────────┘     │         │
│  └──────┬───────────────────────────────────────────────┘         │
│         │ (cuDF DataFrame, GPU-resident)                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  Layer 3: Model Zoo + HPO (Ray Tune + ASHA)          │         │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐              │         │
│  │  │ cuML RF  │ │ cuML GLM │ │ XGB GPU  │              │         │
│  │  └──────────┘ └──────────┘ └──────────┘              │         │
│  │   ↑ Phase A (Baseline) → Phase B (Diversity)        │         │
│  │      → Phase C (Random Search + ASHA) → Phase D      │         │
│  │         (Stacked Ensemble)                           │         │
│  └──────┬───────────────────────────────────────────────┘         │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  Layer 4: Leaderboard + Export                       │         │
│  │  - Best model selection                              │         │
│  │  - Pipeline serialization (dual-runtime artifact)    │         │
│  └──────┬───────────────────────────────────────────────┘         │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  Layer 5: Inference Serving (★ Framework 완결성)     │         │
│  │  ┌──────────────────┐    ┌──────────────────────┐   │         │
│  │  │  ONNX Runtime    │    │  Triton Inference    │   │         │
│  │  │  (portable path) │ OR │  Server (GPU-native) │   │         │
│  │  └──────────────────┘    └──────────────────────┘   │         │
│  │  - Batch 추론 + single-sample 추론                   │         │
│  │  - REST / gRPC 엔드포인트 예제                        │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                    │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  Infrastructure Layer (Ray)                          │         │
│  │  - Actor 프로세스 격리                                │         │
│  │  - Object Store (zero-copy, auto-spilling)           │         │
│  │  - Placement Group (multi-GPU scheduling)            │         │
│  └──────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

### 8-2. Preprocessing Operator Catalog (핵심 기여)

각 operator는 다음 인터페이스를 따른다.

```python
class GPUOperator(ABC):
    def fit(self, X: cudf.DataFrame, y: Optional[cudf.Series] = None) -> 'GPUOperator': ...
    def transform(self, X: cudf.DataFrame) -> cudf.DataFrame: ...
    def fit_transform(self, X, y=None) -> cudf.DataFrame: ...
    def get_params(self) -> dict: ...
    def set_params(self, **params) -> 'GPUOperator': ...
    def to_cpu(self) -> 'CPUOperatorMirror': ...  # 추론 서빙용
```

**필수 operator 목록 (우선순위 순)**

| Priority | Operator | 기반 커널 | 난이도 | 공백 여부 |
|:--------:|:---------|:----------|:-------|:---------|
| P0 | `NumericImputer` (mean/median/constant) | cuDF fillna | 낮음 | 부분 공백 |
| P0 | `CategoricalImputer` (mode/constant) | cuDF fillna | 낮음 | 부분 공백 |
| P0 | `StandardScaler` | cuML preprocessing | 낮음 | 존재 |
| P0 | `OneHotEncoder` | cuML preprocessing | 낮음 | 존재 |
| P0 | `OrdinalEncoder` | cuDF categorical | 낮음 | 부분 공백 |
| P0 | `TargetEncoder` | 커스텀 cuDF groupby agg | 중간 | **완전 공백** |
| P1 | `FrequencyEncoder` | cuDF groupby count | 낮음 | **완전 공백** |
| P1 | `DatetimeFeatureExtractor` (year/month/day/dow/hour) | cuDF dt accessor | 낮음 | 부분 공백 |
| P1 | `OutlierCapper` (IQR/zscore) | cuDF quantile + clip | 중간 | **완전 공백** |
| P1 | `RobustScaler` | cuDF quantile | 낮음 | 존재 |
| P1 | `MinMaxScaler` | cuML preprocessing | 낮음 | 존재 |
| P2 | `FeatureSelector` (mutual info) | 커스텀 GPU 구현 | 높음 | **완전 공백** |
| P2 | `FeatureSelector` (variance threshold) | cuDF variance | 낮음 | 부분 공백 |
| P2 | `PolynomialFeatures` (degree 2) | cuDF pairwise ops | 중간 | **완전 공백** |
| P2 | `FeatureInteraction` (pairwise cross) | cuDF pairwise | 중간 | **완전 공백** |
| P2 | `ClassBalancer` (SMOTE-GPU / undersample) | cuML neighbors + sampling | 높음 | **완전 공백** |
| P3 | `TextLiteFeatures` (length, token count, digit ratio) | cuDF string ops | 중간 | 부분 공백 |
| P3 | `ColumnTypeInferrer` | 규칙 + cuDF dtype | 중간 | **완전 공백** |

**P0 = Month 2 Week 1, P1 = Week 2, P2 = Week 3, P3 = Week 4.**

**Critical operator 상세**: `TargetEncoder`

- Target encoding은 leak-prone이며 올바른 cross-validation-aware 구현이 요구됨
- 기존 RAPIDS에 없는 대표 공백. 구현 품질이 전체 프로젝트의 설득력을 좌우.
- 참고: `category_encoders.TargetEncoder` + sklearn CV 로직을 cuDF groupby로 포팅.

### 8-3. Pipeline Composition Logic

**Rule-based composer (Month 1 설계, Month 2 구현)**

입력: `DataProfiler`가 산출한 column metadata.
출력: `List[GPUOperator]` (순서 있는 파이프라인).

규칙 예시 (초안):

```
FOR each column:
    IF type == numeric AND missing_rate > 0:
        APPEND NumericImputer(strategy=auto_choose(skewness))

    IF type == categorical AND cardinality < 10:
        APPEND OneHotEncoder

    IF type == categorical AND 10 <= cardinality < 100:
        APPEND TargetEncoder (if supervised) OR OrdinalEncoder

    IF type == categorical AND cardinality >= 100:
        APPEND FrequencyEncoder + TargetEncoder

    IF type == datetime:
        APPEND DatetimeFeatureExtractor

    IF type == numeric AND abs(skewness) > 2:
        APPEND OutlierCapper(method='iqr')

AFTER per-column:
    APPEND StandardScaler (on all remaining numeric)
    APPEND FeatureSelector(variance_threshold) (if n_cols > 100)
    IF classification AND imbalance_ratio > 5:
        APPEND ClassBalancer
```

**Meta-learned composer는 Non-Goal. 단, future work로 명시.**

### 8-4. 모델링 + HPO

기존 `docs/04-future/02_ray_process_level_paging.md`와 `03_hyperband_asha_oom_prevention.md`에 정의된 Ray Actor + ASHA 설계를 그대로 사용한다. 본 프로젝트의 **주연이 아니므로 추가 설계 작업 최소화**.

- Phase A (Baseline): 3개 알고리즘 × 기본 HP 1개 = 3 모델
- Phase B (Diversity): ASHA grace_period=5, reduction_factor=3
- Phase C (Random Search): ASHA grace_period=1, reduction_factor=4
- Phase D (Ensemble): Stacked Ensemble + Rank-averaging

### 8-5. 메모리 관리 인프라

`docs/04-future/02_ray_process_level_paging.md` §8의 설계를 채택한다. 핵심 결정사항:

- 각 GPU trainer는 `@ray.remote` Actor로 감싼다
- `num_gpus` 값은 사전 프로파일링으로 확정 (Month 2 Week 1)
- CUDA context 오버헤드 실측을 Month 2 Week 1에 선행
- `paged_manager.py`는 **"실패한 실험" 레퍼런스로 리포에 보존**하되 main path에서 제거

### 8-6. Inference Serving Architecture (Dual Runtime)

학습과 추론은 서로 다른 runtime 제약을 가진다. 학습은 대규모 GPU-resident 데이터를 전제로 하지만, 추론은 포터빌리티·배포 환경·latency 요구사항이 다양하다. 따라서 **Dual Runtime** 설계를 채택한다.

#### 8-6-1. 런타임 분리

```
┌──────────────────── Train-time Runtime ───────────────────┐
│   cuDF DataFrame / cuML / XGBoost GPU                      │
│   Ray Actor 프로세스 격리                                   │
│   GPU-resident 전체 파이프라인                              │
└────────────────┬──────────────────────────────────────────┘
                 │ Exporter interface
                 ▼
┌──────────────── Portable Artifact (중간 표현) ─────────────┐
│   - Operator 파라미터 (fit 결과: mean, std, category map..) │
│   - Model weights (XGBoost → JSON, cuML → ONNX/pickle)     │
│   - Pipeline graph (operator 순서 + 입력/출력 스키마)        │
└────────────────┬──────────────────────────────────────────┘
                 │ 배포 경로 선택 (사용자/상황 의존)
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│ ONNX Runtime │  │ Triton Inference │
│ (portable)   │  │ Server (GPU)     │
└──────────────┘  └──────────────────┘
```

#### 8-6-2. 경로 선택 기준 (Design Decision Pending)

| 기준 | ONNX Runtime 경로 | Triton Server 경로 |
|:-----|:------------------|:-------------------|
| 포터빌리티 | 매우 높음 (CPU/GPU/모바일) | GPU 서버 한정 |
| 지원 operator | 제한적 (ONNX opset 의존) | 제한 없음 (Python backend) |
| Latency | 낮음 (정적 그래프) | 중간 (Python overhead) |
| 배포 복잡도 | 낮음 (단일 파일) | 높음 (서버 구성 필요) |
| 멀티 모델 서빙 | 수동 orchestration | 기본 지원 |
| 커스텀 GPU operator | 불가 | 가능 (cuDF/cuML 직접 호출) |

**결정 시점**: Month 2 Week 1 설계 워크샵에서 확정. 결정 문서는 `06_serving_architecture_decision.md`(신규)로 산출.

**잠정 방향**: 두 경로 모두 지원하되 **Triton을 default**로 설정. 이유 — 우리의 핵심 operator(`TargetEncoder`, `FrequencyEncoder`, `OutlierCapper` 등)가 런타임 lookup/groupby를 필요로 하여 ONNX opset으로 완전 표현이 어려움. 순수 수치 파이프라인(numeric-only)에 한해 ONNX fast path 제공.

#### 8-6-3. Operator별 추론 경로 요건

각 operator는 다음 3개 인터페이스 중 최소 2개를 구현한다:

```python
class GPUOperator(ABC):
    # 학습 경로
    def fit(self, X: cudf.DataFrame, y=None) -> 'GPUOperator': ...
    def transform(self, X: cudf.DataFrame) -> cudf.DataFrame: ...

    # 추론 경로 — ONNX 내보내기 (가능한 경우)
    def to_onnx(self, opset: int = 17) -> Optional[onnx.ModelProto]: ...

    # 추론 경로 — Triton Python backend 호환
    def to_triton_python(self) -> 'TritonPythonModel': ...
```

`to_onnx`가 `None`을 반환하면 해당 operator는 ONNX 경로에서 배제되며, Triton 경로만 사용 가능함을 의미한다.

#### 8-6-4. 추론 품질 보증

**학습-추론 일관성 테스트(Consistency Test)**를 필수 CI에 포함:

- 동일 입력에 대해 학습 경로(`cudf.transform`)와 추론 경로(ONNX 또는 Triton)의 출력이 수치적으로 일치해야 함 (rtol=1e-5)
- 모든 operator × 두 경로 조합에 대해 회귀 테스트 실행

### 8-7. 장애 허용성 (Fault Tolerance)

- Ray Actor 실패 시 자동 재시작 (`max_actor_restarts=2`)
- HPO trial OOM 시 ASHA가 자동 early-stop
- 파이프라인 operator 실패 시 fallback operator로 대체 (예: `TargetEncoder` 실패 → `OrdinalEncoder`)
- 체크포인트: Phase 완료 시마다 Object Store + 디스크 저장

---

## 9. Evaluation Methodology

### 9-1. 데이터셋

**최소 5개, 목표 8개**. 다양성 확보 필수.

| 데이터셋 | 소스 | 행 수 | 열 수 | 타입 특성 | 과업 |
|:---------|:-----|------:|------:|:----------|:-----|
| Higgs (5M/11M) | UCI / Kaggle | 5,000,000 / 11,000,000 | 28 | 모두 numeric | Binary classification |
| Credit Card Fraud | Kaggle | 284,807 | 30 | Numeric, imbalanced | Binary classification |
| KDD Cup 1999 | UCI | 4,898,431 | 41 | Mixed (numeric + categorical) | Multi-class |
| Adult Census | UCI | 48,842 | 14 | Mixed, missing | Binary classification |
| Airlines (OpenML) | OpenML | 539,383 | 8 | Mixed + datetime | Binary classification |
| Porto Seguro | Kaggle | 595,212 | 59 | Heavy categorical, imbalanced | Binary classification |
| NYC Taxi (sampled) | TLC Open Data | 10,000,000 | 18 | Datetime + numeric | Regression |
| Santander Customer Transaction | Kaggle | 200,000 | 200 | Numeric, feature-rich | Binary classification |

**선정 원칙**
- 행 수 다양성: 소(5만) ~ 대(1000만+)
- 열 수 다양성: 소(10개) ~ 대(200개+)
- 타입 다양성: numeric-only, mixed, categorical-heavy, datetime 포함
- 공개 접근성: 재현 가능해야 함
- 과업 다양성: classification + regression + imbalanced

### 9-2. 메트릭

**Primary Metrics (모델 품질)**
- Binary classification: AUC-ROC, AUC-PR
- Multi-class: Macro F1, Accuracy
- Regression: RMSE, R²

**System Metrics**
- E2E wall-clock time (data load → best model selection)
- Peak VRAM usage (GPU별)
- Average GPU utilization (%)
- OOM event count
- HPO trial completion rate

**Scaling Metrics**
- Speedup (1 GPU → N GPU)
- Scaling efficiency = Speedup(N) / N × 100%
- Per-phase wall-clock breakdown

### 9-3. Ablation Study

다음 컴포넌트를 각각 끄고 켜면서 기여도 분리.

| Ablation | 설명 |
|:---------|:-----|
| `-preprocessing` | 전처리를 CPU(pandas/sklearn)로 수행 → GPU 모델링 |
| `-auto_composition` | 전처리 operator를 수동 지정 (고정 파이프라인) |
| `-ray_isolation` | 단일 프로세스에서 실행 (기존 `paged_manager` 방식) |
| `-asha` | Random Search (full budget for every trial) |
| `-fractional_gpu` | 전체 GPU 할당만 사용 |
| `-object_store_sharing` | ray.put 없이 매번 데이터 재로드 |

각 ablation에서 E2E wall-clock과 모델 품질을 측정.

### 9-4. Scaling Study

H100 1 → 2 → 4 → 5장에서 동일 워크로드 실행.

**측정 포인트**
- Strong scaling: 동일 데이터 × GPU 수 증가 → speedup
- Weak scaling: 데이터 크기 × GPU 수 비례 증가 → 시간 일정성
- Per-phase scaling: 전처리 / HPO / 앙상블 각각의 scaling 특성

**예상 병목**
- 전처리: NVLink 대역폭
- HPO: Object Store 경합
- 앙상블: Meta learner의 직렬 특성

### 9-5. 재현성 보장

- Docker 이미지 (CUDA 12.x, RAPIDS 24.x)
- `make benchmark` 단일 명령으로 전체 결과 재생성
- 고정 seed (`--seed 42`)
- 랜덤성이 있는 연산은 3회 반복 후 평균 ± 표준편차 보고

---

## 10. Milestones (상세)

### Month 1 (2026-04-14 ~ 2026-05-14): Gap Analysis + Design

**Week 1 (04-14 ~ 04-21)**
- [ ] H2O AutoML preprocessing 파이프라인 전수 해부 (담당: 리드)
- [ ] AutoGluon feature_generator 모듈 분석 (담당: 공동연구자 A)
- [ ] Docker + H100 환경 확보 + Ray 클러스터 셋업 (담당: 공동연구자 B)

**Week 2 (04-21 ~ 04-28)**
- [ ] TPOT GP pipeline 합성 로직 분석 (담당: 리드)
- [ ] LightAutoML transformers 모듈 분석 (담당: 공동연구자 A)
- [ ] `02_preprocessing_gap_analysis.md` 초안 완성
- [ ] CPU↔GPU pingpong 비용 실측 (Higgs 데이터, 5개 시나리오)

**Week 3 (04-28 ~ 05-05)**
- [ ] Operator 카탈로그 확정 (§8-2 테이블 최종화)
- [ ] Pipeline composition 규칙 초안 (§8-3)
- [ ] `03_gpu_native_operators_design.md` 작성
- [ ] 현재 `paged_manager.py` 실험 결과를 정량 데이터로 정리

**Week 4 (05-05 ~ 05-14)**
- [ ] 아키텍처 리뷰 (팀 내부)
- [ ] GTC Proposal Abstract 초안 작성
- [ ] Ray Actor + CUDA context 오버헤드 실측
- [ ] Sprint retrospective + Month 2 상세 계획 확정

### Month 2 (2026-05-14 ~ 2026-06-14): Core Implementation + 스폰서 데모

**Week 1 (05-14 ~ 05-21)**
- [ ] P0 operator 구현 (NumericImputer, CategoricalImputer, StandardScaler, OneHot, Ordinal, TargetEncoder)
- [ ] 단위 테스트 (vs sklearn equivalent 결과 비교)
- [ ] CI 셋업 (GitHub Actions + GPU runner)
- [ ] **추론 경로 설계 워크샵** — ONNX vs Triton 최종 결정 → `06_serving_architecture_decision.md` 산출
- [ ] Ray Actor + ASHA 통합 (기존 `hpo/` 코드 리팩토링)

**Week 2 (05-21 ~ 05-28) — 🚩 스폰서 데모 주간**
- [ ] P1 operator 구현 (Frequency, Datetime, Outlier, Robust/MinMax Scaler)
- [ ] DataProfiler + PipelineComposer (rule-based) 구현
- [ ] **첫 E2E 스모크 테스트 (Credit Card 데이터)** — 단일 GPU(로컬 또는 저가 렌탈 A10/L4)에서 학습→추론 end-to-end 동작
- [ ] **스폰서십 데모 패키지**:
  - [ ] 5분 데모 영상 (실제 파이프라인 실행 캡처)
  - [ ] 1페이지 제안서 (성과 예상 + 하드웨어 사용 계획)
  - [ ] 재현 가능한 데모 스크립트 (`make demo`)
- [ ] **Gate Review: 스폰서 컨택 → H100 렌탈 크레딧 확보**

**Week 3 (05-28 ~ 06-04)**
- [ ] P2 operator 구현 (FeatureSelector variants, Polynomial, Interaction, ClassBalancer)
- [ ] **Inference Serving Layer 구현 (first cut)**
  - [ ] `Exporter` 인터페이스 + Pipeline 직렬화 포맷 확정
  - [ ] 선택된 경로(ONNX 또는 Triton) 기반 최소 동작 서버
  - [ ] 학습-추론 일관성 테스트 CI 통합
- [ ] 2개 데이터셋 E2E 파이프라인 확인 (Credit Card, Adult)

**Week 4 (06-04 ~ 06-14)**
- [ ] P3 operator 구현 (TextLite, ColumnTypeInferrer)
- [ ] 추론 경로의 나머지 operator 지원 확대
- [ ] 추론 latency 측정 (batch + single-sample)
- [ ] 성능 프로파일링 + 병목 식별
- [ ] H100 환경 셋업 및 smoke test (확보 성공 가정)

### Month 3 (2026-06-14 ~ 2026-07-14): Benchmark + arXiv + GTC

**Week 1 (06-14 ~ 06-21)**
- [ ] 5개 데이터셋 전체 벤치마크 (single H100)
- [ ] 4종 baseline 벤치마크 실행 (동일 환경)
- [ ] 추론 serving latency 벤치마크 포함
- [ ] 초기 결과 분석

**Week 2 (06-21 ~ 06-28)**
- [ ] Ablation study (6개 조합 × 5개 데이터셋)
- [ ] Multi-GPU scaling study (H100 1/2/4/5장)
- [ ] `03-results/gtc_benchmark_report.md` 작성
- [ ] arXiv 초안 구조 확정 (섹션 배분, figure 목록)

**Week 3 (06-28 ~ 07-05)**
- [ ] **arXiv 기술 백서 초안 완성 (필수 deliverable)**
- [ ] GTC talk outline + 슬라이드 구조
- [ ] 데모 영상 스크립트 (학습 + 추론 end-to-end)
- [ ] 공동저자 내부 리뷰 1차

**Week 4 (07-05 ~ 07-14)**
- [ ] arXiv 백서 최종 수정 및 제출
- [ ] GTC proposal 최종 제출
- [ ] 팀 내부 리허설 (2회 이상)
- [ ] 오픈소스 리포 공개 준비 (README, LICENSE, CONTRIBUTING)

### 주요 Gate Review

- **2026-05-14 (Month 1 종료)**: Gap analysis 결과가 "충분한 novelty"를 확보했는지 팀 판단. Go/No-go.
- **🚩 2026-05-28 (Month 2 Week 2 종료) — 스폰서 데모 Gate**: 데모 동작 확인 + 스폰서 컨택 완료. H100 렌탈 크레딧 확보 여부가 Month 3 benchmark 규모를 결정.
- **2026-06-14 (Month 2 종료)**: 핵심 구현이 최소 3개 데이터셋에서 동작 + 추론 서빙 first cut 동작. 기술 실현성 Gate.
- **2026-07-07 (Month 3 Week 3)**: 벤치마크 결과가 §4-3 Success Metrics를 충족하는지 확인. 발표 수준 Gate.

---

## 11. Deliverables

### 11-1. 코드 산출물

- `src/paged_automl/` 리팩토링 완료 버전 (GPU-Native E2E 파이프라인)
  - `preprocessing/` (신규, 핵심 기여)
  - `models/` (기존 유지, LightGBM 제외 — cuML RF/GLM + XGBoost GPU 3종)
  - `hpo/` (Ray Tune + ASHA 통합)
  - `memory/` (Ray 기반으로 재작성, 기존 `paged_manager.py`는 `legacy/` 이동)
  - `serving/` (신규 — Exporter, ONNX/Triton backend, 추론 클라이언트 SDK)
- `benchmarks/` (벤치마크 스크립트)
- `examples/` (튜토리얼 notebooks — 학습 + 추론 서빙)
- `demo/` (스폰서 데모 패키지 — `make demo` 단일 명령)

### 11-2. 문서 산출물

- `docs/05-gtc-2027/00_strategic_direction.md` (완료)
- `docs/05-gtc-2027/01_PRD.md` (본 문서)
- `docs/05-gtc-2027/02_preprocessing_gap_analysis.md` (Month 1 완성)
- `docs/05-gtc-2027/03_gpu_native_operators_design.md` (Month 1 완성)
- `docs/05-gtc-2027/04_benchmark_methodology.md` (Month 2 완성)
- `docs/05-gtc-2027/05_scaling_study_design.md` (Month 2 완성)
- `docs/05-gtc-2027/06_serving_architecture_decision.md` (Month 2 Week 1 완성 — ONNX vs Triton 결정)
- `docs/05-gtc-2027/07_gtc_talk_outline.md` (Month 3 완성)
- `docs/05-gtc-2027/08_arxiv_manuscript.md` 또는 `arxiv/` 하위 LaTeX 소스 (Month 3 Week 3 완성)

### 11-3. 발표 산출물

- GTC 2027 정규세션 proposal abstract
- GTC talk 슬라이드 (45분 분량)
- 라이브 데모 영상 + 스크립트
- **arXiv 기술 백서 (필수 deliverable)** — Month 3 Week 4 제출 목표
- 스폰서 데모 영상 (Month 2 Week 2)

### 11-4. 오픈소스 산출물

- GitHub 공개 리포지토리 (Apache 2.0)
- Docker 이미지 (Docker Hub)
- PyPI 패키지 (`pip install paged-automl`)

---

## 12. Team & Responsibilities

**저자 구조**: 본 프로젝트는 **공동 1저자(co-first authors) 3인 체제**로 진행한다. arXiv 백서와 GTC submission 모두 3인을 equal contribution으로 기재.

| 역할 | 담당 | 책임 범위 |
|:-----|:-----|:----------|
| **공동 1저자 / 프로젝트 리드** | 본인 | 전체 스코프 관리, 아키텍처 스캐폴드, PRD, GTC submission 실무, 발표 |
| **공동 1저자** | 박사 1 (TBD) | 세부 역할 미확정 — 스캐폴드 공유 후 협의로 배정 |
| **공동 1저자** | 박사 2 (TBD) | 세부 역할 미확정 — 스캐폴드 공유 후 협의로 배정 |

**역할 배정 방식**
- 프로젝트 리드가 Month 1 Week 1에 코드 스캐폴드 + 작업 패키지 분해안을 공유
- 공동 1저자 2인의 선호·전문성을 반영해 Week 1 종료 시점까지 역할 확정
- 확정 후 본 테이블을 업데이트하며, GitHub Issue로 작업 단위 할당
- 기여도 equal을 보장하기 위해 주요 설계 결정은 3인 합의, 구현은 작업 패키지 단위로 오너십 분리

**협업 규칙**
- Weekly sync 1회 (금요일 1시간)
- 모든 결정은 GitHub Issue에 기록
- 코드 리뷰는 PR 기반, 최소 1인 리뷰어
- Merge 전 CI 통과 필수

---

## 13. Hardware & Infrastructure

### 13-1. 하드웨어 확보 경로 — 스폰서십 기반 렌탈

**자금 출처**: 외부 스폰서(클라우드 공급자 또는 기관)로부터 H100 크레딧 지원을 받을 예정.

**선결 조건**: 스폰서는 **완성된 계획 + 동작하는 데모**를 확인해야 크레딧을 승인한다. 따라서 프로젝트 구조상 단순 H100 확보 일정이 아닌 **스폰서 데모 Gate**가 선행된다.

**단계별 환경 운용**

| 기간 | 환경 | 비용 부담 | 용도 |
|:-----|:-----|:----------|:-----|
| Month 1 | 로컬 장비 또는 저가 렌탈 (A10/L4 단일) | 자체 부담 (최소화) | 설계, 기본 코드, baseline 해부 |
| Month 2 Week 1~2 | 동일 | 자체 부담 | P0~P1 operator 구현 + 스폰서 데모 준비 |
| **Month 2 Week 2 종료** | **🚩 스폰서 데모 Gate** | — | H100 크레딧 승인 획득 |
| Month 2 Week 3 ~ Month 3 | **H100 × 최대 5장 (스폰서 크레딧)** | 스폰서 부담 | 확장 구현 + 벤치마크 + scaling study |

**H100 환경 최소 요구치** (스폰서와 협의 시 기준)
- H100 80GB SXM5(NVLink) 선호, PCIe도 허용
- CPU: 노드당 vCPU 32 이상
- 시스템 RAM: ≥ 256GB (Object Store spilling 여유)
- NVMe SSD: ≥ 1TB (빠른 spilling 디스크)

**스폰서 후보 (우선순위)**
1. NVIDIA Inception Program (RAPIDS 기여 사례로 설득력 높음)
2. Lambda Labs Research Grant
3. RunPod / Vast.ai Academic 크레딧
4. 국내 클라우드 공급자 (네이버·카카오·NHN) 연구 지원 프로그램

### 13-2. 소프트웨어 스택

- OS: Ubuntu 22.04 LTS
- CUDA: 12.2+
- Driver: 535+
- Docker: 24.x + NVIDIA Container Toolkit
- Python: 3.11
- RAPIDS: 24.06 이상
- Ray: 2.30 이상
- XGBoost: 2.x GPU
- ONNX Runtime GPU: 1.17+ 또는 Triton Inference Server: 24.x (추론 경로 결정 후 하나 선택)

### 13-3. 모니터링

- NVIDIA DCGM (GPU 메트릭)
- Prometheus + Grafana
- Ray Dashboard

---

## 14. Risks & Mitigations

| 리스크 | 가능성 | 영향 | 완화 전략 |
|:-------|:------:|:----:|:----------|
| **스폰서 데모 Gate 실패 (H100 크레딧 미확보)** | 중 | 매우 높음 | 복수 스폰서 병행 접촉. A100 80GB 또는 L40S 기반 축소 벤치마크로 전환. scaling study 규모 축소하되 핵심 contribution(preprocessing operators + E2E throughput)은 유지. |
| 데모 품질 미달 | 저~중 | 높음 | Month 2 Week 1 말 데모 Dry-run 1회 필수. 최소 1개 데이터셋에서 학습→추론 완주 확인. |
| 추론 경로 결정 지연 (ONNX vs Triton) | 중 | 중 | Month 2 Week 1 워크샵 deadline 엄수. 두 경로 비교표(§8-6-2) 기반 객관적 판단. 결정 못 하면 default Triton. |
| ONNX opset 제약으로 핵심 operator 미지원 | 중 | 중 | 이 경우 Triton-only로 전환. 문서에 제약을 투명하게 명시 (future work). |
| 렌탈 비용 초과 (스폰서 크레딧 소진) | 저 | 중 | Month 3 Week 1~2에 집중 사용. Week 3~4는 결과 분석·집필 위주로 GPU 사용 최소화. |
| Target Encoder 등 핵심 operator 구현 난이도 | 중 | 높음 | Week 1 spike 후 re-estimate. 실패 시 category_encoders CPU + async GPU transfer로 fallback. |
| Baseline 4종 중 1개가 예상보다 분석 난이도 높음 | 중 | 중 | 우선순위상 TPOT이 유전 알고리즘이라 시간 소요. 깊이보다 규모(coverage)를 택해 요약 수준 분석으로 한정. |
| Multi-GPU scaling efficiency가 70% 미달 | 중 | 중 | 병목 분석 후 해당 결과를 "limitation + future work"로 솔직하게 제시. GTC reviewer는 투명성을 선호. |
| GTC proposal 탈락 | 저 | 매우 높음 | 보조 타겟: NeurIPS Datasets & Benchmarks, MLSys, SoCC 등에 재활용. |
| RAPIDS API 변경(24.x → 25.x) | 저 | 중 | 24.06 버전에 핀 고정. 최신 추적은 Month 3에 한 번만 수행. |
| 팀원 이탈 | 저 | 높음 | 모든 설계·코드를 문서화하여 bus factor 감소. |
| 블랙박스 라이브러리 OOM 재발 | 중 | 중 | Ray Actor 격리 + Fractional GPU 실측 프로파일 + ASHA early stopping 삼중 방어. |

---

## 15. Open Questions & Decision Log

### 15-1. 미결정 사항

**OQ1.** 프로젝트 명칭을 `paged-automl`로 유지할 것인가, 아니면 신규 명칭으로 변경할 것인가?
- 현 명칭은 바이트 페이징 실패 실험의 잔재. 신규 프레이밍과 mismatch.
- 후보: `rapids-automl`, `gpu-automl`, `cuAutoML`, 기타.
- 결정 시점: Month 1 Week 2까지.

**OQ2.** 추론 서빙 계층을 PRD에 포함할 것인가? — **확정: 포함**
- Framework 완결성을 위해 서빙까지 포함. G6에 명시.
- 하위 결정 사항: **ONNX Runtime vs Triton Inference Server 선택**
  - 학습은 GPU-resident cuDF/cuML, 추론은 배포 환경에 따라 유연해야 함
  - 멀티모달 확장 가능성·포터빌리티·operator 제약을 고려한 Dual Runtime 설계 (§8-6)
  - Month 2 Week 1 설계 워크샵에서 default 경로 확정 → `06_serving_architecture_decision.md`

**OQ3.** LightGBM GPU를 모델 풀에 포함할 것인가? — **확정: 제외**
- 모델 풀은 cuML RF / cuML GLM / XGBoost GPU 3종으로 확정.
- 이유: Ray 통합 복잡도 대비 기여도 한계, 3개월 스코프 집중.

**OQ4.** arXiv 기술 백서를 GTC와 별도로 작성할 것인가? — **확정: 포함 (필수 deliverable)**
- Month 3 Week 3~4에 작성·제출.
- GTC talk 자료가 백서로 자연 확장되도록 구조 설계.

### 15-2. 결정 로그

| 날짜 | 결정 | 근거 |
|:-----|:-----|:-----|
| 2026-04-14 | 프로젝트 스코프를 GTC 2027 정규세션으로 격상 | 팀 합의 |
| 2026-04-14 | Baseline 4종을 H2O/AutoGluon/TPOT/LightAutoML로 확정 | GTC reviewer 설득력 + 조사 범위 현실성 |
| 2026-04-14 | Multi-GPU 포지션 채택 (consumer GPU 프레이밍 폐기) | H100 5장 확보 추진 |
| 2026-04-14 | 기존 `paged_manager.py`를 "실패 실험 증거"로 보존 | Ray 도입 motivation 강화 |
| 2026-04-14 | **Inference serving을 scope에 포함 (OQ2)** | Framework 완결성 필수, 연구 프로젝트가 아닌 프레임워크로 포지셔닝 |
| 2026-04-14 | **LightGBM GPU 제외 (OQ3)** | Ray 통합 복잡도 대비 기여도 한계, 스코프 집중 |
| 2026-04-14 | **arXiv 백서를 필수 deliverable로 승격 (OQ4)** | 학계 인용성 확보 + 후속 연구 확장성 |
| 2026-04-14 | 하드웨어는 스폰서십 기반 렌탈로 확보, Month 2 Week 2에 데모 Gate 설치 | 자체 비용 부담 최소화 |
| 2026-04-14 | Dual Runtime 아키텍처 채택 (학습·추론 런타임 분리) | 학습 GPU-resident vs 추론 포터빌리티 요구사항 충돌 해소 |

---

## 16. References

### 16-1. AutoML Frameworks (Baseline 대상)

1. H2O AutoML — https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
2. AutoGluon (Erickson et al., 2020) — *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data*, arXiv:2003.06505
3. TPOT (Olson et al., 2016) — *Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science*, GECCO 2016
4. LightAutoML (Vakhrushev et al., 2021) — *LightAutoML: AutoML Solution for a Large Financial Services Ecosystem*, arXiv:2109.01528

### 16-2. GPU Data Science Stack

5. RAPIDS cuDF Documentation — https://docs.rapids.ai/api/cudf/stable/
6. RAPIDS cuML Documentation — https://docs.rapids.ai/api/cuml/stable/
7. XGBoost GPU Support — https://xgboost.readthedocs.io/en/stable/gpu/
8. ONNX Runtime — https://onnxruntime.ai/docs/
9. NVIDIA Triton Inference Server — https://github.com/triton-inference-server/server

### 16-3. 분산 실행 / 메모리 관리

10. Ray Core Documentation — https://docs.ray.io/en/latest/ray-core/walkthrough.html
11. Ray Tune (Liaw et al., 2018) — *Tune: A Research Platform for Distributed Model Selection and Training*, arXiv:1807.05118
12. PagedAttention (Kwon et al., 2023) — *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023
13. xgboost_ray — https://github.com/ray-project/xgboost_ray

### 16-4. HPO Theory

14. Hyperband (Li et al., 2018) — JMLR 18(185)
15. ASHA (Li et al., 2020) — *A System for Massively Parallel Hyperparameter Tuning*, MLSys 2020

### 16-5. 프로젝트 내부 문서

16. [00_strategic_direction.md](00_strategic_direction.md) — 전략 방향 전환
17. [../01-research/](../01-research/) — H2O AutoML 연구 자료
18. [../03-results/limitations.md](../03-results/limitations.md) — 기존 PagedAutoML 한계 분석
19. [../04-future/](../04-future/) — Ray + ASHA 인프라 문서 (appendix로 편입)

---

## 부록 A. Glossary

| 용어 | 정의 |
|:-----|:-----|
| AutoML | Automated Machine Learning. 모델 선택, 하이퍼파라미터 튜닝, 전처리를 자동화. |
| GPU-Native | 연산 전 과정이 GPU 메모리에 상주하여 CPU 경유 없이 수행되는 상태. |
| E2E | End-to-End. 데이터 로드부터 최종 모델 선택까지 전체 파이프라인. |
| Operator | 본 프로젝트에서 단일 전처리 단위를 지칭 (fit/transform 인터페이스). |
| Pipeline Composition | 데이터 특성에 따라 operator 순서를 자동으로 결정하는 과정. |
| Blackbox Library | 내부 메모리 할당을 외부에서 제어할 수 없는 라이브러리 (cuML, XGBoost 등). |
| Fractional GPU | Ray의 기능. 하나의 물리 GPU를 논리적으로 분할하여 여러 Actor에 할당. |
| ASHA | Asynchronous Successive Halving Algorithm. 비동기 early stopping HPO. |
| Rung | ASHA/Hyperband에서 동일 리소스 레벨의 trial 그룹. |
| Stacking | 여러 base 모델의 예측을 meta-learner가 결합하는 앙상블 기법. |
| Dual Runtime | 학습과 추론의 runtime 제약이 다르므로 두 경로를 분리하여 설계하는 아키텍처. 학습은 cuDF/cuML GPU-resident, 추론은 ONNX 또는 Triton backend. |
| Exporter | 학습된 GPU 파이프라인을 추론용 포터블 아티팩트로 직렬화하는 인터페이스. |

---

## 부록 B. 변경 이력

| 버전 | 날짜 | 변경 내용 | 작성자 |
|:-----|:-----|:----------|:-------|
| v1.0 | 2026-04-14 | 초안 작성 | 프로젝트 리드 |
| v1.1 | 2026-04-14 | OQ2(서빙 포함)/OQ3(LGBM 제외)/OQ4(arXiv 필수) 확정 반영. §8-6 Inference Serving Architecture 신설. §10 마일스톤에 스폰서 데모 Gate + 서빙 구현 + arXiv 집필 통합. §13 스폰서십 기반 렌탈 운용 모델로 재작성. §14 리스크 재정렬. | 프로젝트 리드 |
