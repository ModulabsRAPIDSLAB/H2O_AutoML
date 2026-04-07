<p align="center">
  <img src="assets/h2o-automl-logo.png" alt="H2O AutoML" width="400"/>
</p>

<h1 align="center">PagedAutoML</h1>

<p align="center">
  <strong>vLLM-inspired Paged Memory Management for GPU AutoML</strong><br/>
  H2O의 검증된 전략을 RAPIDS GPU에서 재조립하고, Page 기반 메모리 관리를 내재화한 프레임워크
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/RAPIDS-26.02-7400B8?logo=nvidia&logoColor=white" alt="RAPIDS"/>
  <img src="https://img.shields.io/badge/XGBoost-3.2_GPU-76B900" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/sklearn-Compatible-F7931E?logo=scikitlearn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/uv-Package_Manager-DE5FE9?logo=uv&logoColor=white" alt="uv"/>
</p>

---

## The Problem

GPU AutoML은 CPU 대비 10 ~ 40배 빠른 모델 탐색이 가능하지만, **GPU VRAM(8 ~ 24GB)이라는 메모리 제약**이 실질적 병목이다.

AutoML은 일반 ML과 다르다 — 모델 1개가 아니라 **수십 개 모델 x 5-fold CV = 수백 번의 훈련**이 동일 GPU에서 경쟁한다. 기존 GPU AutoML 프레임워크는 이 문제를 무시한다.

| | H2O AutoML | AutoGluon+RAPIDS | **PagedAutoML** |
|:--|:----------:|:----------------:|:--------------:|
| 실행 환경 | CPU (JVM) | GPU | GPU |
| 메모리 관리 | JVM GC | 없음 | **Paged Memory** |
| Stacked Ensemble | O | O | O (H2O 전략) |
| VRAM 프로파일링 | X | X | **O** |

---

## The Journey

이 프로젝트는 3단계 연구 여정을 따릅니다:

### Phase 1: Research — H2O AutoML 분석 &nbsp; [> docs/01-research/](docs/01-research/)

H2O AutoML의 Stacking, HPO, 훈련 전략을 심층 분석.
**Key Insight**: H2O의 가치는 코드가 아니라 10년간 검증된 **전략**에 있다.

### Phase 2: Design — GPU 재설계 &nbsp; [> docs/02-design/](docs/02-design/)

RAPIDS 위에서 H2O 전략을 재조립하고, Memory-Aware 아키텍처를 설계.
**Key Insight**: RAPIDS가 부품을 제공하지만, **AutoML 조립 + 메모리 관리**는 아무도 안 했다.

### Phase 3: Build — 구현 & 검증 &nbsp; [> docs/03-results/](docs/03-results/)

`paged_automl/` 프레임워크를 구현하고, RTX 4060 (8GB)에서 실제 데이터로 검증.
**Key Insight**: 8GB GPU에서 OOM 0건, 앙상블이 개별 모델보다 안정적으로 상위권.

---

## Results

> Kaggle Credit Card Fraud Detection (1,296,675 rows x 11 features), RTX 4060 (8GB VRAM), **181초에 12개 모델 완료**

### Model Performance (AUC)

<p align="center">
  <img src="assets/results/model_performance.png" alt="Model Performance (AUC)" width="800"/>
</p>

- **XGBoost Diversity (0.9980)**: 깊은 트리(depth 10)가 사기의 복잡한 패턴을 포착 → Baseline 대비 +0.27%
- **Stacked Ensemble (0.9973)**: XGBoost 72% + RF 28%로 조합. 실패한 GLM은 자동으로 가중치 0 처리
- **GLM 실패 (0.52)**: 극도의 클래스 불균형(사기 0.58%)에서 수렴 실패 → 앙상블이 자동 제외

### Training Time & Memory

<p align="center">
  <img src="assets/results/training_time.png" alt="Training Time" width="800"/>
</p>

<p align="center">
  <img src="assets/results/memory_per_model.png" alt="Memory per Model" width="800"/>
</p>

- 전체 시간의 83%가 Diversity Phase (GPU 병렬화의 최대 효과 구간)
- 모델당 VRAM: 0.006 ~ 0.038 GB — 8GB GPU에서 OOM 0건

### Pipeline Memory Profile

<p align="center">
  <img src="assets/results/memory_per_stage.png" alt="Memory per Stage" width="800"/>
</p>

상세 해석: [docs/03-results/benchmark_interpretation.md](docs/03-results/benchmark_interpretation.md)

---

## Quick Start

### Requirements

| 항목 | 요구사항 |
|:----:|:--------|
| GPU | NVIDIA, Compute Capability 7.0+ (Volta 이상), VRAM 8GB+ |
| Driver | 580.65+ (CUDA 13) |
| Python | 3.12+ |
| OS | Linux (Ubuntu 22.04+) 또는 WSL2 |

### Installation

```bash
git clone https://github.com/ModulabsRAPIDSLAB/H2O_AutoML.git
cd H2O_AutoML

# Python 3.12 venv + RAPIDS/XGBoost GPU 전체 설치
uv venv --python 3.12
uv sync   # 대용량 패키지 타임아웃 시: UV_HTTP_TIMEOUT=300 uv sync
```

### Usage

```python
from paged_automl import GPUAutoML

automl = GPUAutoML(
    max_runtime_secs=300,
    max_models=20,
    memory_aware=True,
    memory_profile=True,
)

automl.fit(X_train, y_train)           # cuDF DataFrame 또는 파일 경로

print(automl.leaderboard())            # 성능 + peak_vram 포함
print(automl.get_memory_report())      # 단계별 VRAM 리포트
preds = automl.predict(X_test)         # 앙상블 예측
```

### Run E2E Test

```bash
source .venv/bin/activate

# 통합 테스트
python -m tests.test_e2e_gpu --rows 10000 --features 20 --models 5

# 결과 차트 생성
python scripts/generate_charts.py --run --compare
```

---

## Architecture

```
GPUAutoML.fit()
  │
  ├─ Data: cuDF load → Preprocessor (GPU-native)
  │
  ├─ Orchestrator (H2O Strategy)
  │    ├─ Phase A: Baseline (default HP, 1 per algorithm)
  │    ├─ Phase B: Diversity (pre-specified HP grids)
  │    └─ Phase C: Random Search (until time budget)
  │
  ├─ Memory-Aware Scheduler          ← core contribution
  │    ├─ VRAMEstimator.estimate()   → 모델별 VRAM 예측
  │    └─ profiler.get_free_vram()   → 가용 VRAM 체크 → skip or proceed
  │
  ├─ 5-fold CV + OOF Collection
  │    ├─ cuML RF / cuML GLM / XGBoost GPU
  │    └─ Streaming OOF (VRAM 절약)
  │
  └─ Stacked Ensemble
       ├─ All Models Ensemble
       ├─ Best of Family Ensemble
       └─ Non-negative GLM Meta Learner
```

상세 설계: [docs/02-design/architecture.md](docs/02-design/architecture.md)

---

## Project Structure

```
H2O_AutoML/
├── paged_automl/              # Memory-Aware GPU AutoML 프레임워크
│   ├── automl.py            #   GPUAutoML (sklearn-compatible API)
│   ├── orchestrator.py      #   H2O training strategy + time control
│   ├── scheduler.py         #   Memory-Aware Scheduler
│   ├── data/                #   cuDF loader, K-fold CV, preprocessor
│   ├── models/              #   XGBoost GPU, cuML RF, cuML GLM
│   ├── ensemble/            #   Stacked Ensemble + Meta Learner
│   ├── memory/              #   VRAM profiler, estimator, rmm pool
│   ├── hpo/                 #   Random Search + presets
│   └── reporting/           #   Leaderboard + memory report
├── docs/
│   ├── 01-research/         #   Phase 1: H2O AutoML 분석
│   ├── 02-design/           #   Phase 2: GPU 재설계 전략 + PRD
│   └── 03-results/          #   Phase 3: 구현 계획 + 실행 로그
├── notebooks/               #   H2O baseline 데모 (01, 02, 03)
├── tests/                   #   E2E GPU 통합 테스트
├── scripts/                 #   차트 생성 스크립트
├── assets/results/          #   벤치마크 차트 (PNG + JSON)
└── pyproject.toml           #   Python 3.12 + RAPIDS CUDA 13
```

---

## Notebooks

| # | Notebook | 내용 |
|:-:|----------|------|
| 1 | [01_h2o_automl_demo](notebooks/01_h2o_automl_demo.ipynb) | H2O AutoML CPU baseline 데모 |
| 2 | [02_explainability](notebooks/02_explainability.ipynb) | h2o.explain() 시각화 |
| 3 | [03_gpu_native_redesign](notebooks/03_gpu_native_redesign.ipynb) | GPU-native 매핑 분석 (개념 설계, 읽기만) |
| 4 | [**04_paged_automl_benchmark**](notebooks/04_paged_automl_benchmark.ipynb) | **Credit Card Fraud 1.29M rows 벤치마크 (GPU)** |

---

## References

- LeDell, E., & Poirier, S. (2020). *H2O AutoML: Scalable Automatic Machine Learning.* 7th ICML Workshop on AutoML.
- Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization.* JMLR.
- [RAPIDS Documentation](https://docs.rapids.ai/)
- [H2O AutoML Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

## Troubleshooting

[docs/troubleshooting.md](docs/troubleshooting.md) 참고
