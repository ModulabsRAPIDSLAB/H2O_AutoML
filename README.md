<p align="center">
  <img src="assets/h2o-automl-logo.png" alt="H2O AutoML" width="400"/>
</p>

<h1 align="center">PagedAutoML</h1>

<p align="center">
  <strong>H2O AutoML의 전략을 GPU에서 재구현하고, 메모리 관리 방향을 탐색하는 연구 프로젝트</strong>
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

## What This Project Is

H2O AutoML은 10년간 검증된 Stacking/HPO 전략을 가지고 있지만, Java/CPU 기반이라 GPU에서 직접 사용할 수 없다.
이 프로젝트는 **H2O의 전략을 분석하고, RAPIDS 생태계(cuDF, cuML, XGBoost GPU)로 재구현**한 뒤,
GPU 환경에서의 **메모리 관리 문제를 탐색**하는 연구 프로젝트이다.

### 실제로 달성한 것

- H2O의 Stacking(5-fold CV + OOF + Two-Type Ensemble)을 GPU에서 재현하고, 실제 데이터(129만 행)로 검증
- H2O의 3-Phase 훈련 전략(Baseline -> Diversity -> Random Search)이 GPU에서도 유효함을 확인
- VRAM 프로파일링을 Leaderboard에 포함하여 모델별 GPU 메모리 사용량을 정량화

### 아직 검증되지 않은 것

- **Memory-Aware Scheduling**: 8GB GPU + 129만 행에서는 모든 모델이 VRAM 체크를 통과하여, skip/eviction이 발생하지 않음. 더 큰 데이터(11M+ rows)에서 실제 효과 검증 필요
- **vLLM-style Paged Memory**: `PagedMemoryManager`를 구현했으나 실제 벤치마크에 미사용. 블록 할당/회수/swap의 실효성 검증 필요
- **CPU 대비 속도**: H2O CPU와의 동일 조건 비교 미실시. "10 ~ 40배 빠르다"는 RAPIDS 공식 벤치마크 수치이며, 이 프로젝트에서 직접 측정한 것이 아님
- **GLM 수렴 실패**: cuML LogisticRegression이 극도의 클래스 불균형(사기 0.58%)에서 수렴 실패 (AUC 0.52). H2O의 GLM은 이 상황을 처리하므로, 완전한 전략 재현은 아님

---

## The Journey

### Phase 1: Research — H2O AutoML 분석

H2O AutoML의 Stacking, HPO, 훈련 전략을 심층 분석했다. <br/>
**Key Insight**: H2O의 가치는 코드가 아니라 10년간 검증된 **전략**에 있다.

-> [docs/01-research/](docs/01-research/)

### Phase 2: Design — GPU 재설계

RAPIDS 위에서 H2O 전략을 재조립하고, 메모리 관리 아키텍처를 설계했다. <br/>
**Key Insight**: RAPIDS가 부품을 제공하지만, AutoML 파이프라인으로 조립하는 것은 별도의 작업이다.

-> [docs/02-design/](docs/02-design/)

### Phase 3: Build — 구현 & 검증

`paged_automl/` 프레임워크를 구현하고, RTX 4060 (8GB)에서 실제 데이터로 검증했다. <br/>
**Key Insight**: H2O 전략은 GPU에서 유효하지만, 메모리 관리의 실효성은 더 큰 규모에서 검증이 필요하다.

-> [docs/03-results/](docs/03-results/)

---

## Results: Kaggle Credit Card Fraud (1.29M rows)

> RTX 4060 (8GB VRAM), 10 base models + 2 ensembles, **181초 완료**

### Model Performance (AUC)

<p align="center">
  <img src="assets/results/model_performance.png" alt="Model Performance (AUC)" width="800"/>
</p>

| 순위 | 모델 | AUC | 의미 |
|:----:|------|:---:|------|
| 1 | xgboost_6 (Diversity) | 0.9980 | 깊은 트리가 사기 패턴 포착. Baseline 대비 +0.27% |
| 2 | SE_BestOfFamily | 0.9973 | XGBoost 72% + RF 28%. GLM은 가중치 0으로 자동 제외 |
| 11 | glm_3 | 0.5181 | 수렴 실패 (클래스 불균형). 앙상블이 자동 배제 |

**검증된 것**: H2O의 Diversity 전략이 GPU에서 성능 향상에 기여. 앙상블이 실패 모델을 자동 제외. <br/>
**한계**: GLM 수렴 실패는 cuML의 L-BFGS 한계. 클래스 가중치 적용 등 추가 대응 필요.

### Training Time & VRAM

<p align="center">
  <img src="assets/results/training_time.png" alt="Training Time" width="800"/>
</p>

<p align="center">
  <img src="assets/results/memory_per_stage.png" alt="Memory per Stage" width="800"/>
</p>

- 전체 시간의 83%가 Diversity Phase — GPU 병렬화 시 가장 큰 효과를 기대할 수 있는 구간
- 모델당 VRAM: 0.006 ~ 0.038 GB — 129만 행에서는 8GB GPU로 충분. **더 큰 데이터에서 Memory-Aware의 가치가 드러날 것으로 예상**

상세 해석: [docs/03-results/benchmark_interpretation.md](docs/03-results/benchmark_interpretation.md)

---

## Quick Start

### Requirements

| 항목 | 요구사항 |
|:----:|:--------|
| GPU | NVIDIA, Compute Capability 7.0+, VRAM 8GB+ |
| Driver | 580.65+ (CUDA 13) |
| Python | 3.12+ |
| OS | Linux (Ubuntu 22.04+) 또는 WSL2 |

### Installation

```bash
git clone https://github.com/ModulabsRAPIDSLAB/H2O_AutoML.git
cd H2O_AutoML

uv venv --python 3.12
uv sync   # 타임아웃 시: UV_HTTP_TIMEOUT=300 uv sync
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

automl.fit(X_train, y_train)
print(automl.leaderboard())            # 성능 + peak_vram 포함
print(automl.get_memory_report())      # 단계별 VRAM 리포트
preds = automl.predict(X_test)
```

### Run Tests

```bash
source .venv/bin/activate

# E2E 테스트
python -m tests.test_e2e_gpu --rows 10000 --features 20 --models 5

# 벤치마크 차트 생성
python scripts/generate_charts.py --from-json assets/results/benchmark_credit_card.json
```

---

## Architecture

```
GPUAutoML.fit()
  |
  +- Data: cuDF load -> Preprocessor (GPU-native)
  |
  +- Orchestrator (H2O Strategy)
  |    +- Phase A: Baseline (default HP per algorithm)
  |    +- Phase B: Diversity (pre-specified HP grids)
  |    +- Phase C: Random Search (until time budget)
  |
  +- Memory-Aware Scheduler
  |    +- VRAMEstimator.estimate()   -> VRAM 예측
  |    +- profiler.get_free_vram()   -> 가용 VRAM 체크
  |
  +- 5-fold CV + OOF Collection
  |    +- cuML RF / cuML GLM / XGBoost GPU
  |    +- Streaming OOF (VRAM 절약)
  |
  +- Stacked Ensemble
       +- All Models Ensemble
       +- Best of Family Ensemble
       +- Non-negative GLM Meta Learner (post-hoc clipping)
```

상세 설계 + vLLM 매핑: [docs/02-design/architecture.md](docs/02-design/architecture.md)

---

## Project Structure

```
H2O_AutoML/
+-- paged_automl/            # GPU AutoML 프레임워크
|   +-- automl.py            #   GPUAutoML (sklearn-compatible API)
|   +-- orchestrator.py      #   H2O training strategy + time control
|   +-- scheduler.py         #   Memory-Aware + Continuous Scheduler
|   +-- data/                #   cuDF loader, K-fold CV, preprocessor
|   +-- models/              #   XGBoost GPU, cuML RF, cuML GLM
|   +-- ensemble/            #   Stacked Ensemble + Meta Learner
|   +-- memory/              #   Profiler, Estimator, rmm Pool, PagedMemoryManager
|   +-- hpo/                 #   Random Search + presets
|   +-- reporting/           #   Leaderboard + memory report
+-- docs/
|   +-- 01-research/         #   Phase 1: H2O AutoML 분석
|   +-- 02-design/           #   Phase 2: GPU 재설계 + PRD + architecture
|   +-- 03-results/          #   Phase 3: 벤치마크 결과 + 해석 가이드
+-- notebooks/               #   01 H2O demo, 02 explain, 03 design, 04 benchmark
+-- tests/                   #   E2E GPU 통합 테스트
+-- scripts/                 #   차트 생성
+-- assets/results/          #   벤치마크 차트 + JSON
```

---

## Notebooks

| # | Notebook | 내용 |
|:-:|----------|------|
| 1 | [01_h2o_automl_demo](notebooks/01_h2o_automl_demo.ipynb) | H2O AutoML CPU baseline 데모 |
| 2 | [02_explainability](notebooks/02_explainability.ipynb) | h2o.explain() 시각화 |
| 3 | [03_gpu_native_redesign](notebooks/03_gpu_native_redesign.ipynb) | GPU 매핑 분석 (개념 설계, 읽기만) |
| 4 | [**04_paged_automl_benchmark**](notebooks/04_paged_automl_benchmark.ipynb) | **Credit Card Fraud 1.29M rows GPU 벤치마크** |

---

## Next Steps

이 프로젝트에서 아직 해결하지 못한 과제:

1. **대규모 데이터 벤치마크** — Higgs Boson (11M rows)에서 Memory-Aware skip/eviction 발생 확인
2. **CPU H2O 동일 조건 비교** — 같은 데이터, 같은 시간 예산으로 GPU vs CPU 속도 직접 측정
3. **PagedMemoryManager 실사용** — `paged_memory=True`로 E2E 실행 후 블록 할당/회수 효과 검증
4. **GLM 클래스 불균형 대응** — cuML GLM에 class_weight 적용 또는 SMOTE 전처리
5. **rmm 풀 전략 비교** — None / Fixed / Managed / Adaptive 4가지 전략의 실측 비교

---

## References

- LeDell, E., & Poirier, S. (2020). *H2O AutoML: Scalable Automatic Machine Learning.* 7th ICML Workshop on AutoML.
- Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization.* JMLR.
- Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP.
- [RAPIDS Documentation](https://docs.rapids.ai/)
- [H2O AutoML Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

## Troubleshooting

[docs/troubleshooting.md](docs/troubleshooting.md)
