<p align="center">
  <img src="assets/paged-automl_image.png" alt="H2O AutoML" width="400"/>
</p>

<h1 align="center">PagedAutoML</h1>

<p align="center">
  <strong>H2O AutoML의 전략을 GPU에서 재구현하고, 메모리 관리 방향을 탐색하는 연구 프로젝트</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAPIDS-26.02-7400B8?logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-3.2_GPU-76B900"/>
  <img src="https://img.shields.io/badge/uv-Package_Manager-DE5FE9?logo=uv&logoColor=white"/>
</p>

---

## What

H2O AutoML의 검증된 전략(Stacking, HPO, 훈련 순서)을 분석하고, RAPIDS GPU에서 재구현한 뒤, GPU 메모리 관리 문제를 탐색하는 연구 프로젝트.

**달성한 것**: H2O Stacking을 GPU에서 재현 (129만 행 검증), 3-Phase 훈련 전략 유효성 확인, VRAM 프로파일링 <br/>
**한계**: Memory-Aware는 skip만 가능, vLLM 수준 paging 미달, CPU 비교 미실시 <br/>
**상세**: [docs/](docs/) 참고

---

## Results

### Credit Card Fraud (1.29M rows) — AUC 0.998, 181초

<p align="center">
  <img src="assets/results/credit_card/model_performance.png" width="800"/>
</p>

### Higgs Boson (5M rows) — AUC 0.813, 153초

<p align="center">
  <img src="assets/results/higgs_5m/model_performance.png" width="800"/>
</p>

Higgs에서 rmm pool ON 시 모든 모델 skip 발생 -> [상세 해석](docs/03-results/benchmark_interpretation.md) / [한계](docs/03-results/limitations.md) / [개선 방향](docs/03-results/future_directions.md)

---

## Quick Start

```bash
git clone https://github.com/ModulabsRAPIDSLAB/H2O_AutoML.git
cd H2O_AutoML

uv venv --python 3.12
uv sync   # 타임아웃 시: UV_HTTP_TIMEOUT=300 uv sync
```

```python
from paged_automl import GPUAutoML

automl = GPUAutoML(max_runtime_secs=300, max_models=20, memory_aware=True)
automl.fit(X_train, y_train)
print(automl.leaderboard())
```

```bash
# E2E 테스트
python -m tests.test_e2e_gpu --rows 10000 --features 20 --models 5
```

---

## Structure

```
H2O_AutoML/
+-- docs/                            # 문서 전체 (클릭해서 목차 확인)
|   +-- 01-research/                 #   H2O AutoML 분석 (Stacking, HPO, GPU 매핑)
|   +-- 02-design/                   #   GPU 재설계 (PRD, 아키텍처, vLLM 비교)
|   +-- 03-results/                  #   벤치마크 해석, 한계, 개선 방향
|
+-- src/
|   +-- paged_automl/                # GPU AutoML 프레임워크
|   |   +-- automl.py                #   GPUAutoML (진입점)
|   |   +-- orchestrator.py          #   H2O 훈련 전략
|   |   +-- scheduler.py             #   Memory-Aware Scheduler
|   |   +-- data/                    #   cuDF 로더, CV, 전처리
|   |   +-- models/                  #   XGBoost GPU, cuML RF, cuML GLM
|   |   +-- ensemble/                #   Stacked Ensemble + Meta Learner
|   |   +-- memory/                  #   Profiler, Estimator, PagedMemoryManager
|   |   +-- hpo/                     #   Random Search + presets
|   |   +-- reporting/               #   Leaderboard + memory report
|   +-- notebooks/                   # Jupyter 노트북
|   |   +-- 01_h2o_automl_demo       #   H2O CPU baseline
|   |   +-- 04_paged_automl_benchmark#   GPU 벤치마크 (Credit Card)
|   +-- scripts/                     # 차트 생성
|   +-- tests/                       # E2E GPU 테스트
|
+-- assets/results/                  # 벤치마크 차트
    +-- credit_card/                 #   Credit Card Fraud 결과
    +-- higgs_5m/                    #   Higgs Boson 결과
```

---

## References

- LeDell & Poirier (2020). *H2O AutoML: Scalable Automatic Machine Learning.* ICML Workshop.
- Bergstra & Bengio (2012). *Random Search for Hyper-Parameter Optimization.* JMLR.
- Kwon et al. (2023). *Efficient Memory Management for LLM Serving with PagedAttention.* SOSP.
- [RAPIDS docs](https://docs.rapids.ai/) / [H2O docs](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) / [Troubleshooting](docs/troubleshooting.md)
