# 🚀 H2O AutoML Benchmarking Project

**OpenML**의 다양한 데이터셋을 활용하여 **H2O AutoML**의 시간 대비 성능 효율성을 측정하는 벤치마크 실험.

## 1. 실험 목적
- **시간 예산(Time Budget)** 변화에 따른 AutoML의 모델 성능 향상 폭 측정.
- 특정 데이터셋(AUC vs Logloss 계열)에 따른 H2O AutoML의 수렴 속도 비교 분석.

## 2. 실험 환경
- **OS:** Windows Subsystem for Linux (WSL2) - Ubuntu
- **Python:** 3.10.14 (managed by `pyenv`)
- **Package Manager:** `uv` (for fast & reproducible dependency management)
- **Main Tool:** H2O AutoML (Java-based server-client architecture)

## 3. 실험 절차 (Pipeline)
1. **Data Load:** `openml-py` 라이브러리를 통해 표준 데이터셋 로드.
2. **Preprocessing:** H2O 전용 데이터 프레임 변환 및 타겟 변수 범주화(Factorization).
3. **Stepped Training:** 1분, 3분, 5분(Full은 5/15/30분) 단위로 점진적 학습 수행.
4. **Evaluation:** 테스트 세트 기반의 `roc-auc` 및 `logloss` 지표 산출.
5. **Report:** CSV 저장 및 시각화를 통한 추이 분석.

## 4. 벤치마크 결과 

### 📊 AutoML Benchmark Performance Table (Mini-Test Results)

실험의 무결성을 검증하기 위해 3개의 대표 데이터셋을 대상으로 수행한 사전 결과입니다.

| dataset    | metric   |    1min |    3min |    5min |
|------------|----------|---------|---------|---------|
| australian | roc-auc  |  0.951  |  0.9491 |  0.9397 |
| credit-g   | roc-auc  |  0.7563 |  0.7467 |  0.7368 |
| car        | -logloss | -0.0011 | -0.0003 | -0.0001 |


## 5. 실행 방법
```bash
# 1. 단일 데이터셋 테스트
uv run benchmarks/scripts/benchmark_single.py

# 2. 미니 벤치마크 실행 (3개 데이터셋)
uv run benchmarks/scripts/benchmark_mini.py

# 3. 시각화 수행
uv run benchmarks/scripts/visualize_results.py