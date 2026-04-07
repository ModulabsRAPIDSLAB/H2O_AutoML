# 발표 논리 구조 및 근거 정리

> 수요일 발표의 핵심 스토리라인: 문제 정의 → 기존 시스템 분석 → 설계 제안 → 근거 → 검증

---

## 1. 문제 정의: GPU AutoML은 왜 아직 해결되지 않았는가

### 1-1. AutoML의 본질적 특성

AutoML은 다른 ML 워크로드와 근본적으로 다르다.

일반 ML: 모델 1개를 잘 훈련하는 것이 목표
AutoML: **모델 수십~수백 개를 훈련하고, 그 결과를 조합**하는 것이 목표

```
일반 ML 워크로드:              AutoML 워크로드:
  데이터 → 모델 1개 훈련         데이터 → 모델 20개 × 5-fold = 100번 훈련
                                       → OOF 수집 → Level-One Data
                                       → Meta Learner 훈련
                                       → 앙상블 구성
```

이 차이가 GPU 환경에서 결정적인 문제를 만든다.

### 1-2. GPU 메모리 제약: AutoML에서 특히 치명적인 이유

```
GPU VRAM 사용 = 원본 데이터 + 활성 모델 + 임시 버퍼 + Level-One Data

일반 ML:  데이터(2GB) + 모델 1개(4GB) = 6GB → 16GB GPU에서 문제 없음
AutoML:   데이터(2GB) + 모델 3개 동시(12GB) + OOF(0.1GB) = 14.1GB → 16GB GPU에서 위험
```

**병렬화를 많이 할수록 빨라지지만, 메모리 제약에 부딪힌다.**
이것이 GPU AutoML의 핵심 딜레마이다.

### 1-3. 기존 접근의 한계

| 시스템 | GPU 활용 | 메모리 관리 | 한계 |
|--------|----------|-------------|------|
| H2O AutoML | XGBoost만 GPU (Java 브릿지) | CPU RAM 기반 (제약 없음) | Java/CPU 종속, GPU 가속 제한적 |
| AutoGluon + RAPIDS | 일부 모델 cuML 대체 | 프레임워크 기본값 | 메모리 관리 부재 |
| TPOT + RAPIDS | 전처리 + cuML | 없음 (싱글 GPU) | 병렬화 없음, 메모리 관리 없음 |

**공통 문제: "속도"에만 집중하고 "메모리"는 무시한다.**

---

## 2. H2O AutoML 분석: 가져갈 것과 버릴 것

### 2-1. H2O에서 가져가는 것 — 10년간 검증된 설계 전략

| 기법 | 왜 가져가는가 |
|------|---------------|
| Stacked Ensemble (All Models + Best of Family) | 모델 다양성을 극대화하면서 실전 배포까지 고려한 구조 |
| OOF 기반 Level-One Data | 데이터 누수 방지의 표준 방법 |
| Non-negative GLM Meta Learner + L1 정규화 | 앙상블 안정성 + sparse ensemble 유도 |
| 훈련 순서 전략 (Baseline → Diversity → Random Search) | 시간 예산 내 최적 탐색을 위한 의도적 설계 |
| 시간 기반 제어 (max_runtime_secs) | 실용적 AutoML UX의 핵심 |

→ 이것들은 **코드가 아니라 전략**이므로 GPU 환경에서도 그대로 유효하다.

### 2-2. H2O에서 버리는 것 — Java/CPU 종속 구현

| 구현 | 왜 버리는가 |
|------|-------------|
| H2O Frame (Java 인메모리 프레임) | JVM 힙 종속, GPU 접근 시 Java-C++ 브릿지 오버헤드 |
| Java 기반 알고리즘 구현 | GPU 가속 불가 |
| 순차 실행 아키텍처 | 모델 하나씩 순차 훈련, 병렬화 불가 |

---

## 3. 우리의 설계 제안: Memory-Aware GPU AutoML

### 3-1. 핵심 아이디어

```
H2O의 검증된 전략  +  RAPIDS의 GPU 인프라  +  메모리 인지 스케줄링
     (방법론)              (구현 도구)              (우리의 기여)
```

### 3-2. 왜 이것이 가능한가 — RAPIDS 생태계가 이미 준비해놨다

**근거 1: cuML은 sklearn 호환 API를 제공한다**

H2O의 알고리즘 풀을 cuML로 대체할 수 있다.
import만 바꾸면 GPU 가속이 적용된다.

> 출처: [RAPIDS cuML 공식 문서 — sklearn 호환 API](https://docs.rapids.ai/api/cuml/stable/cuml_intro/)
> 추가: cuml.accel을 사용하면 코드 변경 없이도 sklearn 코드가 GPU에서 실행됨
> 출처: [cuML Zero Code Change Acceleration](https://docs.rapids.ai/api/cuml/stable/zero-code-change/)

| H2O 알고리즘 | cuML/GPU 대체 | API 호환성 |
|---|---|---|
| Random Forest (DRF) | `cuml.ensemble.RandomForestClassifier` | sklearn 호환 |
| GLM | `cuml.linear_model.LogisticRegression` | sklearn 호환 |
| XGBoost (Java wrapper) | XGBoost native (`tree_method='hist', device='cuda'`) | 파라미터 변경만 |
| DNN | PyTorch MLP | 별도 구현 필요 |

**근거 2: Dask-CUDA는 GPU 메모리 인지 스케줄링을 제공한다**

> 출처: [Dask-CUDA API 문서](https://docs.rapids.ai/api/dask-cuda/nightly/api/)

```python
LocalCUDACluster(
    device_memory_limit="12GB",  # worker당 VRAM 상한 → 초과 시 spill-to-host
    rmm_pool_size="10GB",        # rmm 메모리 풀 사전 할당 → 할당 오버헤드 제거
)
```

이 두 파라미터가 Memory-Aware Scheduling의 기반이다.

**근거 3: rmm은 GPU 메모리 할당을 로깅할 수 있다**

> 출처: [RMM User Guide](https://docs.rapids.ai/api/rmm/stable/user_guide/guide/)

```python
rmm.reinitialize(logging=True, log_file="memory_trace.csv")
# 모든 GPU 메모리 할당/해제가 CSV로 기록됨
```

이것으로 AutoML 파이프라인의 메모리 사용 패턴을 정밀 측정할 수 있다.

**근거 4: XGBoost는 Dask 멀티 GPU 훈련을 공식 지원한다**

> 출처: [XGBoost Dask GPU Training](https://xgboost.readthedocs.io/en/latest/python/dask-examples/gpu_training.html)

```python
# 주의: gpu_hist는 deprecated. 현재 올바른 API:
params = {'tree_method': 'hist', 'device': 'cuda'}
output = dxgb.train(client, params, dtrain)
```

### 3-3. 벤치마크 근거: GPU AutoML은 실제로 빠르다

| 사례 | 결과 | 출처 |
|------|------|------|
| TPOT + RAPIDS | GPU 1시간이 CPU 8시간보다 높은 정확도 (+1.3~2%) | [RAPIDS AI Blog: Faster AutoML with TPOT and RAPIDS](https://medium.com/rapids-ai/faster-automl-with-tpot-and-rapids-758455cd89e5) |
| AutoGluon + RAPIDS | 훈련 최대 40배, 추론 최대 10배 가속 | [NVIDIA Blog: Advancing AutoML 10x Faster](https://developer.nvidia.com/blog/advancing-the-state-of-the-art-in-automl-now-10x-faster-with-nvidia-gpus-and-rapids/) |
| cuML StackingClassifier | 훈련 35배, 추론 350배+ 가속 (10만 행 기준) | [RAPIDS AI Blog: 100x Faster ML Ensembling](https://medium.com/rapids-ai/100x-faster-machine-learning-model-ensembling-with-rapids-cuml-and-scikit-learn-meta-estimators-d869788ee6b1) |

---

## 4. 해결해야 할 기술적 과제 (정직하게)

### 4-1. cuML GLM Non-negative 제약 미지원 (확인됨)

> 출처: [cuML LogisticRegression API](https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.linear_model.logisticregression/)

cuML LogisticRegression은 `non_negative` 옵션을 **지원하지 않는다.**
H2O Stacking의 Meta Learner 핵심 기능이므로 대안이 필요하다.

**대안 전략:**

| 순위 | 방안 | 장단점 |
|------|------|--------|
| 1 | cuML GLM 학습 후 음수 계수 0 클리핑 + 재정규화 | GPU 완결, 간단. 수학적으로 제약 최적화와 다르나 실전에서 유사 결과 가능 |
| 2 | scipy.optimize.nnls CPU fallback | 정확한 NNLS. Level-One Data는 크기가 작아 CPU 전송 비용 미미 |
| 3 | CuPy 기반 NNLS 직접 구현 | GPU 네이티브 + 정확. 구현 복잡도 높음 |

→ Phase 0 기술 스파이크에서 방안 1의 정확도 영향을 실험하고, 유의미한 차이가 있으면 방안 2로 전환.

### 4-2. XGBoost API 변경

`tree_method='gpu_hist'`는 deprecated.
현재 올바른 설정: `tree_method='hist', device='cuda'`

> 출처: [XGBoost GPU Training Docs](https://xgboost.readthedocs.io/en/latest/python/dask-examples/gpu_training.html)

### 4-3. rmm logging 성능 오버헤드

rmm logging은 모든 할당/해제를 기록하므로 성능에 영향을 줄 수 있다.
→ 프로파일링 모드와 프로덕션 모드를 분리해야 한다.

### 4-4. GPU 환경 세팅 복잡성

RAPIDS는 CUDA 버전, 드라이버 버전, Python 버전의 호환성 매트릭스가 까다롭다.
→ Docker 이미지(rapidsai/base) 또는 conda 환경으로 해결.

---

## 5. 발표 스토리라인

```
[도입] 3분
  "AutoML은 모델을 자동으로 찾아주지만, GPU에서 돌리면 메모리가 문제"
  → GPU AutoML의 딜레마: 병렬화 ↑ = 속도 ↑ but 메모리 ↑↑

[기반 이론] 7분 (팀원 발표)
  1. Stacked Ensemble 원리 (권석민)
     - Base Models → OOF → Level-One Data → Meta Learner
     - All Models vs Best of Family
  2. Hyperparameter Tuning (김선아)
     - Random Search가 Grid Search보다 효율적인 이유
     - H2O의 훈련 순서 전략

[우리의 설계] 5분
  "H2O의 전략을 가져가되, RAPIDS로 GPU 네이티브 구현 + 메모리 최적화"
  
  아키텍처 다이어그램 (04번 문서의 기술 스택 그림)
  
  핵심 포인트:
  - cuML이 sklearn 호환이라 import만 바꾸면 GPU 가속
  - Dask-CUDA가 메모리 인지 스케줄링을 이미 제공
  - rmm으로 메모리 프로파일링 가능

[근거 제시] 3분
  벤치마크 표 (위 Section 3.3)
  + 공식 문서 URL로 신뢰성 확보

[기술적 과제] 2분
  "정직하게, 아직 해결해야 할 것들"
  - cuML GLM non-negative 미지원 → 대안 전략
  - rmm logging 오버헤드 → 모드 분리

[간단한 검증] 5분
  코드 스니펫으로:
  1. cuDF 데이터 로드 → cuML RF 훈련 → 예측 (E2E GPU 동작 확인)
  2. Dask-CUDA 클러스터 생성 → 병렬 fold 실행 (병렬화 가능성 확인)
  3. rmm logging → 메모리 사용량 측정 (프로파일링 가능성 확인)

[마무리] 2분
  로드맵 (Phase 0~4)
  "이 설계가 논문으로 이어질 수 있는 이유"
```

---

## 6. 예상 질문 & 답변

**Q: H2O를 안 쓰고 새로 짜는 이유가 뭔가요?**
A: H2O는 Java 기반이라 GPU 활용이 XGBoost 하나뿐이고, 그마저도 Java-C++ 브릿지 오버헤드로 native 대비 8~11배 느립니다 ([GBM-perf 벤치마크](https://github.com/szilard/GBM-perf)). RAPIDS는 전체 파이프라인을 GPU에서 돌릴 수 있어서 근본적으로 다릅니다.

**Q: 그냥 AutoGluon+RAPIDS 쓰면 되지 않나요?**
A: AutoGluon+RAPIDS는 일부 모델만 cuML로 대체하는 수준이고, 메모리 관리를 하지 않습니다. 우리는 Dask-CUDA + rmm으로 메모리 인지 스케줄링까지 포함한 E2E 설계를 제안합니다. 이 부분이 기존 연구에 없는 우리의 기여입니다.

**Q: Non-negative GLM이 안 되면 Stacking 품질이 떨어지지 않나요?**
A: 맞습니다. 그래서 Phase 0 기술 스파이크에서 이것부터 검증합니다. post-hoc 클리핑의 정확도 영향을 실험하고, 유의미한 차이가 있으면 scipy NNLS를 fallback으로 씁니다. Level-One Data는 크기가 작아서 CPU fallback 비용이 미미합니다.

**Q: 메모리 최적화가 실제로 성능(정확도)에 영향을 주나요?**
A: 직접적으로는 아닙니다. 하지만 간접적으로 큽니다. 같은 VRAM에서 메모리를 효율적으로 쓰면 더 많은 모델을 동시에 훈련할 수 있고, 같은 시간에 더 많은 탐색 = 더 좋은 모델을 찾을 확률이 높아집니다. H2O의 철학("빠른 인프라로 많이 돌리자")과 일맥상통합니다.
