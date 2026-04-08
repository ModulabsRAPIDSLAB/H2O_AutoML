# Memory-Aware GPU AutoML Framework 

## 1. 서론

AutoML은 기존 머신러닝과 근본적으로 다른 특성을 가지고 있습니다.  
일반적인 머신러닝은 하나의 모델을 잘 학습시키는 것이 목표이지만,  
AutoML은 수십 개에서 수백 개의 모델을 학습하고 이를 조합하여 최적의 결과를 찾는 것이 목적입니다.

이 과정에서 다음과 같은 단계가 포함됩니다:

- 여러 모델 학습
- Cross Validation 수행
- OOF (Out-of-Fold) 예측 생성
- Meta Learner 학습 및 앙상블 구성

이러한 구조는 GPU 환경에서 심각한 문제를 발생시킵니다.

GPU는 CPU에 비해 연산 성능은 뛰어나지만,  
VRAM이 제한되어 있기 때문에 (보통 16~24GB)  
AutoML과 같은 대규모 워크로드를 처리하는 데 어려움이 있습니다.

결국 핵심 문제는 다음과 같이 정리할 수 있습니다:

- 병렬화를 늘리면 속도는 빨라진다
- 하지만 메모리 사용량도 함께 증가한다
- 결국 Out-of-Memory(OOM) 문제가 발생한다

👉 따라서 GPU AutoML의 본질적인 병목은 연산이 아니라 메모리입니다.

---

## 2. 시스템 아키텍처

본 연구에서는 H2O AutoML의 설계 전략을 기반으로,  
RAPIDS 생태계를 활용하여 GPU 기반 AutoML 시스템을 설계합니다.

전체 파이프라인은 다음과 같습니다:

```
[cuDF 데이터 로드]
        ↓
[전처리 및 Cross Validation 분할]
        ↓
[Hyperparameter 탐색 (Random Search)]
        ↓
[Dask-CUDA 기반 병렬 모델 학습]
        ↓
[OOF 예측 수집]
        ↓
[Stacked Ensemble 구성]
        ↓
[최종 모델 선택]
```

이 구조는 가능한 한 모든 연산을 GPU에서 수행하여  
CPU-GPU 간 데이터 이동을 최소화하는 것을 목표로 합니다.

---

## 3. GPU 병렬 처리 코드 예제

다음은 Dask-CUDA를 이용하여 멀티 GPU 환경을 구성하는 코드입니다:

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster(
    n_workers=2,
    device_memory_limit="12GB",
    rmm_pool_size="10GB"
)

client = Client(cluster)
```

이 코드에서 중요한 설정은 다음과 같습니다:

- device_memory_limit: GPU 메모리 사용 상한 설정 (OOM 방지)
- rmm_pool_size: 메모리 풀을 미리 할당하여 성능 향상

이를 통해 안정적인 병렬 실행이 가능합니다.

---

## 4. Memory-Aware Scheduling

기존 AutoML 시스템은 메모리를 고려하지 않고 작업을 실행합니다.  
이 방식은 GPU 환경에서는 매우 위험합니다.

따라서 우리는 Memory-Aware Scheduling을 도입합니다.

```python
def can_run_model(free_vram, estimated_vram):
    return free_vram > estimated_vram * 1.2

if can_run_model(free_vram, estimated_vram):
    run_model()
else:
    skip_or_queue()
```

이 방식은 다음과 같은 특징을 가집니다:

- 모델 실행 전에 VRAM을 검사
- 실행 가능 여부를 사전에 판단
- OOM을 원천적으로 방지

---

## 5. OOF Streaming 전략

AutoML에서 메모리를 많이 사용하는 부분 중 하나는 Level-One Data입니다.

모델 수가 증가할수록 다음과 같이 데이터가 커집니다:

- N × 모델 수

이를 해결하기 위해 Streaming 방식을 적용합니다:

```
모델 A → OOF 생성 → CPU로 이동
모델 B → OOF 생성 → CPU로 이동
...
필요 시에만 GPU로 다시 로드
```

이 전략을 통해 GPU 메모리 사용량을 크게 줄일 수 있습니다.

---

## 6. VRAM 사용 시각화

GPU 메모리 사용은 다음과 같은 패턴을 보입니다:

```
VRAM 사용량

24GB ┤              █████ Random Forest
20GB ┤       ████ XGBoost
16GB ┤
12GB ┤
 8GB ┤ 데이터 + 버퍼
 4GB ┤
 0GB ┼──────────────── 시간
```

여러 모델이 동시에 실행될 경우  
메모리 사용량이 급격히 증가하는 것을 확인할 수 있습니다.

---

## 7. 실험 결과

Credit Card Fraud 데이터셋을 사용한 실험 결과입니다.

- XGBoost 모델: AUC 약 0.998
- 앙상블 모델: 안정적인 성능 유지
- OOM 발생: 0건

또한 Meta Learner는 성능이 낮은 모델에 대해 자동으로 가중치를 0으로 설정하여  
효율적인 앙상블을 구성했습니다.

---

## 8. 핵심 인사이트

메모리 최적화는 직접적으로 정확도를 향상시키지는 않습니다.

하지만 다음과 같은 효과를 가져옵니다:

- 더 많은 모델을 실행할 수 있음
- 탐색 공간이 확장됨
- 더 좋은 모델을 찾을 확률 증가

👉 결과적으로 AutoML 성능이 향상됩니다.

---

## 9. 한계점

현재 시스템은 다음과 같은 한계를 가지고 있습니다:

- 메모리가 부족하면 skip만 가능
- cuML 및 XGBoost 내부 메모리는 제어 불가
- fine-grained memory 관리 미지원

---

## 10. 향후 연구 방향

### 단기
- CPU 기반 H2O와 성능 비교
- rmm 메모리 풀 전략 실험

### 중기
- Custom allocator 개발
- worker 단위 메모리 관리

### 장기
- GPU AutoML 전용 CUDA 커널 개발
- fine-grained memory paging 구현

---

## 11. 결론

본 연구의 핵심은 다음과 같습니다:

- GPU AutoML의 병목은 메모리이다
- 기존 시스템은 이를 고려하지 않았다
- Memory-Aware Scheduling을 통해 문제를 해결했다

👉 최종 메시지:

"메모리를 1급 자원으로 다루는 것이 GPU AutoML의 핵심이다"
