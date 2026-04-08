# Limitations — 현재 프로젝트의 한계

## 1. Memory-Aware는 "skip"만 할 수 있다

현재 Memory-Aware Scheduling은 훈련 전에 VRAM을 체크하고, 부족하면 해당 모델을 skip한다.
vLLM처럼 "메모리를 재배치해서 실행 가능하게 만드는" 능력이 없다.
Higgs(5M rows)에서 rmm pool이 4GB를 선점하면 **모든 모델이 skip되어 AutoML 자체가 불가능**해진다.

## 2. PagedMemoryManager는 아직 이론이다

`paged_manager.py`에 블록 할당/회수/LRU eviction/swap 코드가 있지만,
실제 AutoML 파이프라인에서 사용한 적이 없다.
`automl.py`에서 `paged_memory=False`가 기본값이고, 벤치마크도 이 기능 없이 실행되었다.

## 3. cuML/XGBoost 내부를 제어할 수 없다

vLLM은 Attention 커널을 직접 작성하여 비연속 메모리를 참조할 수 있다.
우리는 cuML RF, XGBoost를 블랙박스로 사용하므로, **모델 훈련 도중의 메모리를 제어할 수 없다**.
이것이 "task-level coarse-grained paging"에 머무는 근본적 이유이다.

## 4. CPU H2O와의 비교가 없다

"GPU가 빠르다"는 주장에 대한 직접적 근거가 없다.
같은 데이터, 같은 시간 예산으로 H2O CPU를 돌린 비교 실험을 수행하지 않았다.

## 5. GLM이 불완전하다

cuML LogisticRegression은 극도의 클래스 불균형에서 L-BFGS가 수렴에 실패한다.
H2O의 GLM은 이를 처리하므로, "H2O 전략의 완전한 재현"이라고 할 수 없다.
3개 알고리즘 중 1개가 특정 조건에서 동작하지 않는 것이다.

## 6. vLLM 수준의 Paging이 아니다

| | vLLM PagedAttention | 현재 PagedAutoML |
|:--|:---------------------|:-----------------|
| 관리 시점 | 연산 **도중** (Attention 커널 안) | 연산 **사이** (task 시작/끝) |
| 관리 단위 | token 16개 (수 KB) | model 1개 (수 GB) |
| 핵심 기술 | 커스텀 CUDA 커널로 비연속 메모리 접근 | Python에서 블록 수 확인 후 할당/해제 |
| GPU 메모리 활용률 | 20% -> **96%** (논문 검증) | **미검증** |

현재 구현의 실체: 일반적인 메모리 풀 관리 + LRU eviction이며,
"vLLM-inspired"는 **방향성**을 의미하고 현재 구현이 vLLM 수준이라는 뜻이 아니다.
