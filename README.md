<p align="center">
  <img src="assets/h2o-automl-logo.png" alt="H2O AutoML" width="400"/>
</p>

<h1 align="center">H2O AutoML</h1>

<p align="center">
  <strong>Pandas 이후, RAPIDS 이전 — 자동화와 병렬 처리의 교차점</strong><br/>
  RAPIDS LAB 세미나 시리즈: ML 프레임워크 히스토리
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9.4-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/JVM-Java%208~17-ED8B00?logo=openjdk&logoColor=white" alt="Java"/>
  <img src="https://img.shields.io/badge/General-Java%2017-2ea44f?logo=openjdk&logoColor=white" alt="General Java 17"/>
  <img src="https://img.shields.io/badge/Hadoop-Java%2011-orange?logo=apachehadoop&logoColor=white" alt="Hadoop Java 11"/>
  <img src="https://img.shields.io/badge/H2O-3.46.0.10-FFD700?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCI+PHRleHQgeT0iMjAiIGZvbnQtc2l6ZT0iMjAiPvCfkqc8L3RleHQ+PC9zdmc+&logoColor=white" alt="H2O"/>
  <img src="https://img.shields.io/badge/Package%20Manager-uv-DE5FE9?logo=uv&logoColor=white" alt="uv"/>
  <img src="https://img.shields.io/badge/pyenv-3.9.4-green?logo=pyenv&logoColor=white" alt="pyenv"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/XGBoost-GPU%20(CUDA%208+)-76B900?logo=nvidia&logoColor=white" alt="XGBoost GPU"/>
  <img src="https://img.shields.io/badge/sklearn-Compatible-F7931E?logo=scikitlearn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License"/>
</p>

---

## Compatibility Matrix

| Dependency | Required Version | Note |
|:----------:|:----------------:|:-----|
| **Python** | `3.9 ~ 3.11` (본 레포: `3.9.4`) | 본 프로젝트 지원 범위. H2O는 3.6부터 지원하나 `pyproject.toml`에서 `>=3.9,<3.12`로 제한 |
| **Java (JVM)** | `8 ~ 17` (본 레포: `17`) | H2O 런타임 엔진. General vs Hadoop에 따라 상이 (아래 참조) |
| **H2O** | `3.46.0.10` | 현재 최신 안정 버전 |
| **XGBoost GPU** | CUDA `8+` + NVIDIA GPU | 선택사항. CPU-only 환경에서도 동작 |
| **OS** | Linux / macOS (Intel) | Apple Silicon(M1+)에서 XGBoost 비활성 |
| **uv** | `0.1+` | Python 패키지 매니저 (pip 대체) |
| **pyenv** | `2.0+` | Python 버전 관리 |

### Java 버전: General vs Hadoop — 왜 나뉘는가?

H2O는 JVM 위에서 동작하기 때문에 Java 버전이 핵심 의존성입니다.
그런데 공식 문서를 보면 **두 가지 다른 Java 요구사항**이 존재합니다:

| 배포 형태 | 지원 Java 버전 | 권장 |
|:---------:|:--------------:|:----:|
| **General** (Standalone / Local) | `8, 9, 10, 11, 12, 13, 14, 15, 16, 17` | **17** |
| **Hadoop** (YARN / MapReduce) | `8, 11` 만 가능 | **11** |

#### 왜 Hadoop에서는 Java 17을 못 쓰는가?

이 제한은 **H2O의 문제가 아니라 Apache Hadoop 자체의 제약**입니다.

1. **Java 17의 Strong Encapsulation**: Java 17은 deprecated API를 제거하고 모듈 시스템을 강화했습니다. Hadoop 내부에서 사용하던 `sun.misc.Unsafe` 등의 internal JDK API 접근이 차단됩니다.

2. **Hadoop의 보수적 Java 정책**: Apache Hadoop 3.3.x (현재 주력 버전)는 공식적으로 Java 8(컴파일+런타임)과 Java 11(런타임만)만 지원합니다. Java 17 런타임 지원은 [HADOOP-18887](https://issues.apache.org/jira/browse/HADOOP-18887)에서 진행 중이나 아직 안정 릴리즈에 미포함입니다. (Hadoop 3.3.x / H2O 3.46.0.x 기준)

3. **H2O on Hadoop = Hadoop의 JVM 위에서 실행**: H2O가 Hadoop 클러스터에서 YARN Job으로 실행될 때, Hadoop 클러스터의 JVM 버전을 따라야 합니다. 즉, 클러스터가 Java 11이면 H2O도 Java 11로 실행됩니다.

```
┌─────────────────────────────────────────────────┐
│  General (Standalone)                           │
│  ┌───────────────────────────────────────────┐  │
│  │  H2O JVM (Java 8~17 선택 가능)            │  │
│  │  → 로컬 머신에서 직접 실행                 │  │
│  │  → Java 17 권장 (최신 성능 + GC 개선)     │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Hadoop (YARN/MapReduce)                        │
│  ┌───────────────────────────────────────────┐  │
│  │  Hadoop Cluster JVM (Java 8 or 11)        │  │
│  │  └─ H2O Worker (Hadoop의 JVM 버전 상속)   │  │
│  │     → Java 11 권장                        │  │
│  │     → Java 17 사용 불가 (Hadoop 미지원)    │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

#### H2O의 Java 지원 히스토리

| H2O 버전 | 시기 | Java 변경사항 |
|:--------:|:----:|:-------------|
| 3.30.1.1 | 2020.08 | Java 14 공식 지원 추가 |
| 3.32.1.1 | 2021.03 | Java 15 지원 추가 |
| 3.36.0.1 | 2021.12 | **Java 16, 17 공식 지원** (Gradle 7 업그레이드) |
| 3.46.0.x | 현재 | Docker 이미지 Java 11 → 17 전환. **Java 21은 미지원** |

> **본 레포는 Standalone(General) 환경**이므로 Java 17을 사용합니다.
> Hadoop 클러스터 환경에서 실행할 경우 Java 11로 전환이 필요합니다.

---

## Quick Start

### 0. 사전 요구사항 확인

```bash
# Java 8~17 확인 (General 환경 권장: 17 / 없으면 brew install openjdk@17)
java -version

# pyenv 확인 (없으면 brew install pyenv)
pyenv --version

# uv 확인 (없으면 curl -LsSf https://astral.sh/uv/install.sh | sh)
uv --version
```

### 1. 레포지토리 클론

```bash
git clone https://github.com/ModulabsRAPIDSLAB/H2O_AutoML.git
cd H2O_AutoML
```

### 2. Python 버전 설정 (pyenv)

```bash
# Python 3.9.4 설치 (이미 설치되어 있으면 생략)
pyenv install 3.9.4

# 프로젝트 로컬 버전 고정
pyenv local 3.9.4

# 확인
python --version
# → Python 3.9.4
```

### 3. 가상환경 생성 및 활성화 (uv)

```bash
# uv로 venv 생성 (pyenv의 3.9.4를 자동 감지)
uv venv --python 3.9.4

# 활성화
source .venv/bin/activate
```

### 4. 의존성 설치

```bash
# pyproject.toml에 정의된 의존성 일괄 설치
uv pip install .
```

### 5. 데이터셋 다운로드

```bash
# datasets 폴더 생성 + Kaggle 신용카드 사기거래 데이터셋 다운로드
# 출처: https://www.kaggle.com/datasets/kartik2112/fraud-detection
mkdir -p datasets
# Kaggle에서 credit_card_transactions.csv를 다운로드하여 datasets/ 폴더에 배치
# kaggle datasets download -d kartik2112/fraud-detection -p datasets/ --unzip
```

### 6. Jupyter Notebook 실행

```bash
jupyter notebook notebooks/
```

### 7. 데모 실행 순서

| 순서 | 노트북 | 내용 |
|:----:|:-------|:-----|
| 1 | `01_h2o_automl_demo.ipynb` | H2O 초기화 → 데이터 로드 → AutoML 학습 → 리더보드 → 결과 분석 |
| 2 | `02_explainability.ipynb` | `h2o.explain()` 시각화, 변수 중요도, 모델 상관관계 |

---

## H2O AutoML이란?

H2O AutoML은 **H2O.ai**에서 개발한 오픈소스 자동 머신러닝 프레임워크입니다.
사용자가 지정한 시간 또는 모델 수 제한 내에서 다양한 알고리즘을 **자동으로 학습, 튜닝, 앙상블**합니다.

### 핵심 포지셔닝

```
Pandas (단일 스레드)
    ↓ 데이터가 커지면서 한계
Scikit-learn (CPU, 제한적 병렬)
    ↓ 하이퍼파라미터 튜닝 + 모델 선택 자동화 필요
H2O AutoML (JVM 기반 분산 병렬 + 자동화)  ← 여기
    ↓ GPU 가속 필요
RAPIDS cuML (GPU 네이티브 병렬)
```

---

## 핵심 특징

### 1. JVM 기반 분산 병렬 처리
- Java/Scala 기반으로 **멀티코어 자동 활용**
- 단일 머신부터 **멀티노드 클러스터**까지 동일 코드로 확장
- H2O Frame: 자체 인메모리 데이터 구조로 대용량 데이터 처리

### 2. 완전 자동화된 ML 파이프라인
- **6가지 알고리즘** 자동 탐색: GBM, XGBoost, DRF, GLM, DeepLearning, StackedEnsemble
- 하이퍼파라미터 랜덤 그리드 서치 내장
- Cross-validation + Early Stopping 기본 탑재
- **Stacked Ensemble** 자동 생성 (Best of Family + All Models)

### 3. 시간/모델 기반 제어
- `max_runtime_secs`: 전체 탐색 시간 제한
- `max_models`: 탐색할 최대 모델 수
- 둘 다 설정 시 먼저 도달하는 조건에서 중단

### 4. 리더보드 자동 생성
- 5-fold CV 기반 모델 성능 순위표
- 이진분류(AUC), 다중분류(mean_per_class_error), 회귀(RMSE) 기준 자동 정렬
- 학습 시간, 예측 속도 등 부가 정보 제공

### 5. Explainability 내장
- `h2o.explain()` 한 줄로 다중 모델 비교 시각화
- Feature Importance, SHAP, Partial Dependence Plot 등

### 6. 프로덕션 배포 지원
- **MOJO**(Model Object, Optimized) 포맷으로 Java 기반 배포
- sklearn 호환 wrapper 제공 (`h2o.sklearn`)

---

## 언제 H2O AutoML을 사용하는가?

| 시나리오 | 적합도 | 이유 |
|----------|:------:|:-----|
| 빠른 베이스라인 모델 탐색 | **높음** | 코드 몇 줄로 다양한 알고리즘 비교 |
| 대용량 정형 데이터 (수 GB) | **높음** | JVM 분산 처리로 메모리 효율적 |
| 멀티코어 서버 활용 | **높음** | 자동 병렬화, 설정 불필요 |
| GPU 가속이 필수인 경우 | 보통 | XGBoost GPU만 지원, 나머지는 CPU |
| 딥러닝 위주 작업 | 낮음 | MLP만 지원, CNN/RNN 미지원 |
| 실시간 스트리밍 처리 | 낮음 | 배치 학습 중심 |

---

## AutoML 프레임워크 비교 & RAPIDS와의 관계

### AutoML이란?

AutoML은 **"어떤 모델을, 어떤 설정으로 학습할지"를 자동화**하는 기술입니다.
알고리즘 선택, 하이퍼파라미터 튜닝, 교차 검증, 앙상블까지 사람이 수동으로 하던 일을 대신합니다.

### AutoML 프레임워크 4종 비교

| | H2O AutoML | AutoGluon | TPOT | LightAutoML |
|:--|:----------:|:---------:|:----:|:-----------:|
| **만든 곳** | H2O.ai (2017) | AWS (2019) | UPenn (2015) | Sberbank (2021) |
| **한 줄 요약** | JVM 분산 병렬 + 완전 자동화 | 멀티모달 (표+이미지+텍스트) | 유전 알고리즘으로 파이프라인 진화 | 금융 특화, 극한의 속도 |
| **탐색 방법** | 랜덤 그리드 서치 | 베이지안 최적화 + Hyperband | 유전 프로그래밍 (GP) | 전문가 규칙 + TPE |
| **앙상블** | Stacked Ensemble 자동 | 멀티레이어 스태킹 | 없음 (단일 파이프라인) | 스태킹 |
| **데이터 유형** | 정형 데이터 | **정형 + 이미지 + 텍스트 + 시계열** | 정형 데이터 | 정형 데이터 |
| **분산 처리** | O (멀티노드) | X | X | X |
| **GPU** | 부분 (XGBoost만) | O (풀 지원) | X | 부분적 |
| **코드 내보내기** | MOJO (Java 배포) | SageMaker 연동 | **sklearn 코드 자동 생성** | 추론 API |
| **주요 강점** | 자동화 + 설명 가능성 | 멀티모달 + 다양한 알고리즘 | 파이프라인 구조 탐색 | 10분 내 실전급 모델 |

#### 각 프레임워크는 언제 쓰는가?

**H2O AutoML** — "뭘 써야 할지 모르겠고, 설명도 해야 할 때"
- 어떤 알고리즘이 좋을지 감이 없을 때 → 6가지 알고리즘 자동 탐색 + 앙상블
- 모델의 근거가 필요할 때 → `h2o.explain()` 한 줄로 SHAP, PDP 등
- 대용량 데이터를 CPU 클러스터에서 돌릴 때 → JVM 분산 병렬 자동 활용

**AutoGluon** — "표 데이터만이 아니라, 이미지/텍스트도 같이 다뤄야 할 때"
- 상품 이미지 + 설명 텍스트 + 가격표를 동시에 학습해야 할 때
- GPU 환경에서 빠르게 프로토타입을 만들고 싶을 때
- AWS 생태계 안에서 운영할 때

**TPOT** — "전처리부터 모델까지, 파이프라인 구조 자체를 실험하고 싶을 때"
- "PCA를 먼저? 스케일링을 먼저?" 같은 **파이프라인 순서**까지 자동 탐색
- 결과를 **재현 가능한 sklearn 파이썬 코드**로 받고 싶을 때
- 소~중규모 데이터에서 최적 파이프라인을 발견하고 싶을 때

**LightAutoML** — "시간 없고, 빠르게 실전 투입할 모델이 필요할 때"
- 10분 이내에 프로덕션 급 모델이 필요할 때
- 금융 데이터 (사기 탐지, 신용 평가) 등 검증된 도메인
- 베테랑의 경험 규칙이 코드에 녹아 있어, 탐색 낭비 없이 효율적

### RAPIDS는 AutoML이 아니다

위 4개 프레임워크는 모두 **"무엇을 학습할지 (What)"** 를 자동화합니다.
RAPIDS는 이와 다르게 **"얼마나 빨리 학습할지 (How fast)"** 를 해결합니다.

```
AutoML (H2O, AutoGluon, TPOT, LightAutoML)
  → "어떤 알고리즘? 어떤 파라미터? 어떤 조합?"  ← 탐색/자동화

RAPIDS (cuML, cuDF)
  → "정해진 걸 GPU로 10~100배 빠르게"           ← 가속
```

| 비교 축 | AutoML 프레임워크 | RAPIDS |
|---------|:-----------------:|:------:|
| 해결하는 문제 | 모델 선택 + 튜닝 자동화 | 연산 속도 가속 |
| 실행 환경 | CPU (H2O는 JVM, 나머지는 Python) | CUDA (GPU) |
| 데이터 구조 | 각자 고유 (H2O Frame, Pandas 등) | cuDF (GPU DataFrame) |
| 자동화 수준 | 높음 (알고리즘 선택부터 앙상블까지) | 없음 (개별 알고리즘 수동 구성) |
| 주요 강점 | **자동화** | **속도** |

### AutoML + RAPIDS = 함께 쓰면?

```
┌──────────────────────────────────────────────────────────────┐
│  1단계: AutoML로 최적 모델/파이프라인 탐색 (What)              │
│         → "GBM이 최고, max_depth=8, lr=0.05"                 │
│                                                              │
│  2단계: RAPIDS로 해당 모델을 GPU 가속 학습 (How fast)          │
│         → 동일 파이프라인을 10배 빠르게, 대규모 데이터에 적용    │
└──────────────────────────────────────────────────────────────┘
```

> AutoML이 **"최적의 레시피"** 를 찾아주고,
> RAPIDS가 **"업소용 화력으로 대량 조리"** 를 담당합니다.
> **탐색(What)과 가속(How fast)은 단계가 다른 상호 보완적 관계**입니다.

### 타임라인으로 보는 위치

```
2012  H2O 등장       — CPU 대규모 ML의 사실상 표준
2015  TPOT 등장      — 유전 프로그래밍으로 파이프라인 자동 발견
2017  H2O AutoML     — 자동화의 완성 (모델 선택 + 튜닝 + 앙상블)
2018  RAPIDS 등장    — GPU 가속으로 속도 혁신, AutoML의 "속도 우위" 대체
2019  AutoGluon      — AWS 기반 멀티모달 AutoML
2021  LightAutoML    — 금융 실전에서 검증된 초고속 AutoML
```

---

## 프로젝트 구조

```
H2O_AutoML/
├── README.md                          # 발표 개요 및 핵심 정리
├── notebooks/
│   ├── 01_h2o_automl_demo.ipynb       # 라이브 데모 (데이터 로드 → AutoML → 결과 분석)
│   └── 02_explainability.ipynb        # 모델 설명 가능성 시각화
├── docs/
│   └── troubleshooting.md             # 알려진 이슈 및 해결 방법
├── datasets/                          # 데이터셋 (gitignore)
│   └── credit_card_transactions.csv   # Kaggle 신용카드 사기거래 (~129만 건)
├── assets/                            # 발표용 이미지
│   └── h2o-automl-logo.png
├── main.py
├── pyproject.toml
└── .python-version                    # 3.9.4
```

---

## 발표 흐름 (10분)

### Phase 1 — 런타임 시작 (1분)
H2O 클러스터 초기화 후 Kaggle 신용카드 사기거래 데이터셋(~129만 건)으로 AutoML 학습 시작.
`max_runtime_secs=300`으로 설정하여 약 5분간 백그라운드 학습.

### Phase 2 — H2O AutoML 설명 (5분)
학습이 돌아가는 동안:
1. **배경**: 왜 AutoML이 필요한가? (Pandas의 한계, 전문가 부족)
2. **핵심 특징**: JVM 병렬, 6가지 알고리즘, Stacked Ensemble
3. **RAPIDS 이전의 위치**: CPU 병렬 + 자동화의 정점
4. **실무 시나리오**: 베이스라인 탐색, sklearn 연동, MOJO 배포

### Phase 3 — 결과 리뷰 (4분)
학습 완료 후:
1. **리더보드 분석**: 모델별 성능, 학습 시간 비교
2. **최고 모델 분석**: Feature Importance, 파라미터 확인
3. **Explainability**: `h2o.explain()` 시각화

---

## Troubleshooting

알려진 이슈와 해결 방법은 [docs/troubleshooting.md](docs/troubleshooting.md)를 참고하세요.

---

## 참고 자료

- [H2O AutoML 공식 문서](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [H2O GitHub](https://github.com/h2oai/h2o-3)
- LeDell, E., & Poirier, S. (2020). *H2O AutoML: Scalable Automatic Machine Learning.* 7th ICML Workshop on AutoML.
- H2O AutoML 최초 릴리즈: 2017년 6월 6일 (H2O 3.12.0.1)
