# 과제 개요 및 문서 가이드

> RAPIDS LAB 세미나 과제: H2O AutoML 기법 분석 및 GPU 네이티브 적용 방향

---

## 과제 목표

기존 AutoML 시스템(H2O)을 분석하고, 기능을 장점만 가려내서
GPU 기반의 AutoML 프레임워크를 오픈소스로 만드는 것이 핵심 목표.

구체적으로:
1. **Stacked Ensemble** 방법론 및 원리 조사
2. **Hyperparameter Tuning** 방법론 및 원리 조사
3. 위 두 가지를 **RAPIDS 기반 End-to-End GPU AutoML**에 어떻게 적용할 수 있는지 조사

---

## 우리 팀의 접근

H2O는 Java 기반이라 GPU 기반으로 전환하려면 C 기반 커널을 재설계해야 한다.
이건 단기간에 불가능.

**대신:** H2O의 **코드**가 아닌 **기법(설계 전략)**을 가져가고,
구현은 이미 존재하는 GPU 라이브러리(cuDF, cuML, XGBoost GPU)를 조합해서
커널 재작성 없이 적용한다.

---

## 역할 분담

| 담당 | 내용 |
|------|------|
| 권성민 | Stacked Ensemble 이론 + HPO 이론 + GPU 환경 과제 정리 |
| Harrison | H2O 기법의 GPU 네이티브 적용 방향 분석 (이 문서들) |
| 김선아 | (확인 필요) |

---

## 문서 구성

| 파일 | 내용 | 읽는 순서 |
|------|------|-----------|
| `00_overview.md` | 과제 개요 및 가이드 (이 문서) | 1 |
| `01_stacked_ensemble_deep_dive.md` | Stacking 원리, OOF, Meta Learner, Two-type Ensemble | 2 |
| `02_hyperparameter_tuning_deep_dive.md` | HPO 개념, Grid/Random Search, H2O 훈련 순서 | 3 |
| `03_gpu_native_application.md` | GPU 적용 방향, cuDF/cuML 매핑, 병렬화, 메모리 관리 | 4 |
| `260403-smKwon.pdf` | 팀원(권성민) 작성 자료 | 참고 |

---

## 노트북

| 파일 | 내용 |
|------|------|
| `notebooks/01_h2o_automl_demo.ipynb` | H2O AutoML 라이브 데모 |
| `notebooks/02_explainability.ipynb` | 모델 설명 가능성 시각화 |
| `notebooks/03_gpu_native_redesign.ipynb` | GPU 적용 방향 분석 (마크다운 + 비교표) |

---

## 핵심 한 줄 요약

> **"H2O의 기법은 가져가되, 구현은 GPU 네이티브로 새로 짠다"**
> — 이것이 가능한 이유는 RAPIDS(cuDF, cuML)가 H2O의 알고리즘 풀을 GPU로 이미 구현해놨기 때문이다.
