# Documentation

## Phase 1: Research — H2O AutoML 분석

H2O AutoML의 핵심 전략을 심층 분석한 연구 자료.

| 문서 | 내용 |
|------|------|
| [00_overview.md](01-research/00_overview.md) | 프로젝트 목표, 팀 역할, 문서 구성 |
| [01_stacked_ensemble_deep_dive.md](01-research/01_stacked_ensemble_deep_dive.md) | Stacking 원리, OOF, Two-Type Ensemble, GLM Meta Learner |
| [02_hyperparameter_tuning_deep_dive.md](01-research/02_hyperparameter_tuning_deep_dive.md) | Grid vs Random Search, H2O 훈련 순서, 시간 예산 제어 |
| [03_gpu_native_application.md](01-research/03_gpu_native_application.md) | H2O -> GPU 매핑, cuDF/cuML 대체, 병렬화 방법 |
| [05_memory_optimization_research.md](01-research/05_memory_optimization_research.md) | GPU VRAM 병목, rmm 프로파일링, Memory-Aware Scheduling 설계 |
| [260403-smKwon.pdf](01-research/260403-smKwon.pdf) | 팀원 분석 자료 (PDF) |

## Phase 2: Design — GPU 재설계

RAPIDS 생태계에서 H2O 전략을 재조립하는 설계 문서.

| 문서 | 내용 |
|------|------|
| [PRD_memory-aware-gpu-automl.md](02-design/PRD_memory-aware-gpu-automl.md) | 정식 요구사항 정의서 (FR/NFR, 성공 지표, 벤치마크 데이터셋) |
| [04_rapids_implementation_strategy.md](02-design/04_rapids_implementation_strategy.md) | 4-Phase 구현 로드맵, RAPIDS 라이브러리별 역할 |
| [architecture.md](02-design/architecture.md) | 연구 -> 코드 매핑, 설계 결정, vLLM 비교 |
| [06_presentation_narrative.md](02-design/06_presentation_narrative.md) | 발표 스토리라인, 벤치마크 근거, 예상 Q&A |

## Phase 3: Results — 구현 & 검증

실제 데이터로 벤치마크한 결과와 해석.

| 문서 | 내용 |
|------|------|
| [benchmark_interpretation.md](03-results/benchmark_interpretation.md) | 실험 결과 상세 해석 (초심자 가이드, Credit Card + Higgs 비교) |
| [limitations.md](03-results/limitations.md) | 현재 한계 6가지 (Memory-Aware, vLLM 차이, GLM 실패 등) |
| [future_directions.md](03-results/future_directions.md) | 개선 방향 3단계 (단기/중기/장기) |
| [PLAN_memory-aware-gpu-automl.md](03-results/PLAN_memory-aware-gpu-automl.md) | 구현 계획 + 실행 로그 (30/33 완료) |

## Phase 5: GTC 2026 — GPU-Native E2E AutoML

> **2026-04-14 스코프 전환**: 세미나 데모 → GTC 2026 정규세션 투고용 연구. RAPIDS 커널 기반 preprocessing 자동화를 핵심 기여로 재포지셔닝.

| 문서 | 내용 |
|------|------|
| [00_strategic_direction.md](05-gtc-2026/00_strategic_direction.md) | 전략 방향 전환, 포지셔닝, 4종 baseline 확정, 3개월 타임라인 |
| [01_PRD.md](05-gtc-2026/01_PRD.md) | 상세 요구사항 정의서 (Goals/Non-Goals, Operator 카탈로그, 벤치마크 방법론, 마일스톤, 리스크) |

## Other

| 문서 | 내용 |
|------|------|
| [troubleshooting.md](troubleshooting.md) | H2O + NumPy 2.x 호환성 등 알려진 이슈 |
