# Phase 1: H2O AutoML Analysis

> H2O AutoML의 핵심 전략을 분석하여 "무엇을 GPU로 가져올 것인가"를 결정한 단계.

## Key Insight

H2O AutoML의 가치는 Java 코드가 아니라 **10년간 검증된 전략**에 있다.
Stacked Ensemble, HPO 순서, Non-negative Meta Learner — 이 전략들은 실행 엔진과 독립적이다.

## Reading Order

| # | Document | 핵심 질문 |
|:-:|----------|----------|
| 0 | [과제 개요](00_overview.md) | 프로젝트의 목표와 범위는? |
| 1 | [Stacked Ensemble 분석](01_stacked_ensemble_deep_dive.md) | H2O의 앙상블은 어떻게 작동하는가? |
| 2 | [HPO 전략 분석](02_hyperparameter_tuning_deep_dive.md) | 왜 Random Search인가? 훈련 순서는? |
| 3 | [GPU 네이티브 적용 방향](03_gpu_native_application.md) | H2O 기법을 GPU에 어떻게 매핑하는가? |
| 4 | [메모리 최적화 연구](05_memory_optimization_research.md) | GPU VRAM 제약을 어떻게 극복하는가? |

**참고 자료**: [팀원 분석 자료 (PDF)](260403-smKwon.pdf)

## Next Phase

이 분석을 바탕으로 GPU 재설계 전략을 수립합니다 → [Phase 2: Design](../02-design/)

---

[Main README로 돌아가기](../../README.md)
