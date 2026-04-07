# Phase 2: GPU-Native Redesign

> H2O의 전략을 RAPIDS 생태계 위에서 재조립하고, Memory-Aware Scheduling을 핵심 기여로 설계한 단계.

## Key Insight

RAPIDS(cuDF, cuML, XGBoost GPU)가 개별 부품을 제공하지만,
이것들을 **AutoML 파이프라인으로 조립하고 메모리를 관리하는 것**은 아무도 하지 않았다.
우리의 핵심 기여는 "Memory-Aware Scheduling"이다.

## Documents

| Document | 내용 |
|----------|------|
| [RAPIDS 구현 전략](04_rapids_implementation_strategy.md) | 4-Phase 구현 로드맵, 기술 선택 근거 |
| [PRD](PRD_memory-aware-gpu-automl.md) | 정식 요구사항 정의서 (FR/NFR/기술 설계) |
| [아키텍처 매핑](architecture.md) | 연구 문서 → 코드 모듈 매핑 |
| [발표 내러티브](06_presentation_narrative.md) | 전체 논리 흐름 정리 (Problem → Evidence → Proposal) |

## Next Phase

설계를 바탕으로 구현하고 검증합니다 → [Phase 3: Results](../03-results/)

---

[Main README로 돌아가기](../../README.md)
