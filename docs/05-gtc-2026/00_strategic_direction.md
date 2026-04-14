# GTC 2026 전략 방향 전환

> **작성일**: 2026-04-14
> **의사결정권자**: 프로젝트 리드 + 박사급 공동연구자 2명
> **상위 문서**: [01_PRD.md](01_PRD.md)

---

## 1. 프로젝트 스코프 재정의

### 1-1. 이전 포지셔닝 (폐기)

H2O AutoML의 GPU 재구현 세미나 데모. RTX 4060 8GB 환경에서 PagedAutoML 실험.
범위: 메모리 관리 + 알고리즘 선택 + 기본 HPO.

### 1-2. 신규 포지셔닝

**RAPIDS 커널 기반 GPU-Native End-to-End AutoML 연구**.
기존 AutoML 프레임워크(H2O/AutoGluon/TPOT/LightAutoML)의 강점인
**Preprocessing 자동화**를 RAPIDS 생태계로 이식하여,
CPU-GPU pingpong 없는 완전 GPU 상주 파이프라인을 구축한다.

### 1-3. 타겟 발표 형태

- **GTC 2026 정규 세션 (45분 talk)**
- 보조 산출물: 공개 벤치마크 리포트, 오픈소스 코드, 기술 백서

---

## 2. 핵심 기여의 재배치

```
┌─────────────────────────────────────────────────────────────┐
│  이전 프레이밍: PagedAutoML                                   │
│    ├─ 메모리 관리 (★)                                        │
│    ├─ HPO                                                    │
│    └─ 모델 선택                                              │
├─────────────────────────────────────────────────────────────┤
│  신규 프레이밍: GPU-Native E2E AutoML on RAPIDS             │
│    ├─ Preprocessing Automation (★★★ 주 기여)                │
│    │   ├─ GPU 기반 Imputation                               │
│    │   ├─ GPU 기반 Categorical Encoding                     │
│    │   ├─ GPU 기반 Scaling/Normalization                    │
│    │   ├─ GPU 기반 Outlier Detection                        │
│    │   ├─ GPU 기반 Feature Selection                        │
│    │   ├─ GPU 기반 Feature Engineering                      │
│    │   └─ GPU 기반 Datetime / Text-lite Features            │
│    ├─ Model Selection (cuML RF/GLM, XGBoost GPU — 3종)      │
│    ├─ HPO (Ray Tune + ASHA — 인프라 조합)                   │
│    ├─ Memory Management (Ray Process 격리 — 인프라)         │
│    └─ Inference Serving (Dual Runtime — ONNX / Triton)      │
└─────────────────────────────────────────────────────────────┘
```

**심사자용 한 줄 기여 진술**:
> "RAPIDS 커널만으로 E2E preprocessing 자동화를 완결하여,
> 기존 AutoML 대비 X배 throughput을 유지하면서 동일 또는 우수한 모델 품질을 달성한다."

---

## 3. 확정된 주요 결정사항

| 항목 | 결정 내용 |
|:-----|:----------|
| 타임라인 | 2026-04-14 ~ 2026-07-14 (3개월) |
| 팀 구성 | **공동 1저자 3인 체제** (프로젝트 리드 + 박사급 공동연구자 2. equal contribution. 역할 Month 1 Week 1 내 확정) |
| 하드웨어 | **스폰서십 기반 H100 렌탈** (Month 2 Week 2 데모 Gate 후 확보). Month 1~2 Week 2는 자체 부담 저가 환경. |
| 비교 대상 Baseline | **H2O AutoML, AutoGluon, TPOT, LightAutoML** (4종) |
| Preprocessing 조사 범위 | 위 4종의 자동 전처리 파이프라인 전수 해부 |
| 모델링 백엔드 | cuML RF, cuML GLM, XGBoost GPU (3종 확정. LightGBM 제외) |
| HPO 백엔드 | Ray Tune + ASHA |
| 메모리 관리 | Ray Actor 기반 프로세스 격리 + Object Store |
| 추론 서빙 | **포함** (Framework 완결성). Dual Runtime (학습: cuDF/cuML, 추론: ONNX 또는 Triton — Month 2 Week 1 확정). |
| arXiv 백서 | **필수 deliverable** (Month 3 Week 4 제출 목표) |

---

## 4. 기존 자산의 재배치

### 4-1. 현재 `paged_manager.py` 처리 방침

**폐기하지 않고 논문의 "블랙박스 제약 증명" 서사로 편입**.

기존 바이트 수준 페이징 시도의 실패를 정량적으로 문서화하면, "왜 프로세스 수준 격리가 필연적인가"의 증거로 재활용 가능. 이는 Ray 도입의 motivation을 강화한다.

### 4-2. `docs/04-future/` 문서 처리 방침

| 문서 | 신규 위치 | 역할 |
|:-----|:----------|:-----|
| 01_paged_attention_overview.md | Appendix | 배경 이론 |
| 02_ray_process_level_paging.md | Infrastructure 섹션 | 조연 인프라 |
| 03_hyperband_asha_oom_prevention.md | Infrastructure 섹션 | 조연 인프라 |

`docs/05-gtc-2026/` 하위에 main narrative 문서들을 신설하고, 04-future는 참조 자료로만 유지한다.

### 4-3. 신규 작성 필요 문서

- [x] `00_strategic_direction.md` (본 문서)
- [x] `01_PRD.md` — 상세 요구사항 정의서
- [ ] `02_preprocessing_gap_analysis.md` — 4종 baseline 해부 결과 (Month 1)
- [ ] `03_gpu_native_operators_design.md` — 신규 operator 설계 명세 (Month 1~2)
- [ ] `04_benchmark_methodology.md` — 평가 방법론 (Month 2)
- [ ] `05_scaling_study_design.md` — Multi-GPU scaling 실험 설계 (Month 2)
- [ ] `06_serving_architecture_decision.md` — ONNX vs Triton 결정 (Month 2 Week 1)
- [ ] `07_gtc_talk_outline.md` — 발표 스토리라인 (Month 3)
- [ ] `08_arxiv_manuscript.md` / `arxiv/` — arXiv 백서 소스 (Month 3 Week 3~4)

---

## 5. 3개월 타임라인 요약

```
Month 1 (2026-04 ~ 2026-05): Gap Analysis + Architecture Design
  - Week 1-2: 4종 baseline preprocessing 해부
  - Week 3: CPU-GPU pingpong 비용 실측 + operator 카탈로그 확정
  - Week 4: 아키텍처 리뷰 + GTC abstract 초안

Month 2 (2026-05 ~ 2026-06): Core Implementation + 🚩 스폰서 데모
  - Week 1: P0 operators + 추론 경로 설계 워크샵 (ONNX vs Triton)
  - Week 2: P1 operators + E2E 스모크 테스트 + 🚩 스폰서 데모 Gate (H100 크레딧 확보)
  - Week 3: P2 operators + 추론 서빙 first cut + 2개 데이터셋 검증
  - Week 4: P3 operators + H100 환경 셋업

Month 3 (2026-06 ~ 2026-07): Benchmark + arXiv + GTC
  - Week 1: 5개 데이터셋 벤치마크 (single H100) + baseline 실행
  - Week 2: Ablation + Multi-GPU scaling (H100 1/2/4/5장)
  - Week 3: arXiv 백서 초안 + GTC talk outline
  - Week 4: arXiv 제출 + GTC proposal 제출 + 리허설
```

상세 마일스톤은 [01_PRD.md](01_PRD.md) §10 참조.

---

## 6. 성공 기준

### 6-1. 발표 수락(Acceptance) 기준

- GTC 2026 정규세션 프로포절 통과
- Reviewer 피드백에서 "novel contribution"으로 인정

### 6-2. 기술적 성공 기준

- 5개 이상 공개 데이터셋에서 4종 baseline 대비 E2E wall-clock **3배 이상** 단축
- 동일 조건에서 최종 모델 품질(AUC/Accuracy) **동등 이상**
- H100 1→4장 scaling efficiency **70% 이상**

### 6-3. 오픈소스 기여 기준

- 프로젝트 코드 오픈소스 공개 (Apache 2.0) — 명칭 Month 1 Week 2 확정
- 재현 가능한 벤치마크 스위트 동반 공개
- arXiv 백서 공개 (Month 3 Week 4)
