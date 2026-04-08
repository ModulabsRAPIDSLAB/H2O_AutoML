# HyperBand & ASHA: OOM-Free 하이퍼파라미터 튜닝 전략

> PagedAutoML의 Phase B(Diversity), Phase C(Random Search)에서 발생하는 GPU OOM 문제를
> HyperBand/ASHA 기반 Early Stopping으로 해결하는 방법을 상세히 분석한다.

---

## 1. 기존 HPO의 문제: 왜 OOM이 발생하는가

### 1-1. PagedAutoML의 3-Phase 전략 요약

```
Phase A (Baseline)     : 알고리즘별 기본 모델 1개씩 훈련
Phase B (Diversity)    : 사전 정의된 하이퍼파라미터 그리드 탐색
Phase C (Random Search): 시간 예산 내 랜덤 HP 샘플링
```

Phase A는 모델 수가 적어 메모리 문제가 거의 없다.
**문제는 Phase B와 C에서 발생한다.**

### 1-2. Phase B/C에서 OOM이 발생하는 구조적 원인

Phase B에서 XGBoost의 그리드를 탐색한다고 가정하자.

```
max_depth     = [3, 6, 9, 12]
learning_rate = [0.01, 0.05, 0.1, 0.3]
n_estimators  = [100, 300, 500]
───────────────────────────────────
총 조합: 4 × 4 × 3 = 48개 모델
```

GPU 환경(VRAM 16 ~ 24GB)에서 이 48개 모델을 동시에 혹은 순차적으로 훈련하면:

```
문제 1: 동시 훈련 시 메모리 폭발
  - 모델 1개당 VRAM 사용: 1 ~ 4GB (데이터 크기, 모델 복잡도에 따라)
  - 4개 동시 훈련: 4 ~ 16GB → VRAM 한계 도달
  - 특히 max_depth=12, n_estimators=500 같은 대형 모델이 포함되면 즉시 OOM

문제 2: 순차 훈련 시 시간 낭비
  - 48개를 하나씩 훈련하면 시간 예산 초과
  - "나쁜 모델"도 끝까지 훈련하느라 자원 낭비

문제 3: Phase C (Random Search)의 무한 확장
  - Phase C는 시간이 남는 한 계속 새 모델을 생성
  - 랜덤 샘플링으로 "극단적" 하이퍼파라미터 조합 등장
  - batch_size=2048 + max_depth=15 같은 조합 → 단일 모델도 OOM 가능
```

### 1-3. Random Search의 메모리 사용 패턴

Random Search는 모든 trial을 동등하게 취급한다. 각 trial이 전체 리소스(에포크)를 소비한다.

```
                    Random Search 메모리 사용 패턴
VRAM
 24GB ┤
      │  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
 20GB ┤  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
      │  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
 16GB ┤  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
      │  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
 12GB ┤  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
      │  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
  8GB ┤  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
      │  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
  4GB ┤  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
      │  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
  0GB ┼──T1───T2───T3───T4───T5───T6───T7───T8───T9───T10──→ 시간
      │  (모든 trial이 전체 예산을 사용)

      → 나쁜 trial도 끝까지 실행
      → 동시 실행 시 VRAM 피크가 지속적으로 높음
      → OOM 위험 구간이 전체 구간에 걸쳐 존재
```

**핵심 문제**: "어차피 성능이 안 나올 모델"에 동일한 자원을 투입하는 것은 낭비이다.
이것을 해결하는 것이 **Successive Halving**, 그리고 이를 확장한 **HyperBand**와 **ASHA**이다.

---

## 2. Successive Halving 알고리즘 (SHA)

### 2-1. 핵심 아이디어

> "많은 모델을 조금씩 훈련시키고, 나쁜 모델은 일찍 죽여라."

전체 리소스 예산 B가 주어졌을 때, 모든 모델에 B/n 씩 균등하게 배분하는 대신,
**초반에 많은 모델을 저예산으로 평가하고, 상위 1/eta만 승격시키며 예산을 늘려가는** 방식이다.

### 2-2. 동작 원리 (단계별)

```
단계 0: n개의 랜덤 하이퍼파라미터 설정을 샘플링한다.
        각 설정에 최소 리소스 r을 배정하여 훈련한다.
        성능을 평가한다.

단계 1: 상위 ⌊n/η⌋개만 남기고 나머지를 제거(kill)한다.
        생존한 설정에 r·η 리소스를 배정하여 추가 훈련한다.
        성능을 재평가한다.

단계 2: 상위 ⌊n/η²⌋개만 남기고 나머지를 제거한다.
        생존한 설정에 r·η² 리소스를 배정하여 추가 훈련한다.
        ...

단계 k: 1개(또는 소수)의 최종 설정이 최대 리소스 R까지 훈련된다.
```

### 2-3. 수학적 공식

**입력 파라미터:**

| 기호 | 의미 | 예시 |
|------|------|------|
| `R` | 단일 설정에 할당 가능한 최대 리소스 (에포크, 반복 수 등) | 81 |
| `η` (eta) | 줄임 비율. 매 단계에서 1/η만 생존 | 3 |
| `n` | 초기 설정 수 | 81 |

**단계 i에서의 설정 수 n_i 와 리소스 r_i:**

```
n_i = n · η^(-i)        (= ⌊n / η^i⌋)

r_i = r · η^i
```

여기서 `r`은 최소 리소스(시작 리소스)이다.

**총 단계 수:**

```
단계 수 = ⌊log_η(n)⌋ + 1
```

**불변량(Invariant):**

각 단계에서의 총 리소스 사용량은 대략 일정하다.

```
n_i × r_i = (n · η^(-i)) × (r · η^i) = n · r = 상수
```

이것이 SHA의 핵심 속성이다. 매 단계에서 **동일한 총 연산량**을 사용하면서도,
모델 수는 줄어들고 개별 모델의 훈련 깊이는 늘어난다.

### 2-4. 구체적 예시: R=81, η=3

R=81 에포크, η=3 (매 라운드 상위 1/3만 생존), n=81개 초기 설정.

```
┌──────┬────────────────┬─────────────────┬───────────────────┐
│ 단계 │ 설정 수 (n_i)  │ 리소스 (r_i)    │ 총 리소스 (n_i×r_i)│
├──────┼────────────────┼─────────────────┼───────────────────┤
│  0   │ 81             │  1 에포크       │  81               │
│  1   │ 27             │  3 에포크       │  81               │
│  2   │  9             │  9 에포크       │  81               │
│  3   │  3             │ 27 에포크       │  81               │
│  4   │  1             │ 81 에포크       │  81               │
└──────┴────────────────┴─────────────────┴───────────────────┘
                                      총 사용 리소스: 405
```

**메모리 관점에서의 변화:**

```
단계 0: 81개 모델 × 1 에포크 → GPU에 81개의 작은 모델
단계 1: 27개 모델 × 3 에포크 → GPU에 27개 (54개 제거됨!)
단계 2:  9개 모델 × 9 에포크 → GPU에 9개
단계 3:  3개 모델 × 27 에포크 → GPU에 3개
단계 4:  1개 모델 × 81 에포크 → GPU에 1개

→ 단계가 진행될수록 동시 활성 모델 수가 급감
→ OOM 위험이 단계적으로 줄어든다
```

### 2-5. SHA의 시각적 표현

```
단계 0 (81개, 1에포크):
████████████████████████████████████████████████████████████████████████████████████

단계 1 (27개, 3에포크):       ← 하위 2/3 제거
███████████████████████████

단계 2 (9개, 9에포크):        ← 하위 2/3 제거
█████████

단계 3 (3개, 27에포크):       ← 하위 2/3 제거
███

단계 4 (1개, 81에포크):       ← 하위 2/3 제거
█  ← 최종 승자
```

---

## 3. HyperBand

### 3-1. Successive Halving의 한계: n vs B/n 트레이드오프

SHA에는 근본적인 딜레마가 있다.

주어진 총 예산 B에서 초기 설정 수 n을 정해야 한다.

```
n이 크면 (Exploration 우선):
  - 많은 설정을 탐색할 수 있다 (좋음)
  - 각 설정에 할당되는 최소 리소스 r = B/(n·s_max)이 매우 작아진다 (나쁨)
  - 1 에포크만으로는 좋은 설정인지 판단하기 어려울 수 있다

n이 작으면 (Exploitation 우선):
  - 각 설정에 충분한 리소스가 할당된다 (좋음)
  - 탐색하는 설정 수가 적어 최적값을 놓칠 수 있다 (나쁨)
```

**예시로 보는 트레이드오프:**

```
예산 B = 81, R = 81, η = 3

선택 1: n=81, r=1   → 81개 설정을 각 1에포크로 시작 (aggressive)
선택 2: n=27, r=3   → 27개 설정을 각 3에포크로 시작 (balanced)
선택 3: n=9,  r=9   → 9개 설정을 각 9에포크로 시작 (conservative)
선택 4: n=3,  r=27  → 3개 설정을 각 27에포크로 시작 (very conservative)
선택 5: n=1,  r=81  → 1개 설정을 81에포크로 시작 (= Random Search)

어떤 것이 최선인지는 사전에 알 수 없다!
```

### 3-2. 브라켓 시스템: 모든 트레이드오프를 동시에 시도

HyperBand의 핵심 통찰:

> "어떤 n이 최선인지 모르겠으면, **여러 개를 동시에 돌려라**."

위의 5가지 선택을 각각 하나의 **브라켓(bracket)** 으로 만들어 병렬 실행한다.
이것이 HyperBand의 핵심이다.

### 3-3. 수학적 정의

**s_max 계산:**

```
s_max = ⌊log_η(R)⌋
```

여기서 R은 단일 설정에 할당 가능한 최대 리소스이다.

R=81, η=3인 경우:
```
s_max = ⌊log_3(81)⌋ = ⌊4⌋ = 4
```

총 브라켓 수 = s_max + 1 = 5 (s=4, 3, 2, 1, 0)

**총 예산:**

```
B = (s_max + 1) × R
```

R=81인 경우: B = 5 × 81 = 405

**각 브라켓 s에서의 초기 설정 수 n과 최소 리소스 r:**

```
n = ⌈(B / R) / (s + 1)⌉ × η^s

r = R × η^(-s)
```

### 3-4. 전체 알고리즘 수도코드

```
HyperBand 알고리즘
─────────────────────────────────────────────────────
입력: R (최대 리소스), η (줄임 비율)
─────────────────────────────────────────────────────

s_max = ⌊log_η(R)⌋
B = (s_max + 1) × R

for s = s_max, s_max-1, ..., 1, 0 do          ← 외부 루프: 각 브라켓
    n = ⌈(B/R)/(s+1)⌉ × η^s                   ← 초기 설정 수
    r = R × η^(-s)                             ← 최소 리소스

    // ── Successive Halving (내부 루프) ──
    T = get_random_hyperparameter_configurations(n)

    for i = 0, 1, ..., s do                    ← SHA 단계
        n_i = ⌊n × η^(-i)⌋
        r_i = r × η^i

        L = { run_then_return_val_loss(r_i, t) : t ∈ T }

        T = top_k(T, L, ⌊n_i / η⌋)           ← 상위 1/η만 생존
    end for

    return 최적 설정 from T
end for
```

### 3-5. R=81, η=3 전체 브라켓 테이블

```
┌────────┬──────┬──────────────────────────────────────────────────────┐
│ 브라켓 │  s   │  단계별 (n_i 설정 × r_i 리소스)                     │
│        │      │  단계0    단계1    단계2    단계3    단계4           │
├────────┼──────┼──────────────────────────────────────────────────────┤
│   0    │  4   │  81×1  → 27×3  →  9×9  →  3×27 →  1×81           │
│   1    │  3   │  27×3  →  9×9  →  3×27 →  1×81                   │
│   2    │  2   │   9×9  →  3×27 →  1×81                           │
│   3    │  1   │   6×27 →  2×81                                    │
│   4    │  0   │   5×81                                            │
└────────┴──────┴──────────────────────────────────────────────────────┘

브라켓 0 (s=4): 최대 탐색 (81개 설정, 매우 낮은 초기 리소스)
브라켓 4 (s=0): 최소 탐색 (5개 설정, 최대 리소스로 직접 실행)
```

**각 브라켓의 총 리소스 사용량:**

```
브라켓 0: 81 + 81 + 81 + 81 + 81  = 405
브라켓 1: 81 + 81 + 81 + 81       = 324
브라켓 2: 81 + 81 + 81            = 243
브라켓 3: 162 + 162               = 324
브라켓 4: 405                     = 405

전체 HyperBand 1 round 총 리소스 ≈ 1701
```

### 3-6. 브라켓별 특성 비교

```
   탐색 범위 (넓음)                    모델 품질 (높음)
   ◀──────────────────────────────────────────────▶
   
   브라켓0    브라켓1    브라켓2    브라켓3    브라켓4
   81설정     27설정     9설정      6설정      5설정
   1에포크    3에포크    9에포크    27에포크   81에포크
   
   "많이 맛보기"                  "소수를 깊이 훈련"
   
   → 초반 성능으로 판별 가능한     → 충분한 훈련이 필요한
     하이퍼파라미터에 유리           하이퍼파라미터에 유리
     (예: learning_rate)           (예: regularization)
```

---

## 4. ASHA (비동기 Successive Halving)

### 4-1. 동기 방식의 straggler 문제

표준 SHA와 HyperBand는 **동기(synchronous)** 방식이다.
각 단계(rung)에서 **모든 trial이 완료될 때까지 다음 단계로 진행하지 않는다.**

```
동기 SHA: straggler 문제

Worker 1: ████████████████████ Trial A (빠름)  [대기..........]  다음 단계
Worker 2: █████████████████████████████████████ Trial B (느림)   다음 단계
Worker 3: ████████████████ Trial C (빠름)      [대기..........]  다음 단계
Worker 4: ██████████████████████ Trial D (중간) [대기..........]  다음 단계
                                                ↑
                                        Trial B를 기다리느라
                                        Worker 1,3,4가 놀고 있음!
```

**straggler(느림보) 문제:**
- GPU 환경에서 하이퍼파라미터에 따라 훈련 시간이 크게 다르다
- `max_depth=3`인 모델은 5분, `max_depth=15`인 모델은 30분 걸릴 수 있다
- 동기 방식에서는 가장 느린 trial을 기다려야 하므로 GPU 활용률이 떨어진다
- 멀티 GPU 환경에서 이 문제가 더욱 심각해진다

### 4-2. ASHA의 비동기 승격 메커니즘

ASHA (Asynchronous Successive Halving Algorithm, Li et al., 2020)는
동기화 지점을 완전히 제거한다.

**핵심 규칙:**

> 현재 rung에서 η개 이상의 관측값이 모이면,
> 그 중 상위 1/η를 **즉시** 다음 rung으로 승격시킨다.
> 모든 trial을 기다리지 않는다.

```
ASHA: 비동기 승격

Worker 1: ████ Trial A → 즉시 평가 → 승격! → ████████ Trial A (rung 2)
Worker 2: ██████████ Trial B → 평가 → 탈락
Worker 3: ████ Trial C → 즉시 평가 → 대기... → ████ Trial E (새 trial!)
Worker 4: ████████ Trial D → 평가 → 탈락

                  ↑ Worker가 비면:
                    1. 승격 가능한 trial이 있으면 → 승격시켜 다음 rung 실행
                    2. 없으면 → 새로운 trial을 rung 0에서 시작
                    → Worker가 놀지 않는다!
```

### 4-3. ASHA 수도코드

```
ASHA 알고리즘
─────────────────────────────────────────────────────
입력: R (최대 리소스), η (줄임 비율), r_min (최소 리소스)
Rung 레벨: r_0=r_min, r_1=r_min·η, r_2=r_min·η², ...
─────────────────────────────────────────────────────

repeat (Worker가 available할 때):
    1. 승격 가능한 trial이 있는지 확인:
       - 각 rung k를 위에서부터 검사
       - rung k에 η개 이상의 완료된 trial이 있고,
         그 중 상위 1/η가 아직 승격되지 않았으면:
         → 해당 trial을 rung k+1로 승격
         → 리소스 r_{k+1}까지 추가 훈련

    2. 승격할 trial이 없으면:
       - 새로운 랜덤 설정을 샘플링
       - rung 0 (r_min 리소스)에서 훈련 시작

until 총 예산 소진 또는 최대 trial 수 도달
```

### 4-4. 동기 SHA vs ASHA 비교

```
┌──────────────────┬─────────────────────┬─────────────────────┐
│ 항목             │ 동기 SHA            │ ASHA                │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 동기화 지점      │ 매 rung마다 대기    │ 없음                │
│ Worker 활용률    │ 낮음 (straggler)    │ 높음 (항상 활동)    │
│ 승격 판단 시점   │ 전체 완료 후        │ η개 관측 즉시       │
│ 승격 품질        │ 정확 (전체 비교)    │ 약간 suboptimal     │
│ 병렬 확장성      │ 제한적              │ 매우 우수            │
│ 구현 복잡도      │ 단순                │ 중간                │
│ GPU 유휴 시간    │ 많음                │ 거의 없음           │
│ 적합 환경        │ 균일한 모델 크기    │ 이기종 모델 혼합    │
└──────────────────┴─────────────────────┴─────────────────────┘
```

### 4-5. ASHA의 suboptimal 승격이 괜찮은 이유

ASHA는 모든 trial을 보지 않고 승격을 결정하므로 이론적으로 suboptimal 한 결정을 할 수 있다.
하지만 실제로는 영향이 미미하다.

1. **하이퍼파라미터 랭킹의 일관성**: 초기 rung에서 좋은 설정은 후기 rung에서도 대체로 좋다.
   즉, 조기 평가의 순위와 최종 순위 간 상관관계가 높다.

2. **Rung의 자기 교정**: 시간이 지나면 각 rung에 더 많은 관측값이 쌓이므로,
   이후의 승격 결정은 점점 더 정확해진다.

3. **실증적 검증**: Li et al. (2020)의 실험에서 ASHA는 동기 SHA/BOHB보다
   동일 시간 내에 더 좋은 최종 성능을 달성했다. Worker 활용률 향상이
   suboptimal 승격의 손실을 크게 상쇄한다.

---

## 5. OOM 방지 메커니즘 상세 분석

### 5-1. Random Search vs HyperBand 메모리 사용 프로파일 비교

**Random Search: 모든 trial이 전체 리소스를 소비**

```
         Random Search — 메모리 사용 타임라인
         (10개 trial, 각 81 에포크, VRAM 24GB)

VRAM
 24GB ┤ ··· OOM 위험선 ·····················································
      │                                                                     
 20GB ┤  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
      │  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
 16GB ┤  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
      │  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
 12GB ┤  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
      │  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
  8GB ┤  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
      │  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
  4GB ┤  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
      │  ████  ████  ████  ████  ████  ████  ████  ████  ████  ████
  0GB ┼──T1────T2────T3────T4────T5────T6────T7────T8────T9────T10──→ 시간
      
      모든 trial이 동일 VRAM을 동일 시간 동안 점유.
      나쁜 trial도 81 에포크를 전부 소비한다.
      총 리소스: 10 × 81 = 810 에포크
```

**HyperBand (브라켓 0, s=4): Early Stopping으로 메모리가 단계적으로 감소**

```
         HyperBand — 메모리 사용 타임라인
         (브라켓 0: 81개 → 27개 → 9개 → 3개 → 1개, η=3)

VRAM
 24GB ┤ ··· OOM 위험선 ·····················································
      │
 20GB ┤
      │                    
 16GB ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 단계 0: 81개 모델 (짧게 훈련)
      │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     BUT 각 모델이 1에포크만 실행
 12GB ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     → 실제 동시 VRAM 점유는 낮음
      │                       (순차 또는 소규모 병렬)
 10GB ┤          ████████  ← 단계 1: 27개 모델 × 3에포크
      │          ████████
  8GB ┤
      │               ████  ← 단계 2: 9개 모델 × 9에포크
  6GB ┤               ████
      │
  4GB ┤                  ██  ← 단계 3: 3개 모델 × 27에포크
      │
  2GB ┤                   █  ← 단계 4: 1개 모델 × 81에포크
      │
  0GB ┼──────────────────────────────────────────────────────────→ 시간
      
      단계가 진행될수록 활성 모델 수가 1/3로 줄어듦.
      VRAM 피크가 점진적으로 낮아짐.
      나쁜 모델은 1 에포크 만에 제거됨.
      총 리소스: 81+81+81+81+81 = 405 에포크 (Random Search 대비 50%)
```

### 5-2. 왜 각 rung에서 메모리가 줄어드는가

메모리가 줄어드는 메커니즘을 구체적으로 분석한다.

**메커니즘 1: 활성 모델 수의 감소**

```
GPU에 동시에 적재되는 모델 수:

Rung 0: 최대 81개 (하지만 각각 1에포크만 실행 → 빠르게 순환)
Rung 1: 최대 27개
Rung 2: 최대 9개
Rung 3: 최대 3개
Rung 4: 최대 1개

→ 각 rung에서 모델 수가 η=3 배씩 감소
→ 동시 활성 모델의 총 VRAM 점유량이 rung마다 약 1/3로 감소
```

**메커니즘 2: 대형 모델의 조기 탈락**

```
VRAM을 많이 사용하는 "위험한" 하이퍼파라미터 조합:
  - max_depth=15, n_estimators=2000   → 단일 모델 4GB
  - batch_size=2048, hidden_layers=4  → 단일 모델 3GB

이런 모델이 성능까지 나쁘다면:
  Random Search: 81 에포크를 전부 소비하며 4GB를 계속 점유
  HyperBand:     1 에포크 후 탈락 → 4GB 즉시 해제!
```

**메커니즘 3: 병렬도(concurrency) 제어와의 시너지**

```
VRAM 24GB, 모델당 평균 2GB 가정:

Random Search (동시 실행 제한 없을 때):
  12개 모델 동시 실행 → 24GB → OOM 위험

HyperBand Rung 3 (3개 모델만 활성):
  3개 모델 동시 실행 → 6GB → 여유 18GB
  → 다른 작업이나 데이터 로딩에 활용 가능
```

### 5-3. Phase 구조와의 결합: 메모리 안전 보장

PagedAutoML의 Phase 구조에 HyperBand를 결합하면 다음과 같은 메모리 프로파일이 가능하다.

```
    PagedAutoML + HyperBand 메모리 프로파일 (예상)

VRAM
 24GB ┤ ··· VRAM 한계 ······················································
      │
 20GB ┤
      │
 16GB ┤                  ▓▓▓▓                                    ▓▓▓▓
      │                  ▓▓▓▓  Phase B                           ▓▓▓▓
 14GB ┤  ████            ▓▓▓▓  Rung 0                            ▓▓▓▓
      │  ████            ▓▓▓▓  (많은 모델,                       ▓▓▓▓
 12GB ┤  ████  Phase A   ▓▓▓▓   적은 리소스)           Phase C   ▓▓▓▓
      │  ████  (기본      ▓▓▓▓                         Rung 0    ▓▓▓▓
 10GB ┤  ████   모델)        ▓▓▓▓  Rung 1                  ▓▓▓▓
      │  ████                ▓▓▓▓                          ▓▓▓▓
  8GB ┤  ████                    ▓▓  Rung 2                    ▓▓
      │                          ▓▓                            ▓▓
  6GB ┤                           ▓  Rung 3                     ▓
      │                                                          
  4GB ┤                                          ████████████████████
      │                                          Stacked Ensemble
  2GB ┤
      │
  0GB ┼─────────────────────────────────────────────────────────────→ 시간
          Phase A         Phase B              Phase C     Ensemble
      
      → OOM 위험 구간(16GB 이상)이 극히 짧음
      → 대부분의 시간에서 VRAM 여유가 충분
```

---

## 6. Ray Tune에서의 구현

### 6-1. ASHAScheduler API 상세

Ray Tune에서 ASHA는 `ASHAScheduler` (= `AsyncHyperBandScheduler`)로 제공된다.

```python
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    time_attr="training_iteration",  # 시간 축 (기본: 'training_iteration')
    metric="val_loss",               # 최적화 대상 메트릭
    mode="min",                      # 'min' 또는 'max'
    max_t=100,                       # 최대 리소스 (에포크 수)
    grace_period=1,                  # 최소 실행 리소스 (early stopping 전 최소 에포크)
    reduction_factor=3,              # η: 줄임 비율 (기본: 4)
    brackets=1,                      # 브라켓 수 (기본: 1)
    stop_last_trials=True,           # max_t 도달 시 trial 종료 여부
)
```

**파라미터 상세 설명:**

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `time_attr` | str | `'training_iteration'` | 리소스 축. 에포크 수, 훈련 시간 등 단조 증가하는 값 |
| `metric` | str \| None | None | 최적화할 메트릭 이름 (예: `'val_loss'`, `'accuracy'`) |
| `mode` | str \| None | None | `'min'`(손실 최소화) 또는 `'max'`(정확도 최대화) |
| `max_t` | int | 100 | trial당 최대 리소스. 이 값을 넘으면 trial 중단 |
| `grace_period` | int | 1 | 최소 리소스. 이 값 이전에는 early stop하지 않음 |
| `reduction_factor` | float | 4 | η에 해당. 매 rung에서 1/η만 생존 |
| `brackets` | int | 1 | HyperBand 브라켓 수. 1이면 순수 ASHA |
| `stop_last_trials` | bool | True | 최종 rung 이후 trial 종료 여부 |

**grace_period과 reduction_factor의 관계:**

```
grace_period=1, reduction_factor=3, max_t=81 인 경우:

Rung 레벨:  r_0=1  →  r_1=3  →  r_2=9  →  r_3=27  →  r_4=81
                ×3        ×3        ×3         ×3

각 rung에서 상위 1/3만 승격
```

```
grace_period=10, reduction_factor=2, max_t=160 인 경우:

Rung 레벨:  r_0=10  →  r_1=20  →  r_2=40  →  r_3=80  →  r_4=160
                 ×2         ×2         ×2         ×2

각 rung에서 상위 1/2만 승격
```

### 6-2. 코드 예시: XGBoost 하이퍼파라미터 튜닝

```python
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def train_xgboost(config):
    """Ray Tune trainable function for XGBoost."""
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "subsample": config["subsample"],
        "colsample_bytree": config["colsample_bytree"],
        "min_child_weight": config["min_child_weight"],
        "tree_method": "gpu_hist",  # GPU 사용
        "device": "cuda",
    }

    # Incremental training: 매 iteration마다 메트릭 보고
    model = None
    for epoch in range(config["n_estimators"]):
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1,     # 1 라운드씩 증분 훈련
            evals=[(dval, "val")],
            xgb_model=model,       # 이전 모델 이어서 훈련
            verbose_eval=False,
        )

        val_loss = float(model.eval(dval).split(":")[1])
        # Ray Tune에 메트릭 보고 → ASHA가 이걸 보고 early stop 결정
        tune.report(val_loss=val_loss, training_iteration=epoch + 1)


# ── ASHA Scheduler 설정 ──
scheduler = ASHAScheduler(
    time_attr="training_iteration",
    metric="val_loss",
    mode="min",
    max_t=500,              # 최대 500 boosting rounds
    grace_period=10,        # 최소 10 rounds는 실행
    reduction_factor=3,     # 매 rung에서 1/3만 생존
    brackets=1,             # 순수 ASHA (단일 브라켓)
)

# ── 탐색 공간 정의 ──
search_space = {
    "max_depth": tune.randint(3, 15),
    "learning_rate": tune.loguniform(1e-3, 0.3),
    "subsample": tune.uniform(0.5, 1.0),
    "colsample_bytree": tune.uniform(0.5, 1.0),
    "min_child_weight": tune.randint(1, 10),
    "n_estimators": 500,  # max_t와 동일
}

# ── 실행 ──
tuner = tune.Tuner(
    tune.with_resources(
        train_xgboost,
        resources={"gpu": 0.25}  # trial당 GPU 1/4 할당 → 최대 4개 동시 실행
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=50,           # 총 50개 설정 탐색
        max_concurrent_trials=4,  # 동시 최대 4개
    ),
    param_space=search_space,
)

results = tuner.fit()
best_result = results.get_best_result("val_loss", mode="min")
print(f"Best config: {best_result.config}")
print(f"Best val_loss: {best_result.metrics['val_loss']:.4f}")
```

### 6-3. GPU 리소스 관리 연동

Ray Tune의 리소스 관리와 ASHA를 결합하면 OOM을 체계적으로 방지할 수 있다.

```python
# ── GPU 리소스 할당 전략 ──

# 전략 1: 고정 분할 (단순, 안전)
# GPU 1개를 4등분 → 동시 4개 trial
tune.with_resources(trainable, resources={"gpu": 0.25})

# 전략 2: 모델 크기 기반 동적 분할
# 큰 모델은 GPU를 더 많이 할당
def get_resources(config):
    if config["max_depth"] > 10:
        return {"gpu": 0.5}   # 대형 모델: GPU 절반
    else:
        return {"gpu": 0.25}  # 소형 모델: GPU 1/4

tune.with_resources(trainable, resources=tune.PlacementGroupFactory([
    {"GPU": 0.25}  # 기본 할당
]))

# 전략 3: max_concurrent_trials로 동시 실행 제한
tune.TuneConfig(
    scheduler=scheduler,
    num_samples=100,
    max_concurrent_trials=4,  # 절대 4개 초과 동시 실행 안 함
)
```

**ASHA + GPU 리소스 관리의 시너지:**

```
시간 → ─────────────────────────────────────────────────────→

GPU 0.00-0.25: [Trial 1: 10ep ✗] [Trial 5: 10ep ✗] [Trial 9: 30ep → 승격]...
GPU 0.25-0.50: [Trial 2: 10ep → 승격 → 30ep → 승격 → 90ep ✓ Best!]
GPU 0.50-0.75: [Trial 3: 10ep ✗] [Trial 6: 10ep ✗] [Trial 7: 10ep ✗]...
GPU 0.75-1.00: [Trial 4: 10ep → 승격 → 30ep ✗] [Trial 8: 10ep ✗]...

✗ = ASHA에 의해 early stop (GPU 즉시 해제)
✓ = 최종 rung까지 도달한 우수 trial

→ 나쁜 trial이 빠르게 종료되면서 GPU 슬롯이 빈번하게 재활용됨
→ 같은 시간에 훨씬 더 많은 설정을 탐색 가능
```

### 6-4. Optuna + ASHA 결합

```python
from ray.tune.search.optuna import OptunaSearch

# Optuna의 TPE sampler로 설정을 생성하고,
# ASHA로 나쁜 trial을 조기 종료하는 조합
search_algo = OptunaSearch(
    metric="val_loss",
    mode="min",
)

scheduler = ASHAScheduler(
    metric="val_loss",
    mode="min",
    max_t=500,
    grace_period=10,
    reduction_factor=3,
)

tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        search_alg=search_algo,   # 어떤 설정을 시도할지 (Optuna TPE)
        scheduler=scheduler,      # 언제 중단할지 (ASHA)
        num_samples=100,
        max_concurrent_trials=4,
    ),
    param_space=search_space,
)
```

---

## 7. PagedAutoML Phase 구조에 적용

### 7-1. Phase B (Diversity)에 ASHA 적용

Phase B의 목적은 **다양한 특성의 모델을 확보**하는 것이다.
사전 정의된 하이퍼파라미터 그리드에서 모델을 훈련하되,
명백히 나쁜 조합은 조기에 제거한다.

**적용 설계:**

```python
# Phase B: Diversity Grid + ASHA
phase_b_scheduler = ASHAScheduler(
    metric="val_auc",
    mode="max",
    max_t=100,              # 최대 100 에포크
    grace_period=5,         # 최소 5 에포크는 관찰 (diversity 보장)
    reduction_factor=3,     # 상위 1/3만 생존
    brackets=1,
)

# Phase B 전용 그리드
phase_b_grid = {
    "xgboost": [
        {"max_depth": 3, "learning_rate": 0.1},
        {"max_depth": 6, "learning_rate": 0.1},
        {"max_depth": 9, "learning_rate": 0.05},
        {"max_depth": 12, "learning_rate": 0.01},
    ],
    "lightgbm": [
        {"num_leaves": 31, "learning_rate": 0.1},
        {"num_leaves": 63, "learning_rate": 0.05},
        {"num_leaves": 127, "learning_rate": 0.01},
    ],
    # ... 다른 알고리즘
}
```

**Phase B에서 ASHA 적용 시 동작 예시:**

```
XGBoost 그리드 (4개 설정):

Rung 0 (5 에포크):
  max_depth=3,  lr=0.1   → AUC=0.82  ✓ 생존
  max_depth=6,  lr=0.1   → AUC=0.85  ✓ 생존
  max_depth=9,  lr=0.05  → AUC=0.79  ✗ 탈락 (리소스 해제!)
  max_depth=12, lr=0.01  → AUC=0.71  ✗ 탈락 (리소스 해제!)

Rung 1 (15 에포크):
  max_depth=3,  lr=0.1   → AUC=0.88  ✓ 최종 생존
  max_depth=6,  lr=0.1   → AUC=0.91  ✓ 최종 생존

→ 4개 중 2개만 전체 훈련 → VRAM 사용 50% 절감
→ 나머지 2개의 GPU 시간을 다른 알고리즘에 할당 가능
```

**Phase B에서 grace_period를 높게 설정하는 이유:**

Phase B의 목적은 성능 극대화가 아닌 diversity이다.
따라서 너무 이른 조기 종료는 피해야 한다.
`grace_period=5` 이상으로 설정하여 최소한의 훈련 후 판단한다.

### 7-2. Phase C (Random Search)에 ASHA 적용

Phase C는 **시간 예산 내에서 최대한 많은 설정을 탐색**하는 것이 목적이다.
여기서 ASHA의 효과가 극대화된다.

**적용 설계:**

```python
# Phase C: Random Search + ASHA (aggressive)
phase_c_scheduler = ASHAScheduler(
    metric="val_auc",
    mode="max",
    max_t=200,              # 최대 200 에포크
    grace_period=1,         # 1 에포크만에도 판단 (aggressive)
    reduction_factor=4,     # 상위 1/4만 생존 (더 aggressive)
    brackets=1,
)

# Phase C 전용 넓은 탐색 공간
phase_c_space = {
    "max_depth": tune.randint(2, 20),
    "learning_rate": tune.loguniform(1e-4, 0.5),
    "subsample": tune.uniform(0.3, 1.0),
    "colsample_bytree": tune.uniform(0.3, 1.0),
    "min_child_weight": tune.randint(1, 20),
    "reg_alpha": tune.loguniform(1e-8, 10.0),
    "reg_lambda": tune.loguniform(1e-8, 10.0),
}

tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        scheduler=phase_c_scheduler,
        num_samples=200,           # 많은 설정 탐색
        max_concurrent_trials=4,
        time_budget_s=remaining_time,  # 남은 시간 예산
    ),
    param_space=phase_c_space,
)
```

**Phase C에서 ASHA가 메모리를 절약하는 시나리오:**

```
200개 설정 중 ASHA에 의한 단계별 생존:

Rung 0 (1 에포크):  200개 시작 → 50개 생존 (150개 즉시 GPU 해제)
Rung 1 (4 에포크):   50개      → 12개 생존
Rung 2 (16 에포크):  12개      →  3개 생존
Rung 3 (64 에포크):   3개      →  1개 생존
Rung 4 (200 에포크):  1개      →  최종 최적 모델

Random Search로 200개 전부 200 에포크 실행 시:
  총 리소스 = 200 × 200 = 40,000 에포크

ASHA로 실행 시:
  총 리소스 ≈ 200 + 200 + 192 + 192 + 200 = 984 에포크

→ 약 40배의 리소스 절감!
→ 같은 시간에 훨씬 더 많은 설정을 탐색 가능
```

### 7-3. 전체 Phase 구조 통합

```python
class PagedAutoMLWithASHA:
    """ASHA를 결합한 PagedAutoML 학습 파이프라인."""

    def __init__(self, total_time_budget: int, vram_gb: float = 24.0):
        self.total_time = total_time_budget
        self.vram_gb = vram_gb
        # GPU 메모리 기반 동시 실행 수 결정
        self.max_concurrent = max(1, int(vram_gb / 4))  # 모델당 ~4GB 가정

    def run(self):
        # Phase A: Baseline (ASHA 불필요)
        # 알고리즘별 기본 모델 1개씩, 전체 리소스로 훈련
        phase_a_time = self.total_time * 0.20
        baseline_models = self._run_phase_a(phase_a_time)

        # Phase B: Diversity + ASHA (보수적)
        phase_b_time = self.total_time * 0.30
        diversity_models = self._run_phase_b(phase_b_time)

        # Phase C: Random Search + ASHA (공격적)
        phase_c_time = self.total_time * 0.40
        random_models = self._run_phase_c(phase_c_time)

        # Phase D: Stacked Ensemble
        phase_d_time = self.total_time * 0.10
        all_models = baseline_models + diversity_models + random_models
        ensemble = self._build_ensemble(all_models, phase_d_time)

        return ensemble

    def _run_phase_b(self, time_budget):
        scheduler = ASHAScheduler(
            metric="val_auc", mode="max",
            max_t=100,
            grace_period=5,        # diversity를 위해 보수적
            reduction_factor=3,
        )
        # ... (tune.Tuner로 실행)

    def _run_phase_c(self, time_budget):
        scheduler = ASHAScheduler(
            metric="val_auc", mode="max",
            max_t=200,
            grace_period=1,        # 탐색량 극대화를 위해 공격적
            reduction_factor=4,
        )
        # ... (tune.Tuner로 실행)
```

### 7-4. 예상 효과

```
┌──────────────────────────┬──────────────────┬───────────────────┐
│ 지표                     │ ASHA 적용 전     │ ASHA 적용 후      │
├──────────────────────────┼──────────────────┼───────────────────┤
│ Phase B OOM 발생률       │ ~15%             │ ~2%               │
│ Phase C OOM 발생률       │ ~30%             │ ~3%               │
│ Phase B 탐색 설정 수     │ 20 ~ 30개        │ 50 ~ 80개         │
│ Phase C 탐색 설정 수     │ 30 ~ 50개        │ 100 ~ 200개       │
│ GPU 활용률               │ 60 ~ 70%         │ 85 ~ 95%          │
│ 동일 시간 대비 최종 성능 │ baseline         │ 2 ~ 4배 개선       │
│ VRAM 피크 사용량         │ 20 ~ 24GB (위험) │ 12 ~ 16GB (안전)  │
└──────────────────────────┴──────────────────┴───────────────────┘

* 수치는 VRAM 24GB, 데이터셋 1GB 기준 예상치이며 실측 벤치마크가 필요하다.
```

**핵심 개선 요약:**

1. **OOM 방지**: 나쁜 trial을 조기 종료하여 동시 활성 모델 수를 제한
2. **탐색 효율**: 같은 시간에 2 ~ 4배 더 많은 설정 탐색 가능
3. **GPU 활용률**: ASHA의 비동기 특성으로 Worker 유휴 시간 최소화
4. **메모리 안전**: 각 rung에서 활성 모델 수가 기하급수적으로 감소하여 VRAM 피크 제어

---

## 8. References

1. **Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A.** (2018).
   Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.
   *Journal of Machine Learning Research*, 18(185), 1-52.
   https://jmlr.org/papers/v18/16-558.html

2. **Li, L., Jamieson, K., Rostamizadeh, A., Gonina, E., Ben-tzur, J., Hardt, M., Recht, B., & Talwalkar, A.** (2020).
   A System for Massively Parallel Hyperparameter Tuning.
   *MLSys 2020*.
   https://arxiv.org/abs/1810.05934

3. **Bergstra, J. & Bengio, Y.** (2012).
   Random Search for Hyper-Parameter Optimization.
   *Journal of Machine Learning Research*, 13, 281-305.

4. **Jamieson, K. & Talwalkar, A.** (2016).
   Non-stochastic Best Arm Identification and Hyperparameter Optimization.
   *AISTATS 2016*.

5. **Ray Tune Documentation** — ASHAScheduler.
   https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.ASHAScheduler.html

6. **Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J.** (2023).
   Dive into Deep Learning — 19.5 Asynchronous Successive Halving.
   https://d2l.ai/chapter_hyperparameter-optimization/sh-async.html
