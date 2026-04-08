# Future Directions — 개선 방향

## 단기 (현재 프레임워크 내에서 가능)

1. **CPU H2O 동일 조건 벤치마크** <br/>
   같은 데이터, 같은 `max_runtime_secs`로 H2O CPU를 돌려서 속도를 직접 비교

2. **PagedMemoryManager 실사용 검증** <br/>
   `paged_memory=True`로 Higgs 데이터를 돌려서 블록 할당/회수가 실제로 skip을 줄이는지 확인

3. **GLM class_weight 적용** <br/>
   cuML GLM에 클래스 가중치를 부여하여 불균형 데이터 대응

4. **rmm pool 전략 비교** <br/>
   None / Fixed / Managed / Adaptive 4가지를 같은 조건에서 비교

5. **16GB+ GPU 벤치마크** <br/>
   Higgs 전체(11M rows)를 돌려서 Memory-Aware의 "부분 skip" 시나리오 재현

## 중기 (아키텍처 변경 필요)

6. **rmm Custom Allocator** <br/>
   rmm의 `device_memory_resource`를 상속하여 블록 단위 할당을 cuML/XGBoost에 투명하게 적용.
   이렇게 하면 라이브러리 내부 할당도 블록 관리 가능

7. **Dask worker 단위 메모리 관리** <br/>
   `device_memory_limit` + Dask resource annotation으로 worker별 VRAM 예산 관리

8. **모델 체크포인트 swap** <br/>
   훈련 중간 결과를 Host로 swap하고, VRAM이 확보되면 resume.
   cuML은 미지원이지만 XGBoost의 callback으로 부분 구현 가능

## 장기 (연구 수준)

9. **cuML 내부 메모리 API** <br/>
   cuML이 메모리 접근 패턴을 외부에 노출하면, vLLM처럼 연산 중 Page Table 참조가 가능해진다.
   이것이 진정한 "fine-grained paging"의 전제 조건

10. **GPU AutoML 전용 커널** <br/>
    Stacking의 OOF 수집, Meta Learner 훈련 등 AutoML 특화 연산을 CUDA 커널로 작성하면,
    해당 구간에서는 vLLM 수준의 메모리 관리가 가능
