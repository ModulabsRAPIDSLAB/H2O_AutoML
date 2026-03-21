# Troubleshooting

## H2O `explain()` + NumPy 2.x 호환성 오류

### 증상

`h2o.explain()` 또는 SHAP Summary 관련 explanation 호출 시 아래와 유사한 에러 발생:

```
Incorrect number of arguments; 'cols_py' expects 2 but was passed 3
```

Rapids AST 내부에 컬럼명이 `np.str_('merchant')` 형태로 찍히는 현상도 함께 나타남.

### 환경

| 패키지 | 버전 |
|--------|------|
| h2o | 3.46.0.10 |
| numpy | 2.0.2 |
| Python | 3.9 |

### 원인

H2O explain 모듈(`h2o/explanation/_explain.py`)의 `NumpyFrame.__init__`에서 factor 컬럼을 추출할 때 다음 패턴을 사용한다:

```python
# h2o/explanation/_explain.py:236-237
self._factors = {col: h2o_frame[col].asfactor().levels()[0] for col in
                 np.array(h2o_frame.columns)[_is_factor]}
```

`np.array(h2o_frame.columns)[_is_factor]`의 각 원소는 **`numpy.str_`** 타입이다.

- **NumPy 1.x**: `numpy.str_`가 Python `str`의 서브클래스로 동작하여 H2O Rapids 직렬화에 문제가 없었음.
- **NumPy 2.x**: `numpy.str_` 동작이 변경되어, H2O Rapids AST/직렬화 과정에서 컬럼 참조를 올바르게 파싱하지 못함. 결과적으로 `cols_py` 인자 개수 불일치 에러로 이어짐.

### 해결 방법

#### 방법 1: 해당 explanation 우회 (라이브러리 수정 없음)

```python
model.explain(frame, exclude_explanations=["shap_summary"])
```

SHAP Summary 경로를 건너뛰어 오류를 회피한다.

#### 방법 2: NumPy 다운그레이드

```bash
uv pip install "numpy<2"
# 또는
pip install "numpy<2"
```

`numpy 1.26.x`로 내려가면 `numpy.str_`가 기존처럼 동작하므로 에러가 해소된다.

#### 방법 3: H2O 패키지 내부 패치 (권장하지 않음)

`.venv/lib/python{VERSION}/site-packages/h2o/explanation/_explain.py` 236-237행을 수정 (경로의 Python 버전은 환경에 맞게 조정):

```python
self._factors = {str(col): h2o_frame[str(col)].asfactor().levels()[0] for col in
                 np.array(h2o_frame.columns)[_is_factor]}
```

`str(col)`로 감싸면 `np.str_` → Python `str` 변환이 되어 정상 동작한다.
단, venv 패키지 직접 수정이므로 환경 재구성 시 매번 적용해야 한다.

### 검증 절차

1. 수정 전 현재 numpy 버전 확인: `python -c "import numpy; print(numpy.__version__)"`
2. 위 해결 방법 중 하나 적용
3. 노트북 커널 재시작 후 동일 셀 재실행
4. `explain()` 호출이 에러 없이 완료되는지 확인
