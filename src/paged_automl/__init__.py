"""PagedAutoML — vLLM-inspired Paged Memory Management for GPU AutoML.

H2O AutoML의 검증된 Stacking/HPO 전략을 RAPIDS 생태계로 재구현하고,
vLLM의 PagedAttention에서 영감받은 Block 기반 GPU 메모리 관리를 내재화한 프레임워크.
"""

from paged_automl.automl import GPUAutoML

__all__ = ["GPUAutoML"]
__version__ = "0.1.0"
