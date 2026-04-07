"""Memory-Aware GPU AutoML Framework.

H2O AutoML의 검증된 Stacking/HPO 전략을 RAPIDS 생태계로 재구현하고,
GPU 메모리 최적화를 핵심 기능으로 내재화한 프레임워크.
"""

from gpu_automl.automl import GPUAutoML

__all__ = ["GPUAutoML"]
__version__ = "0.1.0"
