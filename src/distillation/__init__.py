"""Knowledge Distillation modules for NCF."""

from .base import BaseDistillation
from .response import ResponseDistillation, SoftTargetDistillation
from .feature import FeatureDistillation
from .attention import AttentionDistillation

__all__ = [
    'BaseDistillation',
    'ResponseDistillation', 
    'SoftTargetDistillation',
    'FeatureDistillation',
    'AttentionDistillation'
]