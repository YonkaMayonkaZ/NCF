"""Unified knowledge distillation module combining multiple strategies."""

import torch
import torch.nn as nn
from .base import BaseDistillation
from .feature import FeatureDistillation
from .attention import AttentionDistillation


class UnifiedDistillation(BaseDistillation):
    """Combine response, feature and attention based distillation."""

    def __init__(self, teacher_model, student_model, temperature=2.0,
                 alpha=0.5, beta=0.3, gamma=0.2):
        super(UnifiedDistillation, self).__init__(
            teacher_model, student_model, temperature, alpha
        )
        self.beta = beta
        self.gamma = gamma

        # Helper modules for feature and attention distillation
        self._feature_helper = FeatureDistillation(
            teacher_model, student_model, temperature, alpha, beta
        )
        self._attention_helper = AttentionDistillation(
            teacher_model, student_model, temperature, alpha, gamma
        )

    def forward(self, user, item, label):
        """Forward pass computing the combined distillation loss."""
        with torch.no_grad():
            teacher_logits = self.teacher_model(user, item)
            teacher_features = self._feature_helper.extract_features(
                self.teacher_model, user, item
            )
            teacher_attention = self._attention_helper.extract_attention_features(
                self.teacher_model, user, item
            )

        student_logits = self.student_model(user, item)
        student_features = self._feature_helper.extract_features(
            self.student_model, user, item
        )
        student_attention = self._attention_helper.extract_attention_features(
            self.student_model, user, item
        )

        # Losses
        task_loss = self.task_loss(student_logits, label)
        response_loss = self.knowledge_distillation_loss(
            teacher_logits, student_logits
        )
        feature_loss = self._feature_helper.feature_matching_loss(
            teacher_features, student_features
        )
        attention_loss = self._attention_helper.attention_transfer_loss(
            teacher_attention, student_attention
        )

        total_loss = (
            self.alpha * task_loss
            + (1 - self.alpha - self.beta - self.gamma) * response_loss
            + self.beta * feature_loss
            + self.gamma * attention_loss
        )

        return total_loss

