import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDistillation

class ResponseDistillation(BaseDistillation):
    """
    Response-based knowledge distillation.
    Transfers knowledge from teacher's final outputs to student.
    """
    
    def __init__(self, teacher_model, student_model, temperature=2.0, alpha=0.5):
        super(ResponseDistillation, self).__init__(teacher_model, student_model, temperature, alpha)
    
    def forward(self, user, item, label):
        """Forward pass for response-based distillation."""
        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_logits = self.teacher_model(user, item)
        
        # Get student predictions
        student_logits = self.student_model(user, item)
        
        # Compute combined loss
        loss = self.combined_loss(teacher_logits, student_logits, label)
        return loss
    
    def knowledge_distillation_loss(self, teacher_logits, student_logits):
        """MSE loss for response distillation in recommendation systems."""
        # For recommendation, we use MSE instead of KL divergence
        # since we're dealing with continuous rating predictions
        return F.mse_loss(student_logits, teacher_logits)

class SoftTargetDistillation(BaseDistillation):
    """
    Soft target distillation using temperature scaling.
    """
    
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        super(SoftTargetDistillation, self).__init__(teacher_model, student_model, temperature, alpha)
    
    def forward(self, user, item, label):
        """Forward pass for soft target distillation."""
        with torch.no_grad():
            teacher_logits = self.teacher_model(user, item)
        
        student_logits = self.student_model(user, item)
        
        # Apply temperature scaling
        teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
        student_soft = torch.sigmoid(student_logits / self.temperature)
        
        # Soft target loss
        soft_loss = F.mse_loss(student_soft, teacher_soft)
        
        # Hard target loss
        hard_loss = self.task_loss(student_logits, label)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss * (self.temperature ** 2)
        return total_loss