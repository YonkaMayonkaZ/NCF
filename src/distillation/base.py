import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseDistillation(nn.Module):
    """Base class for knowledge distillation techniques."""
    
    def __init__(self, teacher_model, student_model, temperature=2.0, alpha=0.5):
        super(BaseDistillation, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
    def forward(self, user, item, label):
        """
        Forward pass for distillation.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def knowledge_distillation_loss(self, teacher_logits, student_logits):
        """Compute KL divergence loss between teacher and student."""
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kd_loss * (self.temperature ** 2)
    
    def task_loss(self, predictions, labels):
        """Compute original task loss (BCE)."""
        return F.binary_cross_entropy_with_logits(predictions, labels)
    
    def combined_loss(self, teacher_logits, student_logits, labels):
        """Combine distillation loss and task loss."""
        # Task loss (original objective)
        task_loss = self.task_loss(student_logits, labels)
        
        # Knowledge distillation loss
        kd_loss = self.knowledge_distillation_loss(teacher_logits, student_logits)
        
        # Combined loss
        total_loss = self.alpha * task_loss + (1 - self.alpha) * kd_loss
        return total_loss