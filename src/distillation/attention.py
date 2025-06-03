import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDistillation

class AttentionDistillation(BaseDistillation):
    """
    Attention-based knowledge distillation.
    Transfers attention maps from teacher to student.
    """
    
    def __init__(self, teacher_model, student_model, temperature=2.0, alpha=0.5, gamma=0.2):
        super(AttentionDistillation, self).__init__(teacher_model, student_model, temperature, alpha)
        self.gamma = gamma  # Weight for attention transfer loss
    
    def compute_attention_map(self, features):
        """
        Compute attention map from feature representations.
        For NCF, we'll use the embedding features to compute attention.
        """
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        
        # Compute attention as the L2 norm of features
        attention = torch.norm(features_norm, p=2, dim=-1, keepdim=True)
        attention = F.softmax(attention, dim=0)
        
        return attention
    
    def extract_attention_features(self, model, user, item):
        """Extract features for attention computation."""
        attention_features = {}
        
        if hasattr(model, 'embed_user_GMF'):
            user_embed = model.embed_user_GMF(user)
            item_embed = model.embed_item_GMF(item)
            gmf_features = user_embed * item_embed
            attention_features['gmf_attention'] = self.compute_attention_map(gmf_features)
        
        if hasattr(model, 'embed_user_MLP'):
            user_embed_mlp = model.embed_user_MLP(user)
            item_embed_mlp = model.embed_item_MLP(item)
            concat = torch.cat((user_embed_mlp, item_embed_mlp), -1)
            attention_features['mlp_attention'] = self.compute_attention_map(concat)
        
        return attention_features
    
    def attention_transfer_loss(self, teacher_attention, student_attention):
        """Compute attention transfer loss."""
        total_loss = 0
        count = 0
        
        for key in teacher_attention:
            if key in student_attention:
                teacher_att = teacher_attention[key]
                student_att = student_attention[key]
                
                # Compute attention transfer loss using KL divergence
                teacher_att_flat = teacher_att.view(-1)
                student_att_flat = student_att.view(-1)
                
                # Add small epsilon to avoid log(0)
                eps = 1e-8
                teacher_att_flat = teacher_att_flat + eps
                student_att_flat = student_att_flat + eps
                
                # Normalize to make them probability distributions
                teacher_att_flat = teacher_att_flat / teacher_att_flat.sum()
                student_att_flat = student_att_flat / student_att_flat.sum()
                
                att_loss = F.kl_div(
                    torch.log(student_att_flat), 
                    teacher_att_flat, 
                    reduction='batchmean'
                )
                total_loss += att_loss
                count += 1
        
        return total_loss / max(count, 1)
    
    def forward(self, user, item, label):
        """Forward pass for attention-based distillation."""
        # Extract teacher attention (no gradient)
        with torch.no_grad():
            teacher_attention = self.extract_attention_features(self.teacher_model, user, item)
            teacher_logits = self.teacher_model(user, item)
        
        # Extract student attention
        student_attention = self.extract_attention_features(self.student_model, user, item)
        student_logits = self.student_model(user, item)
        
        # Compute losses
        task_loss = self.task_loss(student_logits, label)
        response_loss = self.knowledge_distillation_loss(teacher_logits, student_logits)
        attention_loss = self.attention_transfer_loss(teacher_attention, student_attention)
        
        # Combined loss
        total_loss = (self.alpha * task_loss + 
                     (1 - self.alpha - self.gamma) * response_loss + 
                     self.gamma * attention_loss)
        
        return total_loss