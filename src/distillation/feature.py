import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDistillation

class FeatureDistillation(BaseDistillation):
    """
    Feature-based knowledge distillation.
    Transfers intermediate feature representations from teacher to student.
    """
    
    def __init__(self, teacher_model, student_model, temperature=2.0, alpha=0.5, beta=0.3):
        super(FeatureDistillation, self).__init__(teacher_model, student_model, temperature, alpha)
        self.beta = beta  # Weight for feature matching loss
        
        # Feature adaptation layers to match dimensions
        self.adaptation_layers = nn.ModuleDict()
        self._setup_adaptation_layers()
    
    def _setup_adaptation_layers(self):
        """Setup adaptation layers to match teacher and student feature dimensions."""
        # This is a simplified version - in practice, you'd need to carefully
        # match the dimensions of intermediate features
        
        # For NCF models, we'll match the embedding dimensions
        teacher_embed_dim = self.teacher_model.embed_user_GMF.embedding_dim
        student_embed_dim = self.student_model.embed_user_GMF.embedding_dim
        
        if teacher_embed_dim != student_embed_dim:
            self.adaptation_layers['embed_adapter'] = nn.Linear(student_embed_dim, teacher_embed_dim)
    
    def extract_features(self, model, user, item):
        """Extract intermediate features from the model."""
        features = {}
        
        # Get embedding features
        if hasattr(model, 'embed_user_GMF'):
            user_embed = model.embed_user_GMF(user)
            item_embed = model.embed_item_GMF(item)
            features['gmf_features'] = user_embed * item_embed
        
        if hasattr(model, 'embed_user_MLP'):
            user_embed_mlp = model.embed_user_MLP(user)
            item_embed_mlp = model.embed_item_MLP(item)
            concat = torch.cat((user_embed_mlp, item_embed_mlp), -1)
            
            # Extract MLP intermediate features
            if hasattr(model, 'MLP_layers'):
                x = concat
                features['mlp_input'] = x
                for i, layer in enumerate(model.MLP_layers):
                    x = layer(x)
                    if isinstance(layer, nn.ReLU):
                        features[f'mlp_layer_{i}'] = x
        
        return features
    
    def feature_matching_loss(self, teacher_features, student_features):
        """Compute feature matching loss between teacher and student."""
        total_loss = 0
        count = 0
        
        for key in teacher_features:
            if key in student_features:
                teacher_feat = teacher_features[key]
                student_feat = student_features[key]
                
                # Adapt student features if needed
                if key in self.adaptation_layers:
                    student_feat = self.adaptation_layers[key](student_feat)
                
                # Compute MSE loss between features
                feat_loss = F.mse_loss(student_feat, teacher_feat)
                total_loss += feat_loss
                count += 1
        
        return total_loss / max(count, 1)
    
    def forward(self, user, item, label):
        """Forward pass for feature-based distillation."""
        # Extract teacher features (no gradient)
        with torch.no_grad():
            teacher_features = self.extract_features(self.teacher_model, user, item)
            teacher_logits = self.teacher_model(user, item)
        
        # Extract student features
        student_features = self.extract_features(self.student_model, user, item)
        student_logits = self.student_model(user, item)
        
        # Compute losses
        task_loss = self.task_loss(student_logits, label)
        response_loss = self.knowledge_distillation_loss(teacher_logits, student_logits)
        feature_loss = self.feature_matching_loss(teacher_features, student_features)
        
        # Combined loss
        total_loss = (self.alpha * task_loss + 
                     (1 - self.alpha - self.beta) * response_loss + 
                     self.beta * feature_loss)
        
        return total_loss