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
        # Get teacher and student dimensions
        teacher_gmf_dim = self.teacher_model.embed_user_GMF.embedding_dim
        student_gmf_dim = self.student_model.embed_user_GMF.embedding_dim
        
        teacher_mlp_user_dim = self.teacher_model.embed_user_MLP.embedding_dim
        student_mlp_user_dim = self.student_model.embed_user_MLP.embedding_dim
        
        teacher_mlp_item_dim = self.teacher_model.embed_item_MLP.embedding_dim
        student_mlp_item_dim = self.student_model.embed_item_MLP.embedding_dim
        
        print(f"Teacher dims: GMF={teacher_gmf_dim}, MLP_user={teacher_mlp_user_dim}")
        print(f"Student dims: GMF={student_gmf_dim}, MLP_user={student_mlp_user_dim}")
        
        # Adaptation layers for different feature types
        if teacher_gmf_dim != student_gmf_dim:
            self.adaptation_layers['gmf_features'] = nn.Linear(student_gmf_dim, teacher_gmf_dim)
            print(f"Created GMF adapter: {student_gmf_dim} -> {teacher_gmf_dim}")
        
        # MLP input concatenation dimension
        teacher_mlp_concat_dim = teacher_mlp_user_dim + teacher_mlp_item_dim
        student_mlp_concat_dim = student_mlp_user_dim + student_mlp_item_dim
        
        if teacher_mlp_concat_dim != student_mlp_concat_dim:
            self.adaptation_layers['mlp_input'] = nn.Linear(student_mlp_concat_dim, teacher_mlp_concat_dim)
            print(f"Created MLP input adapter: {student_mlp_concat_dim} -> {teacher_mlp_concat_dim}")
        
        # For MLP intermediate layers, we'll need to check dimensions dynamically
        # This is more complex since layer dimensions depend on the architecture
    
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
            features['mlp_input'] = concat
            
            # Extract MLP intermediate features (with safer indexing)
            if hasattr(model, 'MLP_layers'):
                x = concat
                layer_idx = 0
                for i, layer in enumerate(model.MLP_layers):
                    if isinstance(layer, nn.Linear):
                        x = layer(x)
                        features[f'mlp_linear_{layer_idx}'] = x
                        layer_idx += 1
                    elif isinstance(layer, nn.ReLU):
                        x = layer(x)
                        # Store ReLU outputs with previous linear layer index
                        features[f'mlp_relu_{layer_idx-1}'] = x
        
        return features
    
    def feature_matching_loss(self, teacher_features, student_features):
        """Compute feature matching loss between teacher and student."""
        total_loss = 0
        count = 0
        
        for key in teacher_features:
            if key in student_features:
                teacher_feat = teacher_features[key]
                student_feat = student_features[key]
                
                # Debug print
                # print(f"Matching {key}: Teacher {teacher_feat.shape} vs Student {student_feat.shape}")
                
                # Check if we need adaptation
                if teacher_feat.shape != student_feat.shape:
                    if key in self.adaptation_layers:
                        # Use adaptation layer
                        student_feat = self.adaptation_layers[key](student_feat)
                        # print(f"  After adaptation: {student_feat.shape}")
                    else:
                        # Skip features that can't be matched
                        print(f"Warning: Skipping {key} - no adapter available "
                              f"(teacher: {teacher_feat.shape}, student: {student_feat.shape})")
                        continue
                
                # Compute MSE loss between features
                try:
                    feat_loss = F.mse_loss(student_feat, teacher_feat)
                    total_loss += feat_loss
                    count += 1
                except RuntimeError as e:
                    print(f"Error computing loss for {key}: {e}")
                    print(f"  Teacher shape: {teacher_feat.shape}")
                    print(f"  Student shape: {student_feat.shape}")
                    continue
        
        if count == 0:
            print("Warning: No features could be matched!")
            return torch.tensor(0.0, device=teacher_feat.device)
        
        return total_loss / count
    
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
        
        # Combined loss with safe weighting
        remaining_weight = max(0, 1 - self.alpha - self.beta)
        total_loss = (self.alpha * task_loss + 
                     remaining_weight * response_loss + 
                     self.beta * feature_loss)
        
        return total_loss