import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model_type):
        super(NCF, self).__init__()
        self.model_type = model_type
        self.dropout = dropout

        # GMF embeddings
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)

        # MLP embeddings
        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))

        # MLP layers
        MLP_modules = []
        input_size = factor_num * (2 ** num_layers)
        for i in range(num_layers):
            output_size = input_size // 2
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, output_size))
            MLP_modules.append(nn.ReLU())
            input_size = output_size
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # Final prediction layer
        if self.model_type in ["GMF", "MLP"]:
            predict_size = factor_num
        else:  # NeuMF-end, NeuMF-pre
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def load_pretrain_weights(self, gmf_state, mlp_state):
        """Load pretrained GMF and MLP weights for NeuMF-pre."""
        if self.model_type != "NeuMF-pre":
            return
        
        try:
            # Load GMF weights
            self.embed_user_GMF.weight.data.copy_(gmf_state['embed_user_GMF.weight'])
            self.embed_item_GMF.weight.data.copy_(gmf_state['embed_item_GMF.weight'])
            print("    GMF weights loaded successfully")
            
            # Load MLP embedding weights
            self.embed_user_MLP.weight.data.copy_(mlp_state['embed_user_MLP.weight'])
            self.embed_item_MLP.weight.data.copy_(mlp_state['embed_item_MLP.weight'])
            print("    MLP embedding weights loaded successfully")
            
            # Load MLP layers weights - handle the state dict structure properly
            linear_layer_count = 0
            for i, layer in enumerate(self.MLP_layers):
                if isinstance(layer, nn.Linear):
                    # Look for corresponding weight and bias in the loaded state
                    weight_key = f'MLP_layers.{i}.weight'
                    bias_key = f'MLP_layers.{i}.bias'
                    
                    if weight_key in mlp_state and bias_key in mlp_state:
                        layer.weight.data.copy_(mlp_state[weight_key])
                        layer.bias.data.copy_(mlp_state[bias_key])
                        print(f"    Loaded MLP layer {linear_layer_count} weights")
                        linear_layer_count += 1
                    else:
                        print(f"    Warning: Could not find weights for MLP layer {linear_layer_count}")
                        # Initialize this layer randomly if weights not found
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        linear_layer_count += 1
            
            # Initialize predict layer randomly (as in original paper)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
            if self.predict_layer.bias is not None:
                nn.init.zeros_(self.predict_layer.bias)
            print("    NeuMF prediction layer initialized")
            
        except Exception as e:
            print(f"    Error loading pretrained weights: {e}")
            print("    Available keys in GMF state:", list(gmf_state.keys())[:5])  # Show first 5 keys
            print("    Available keys in MLP state:", list(mlp_state.keys())[:5])  # Show first 5 keys
            raise RuntimeError(f"Failed to load pretrained weights: {e}")

    def forward(self, user, item):
        if self.model_type == "GMF":
            user_embed = self.embed_user_GMF(user)
            item_embed = self.embed_item_GMF(item)
            output = user_embed * item_embed
        elif self.model_type == "MLP":
            user_embed = self.embed_user_MLP(user)
            item_embed = self.embed_item_MLP(item)
            concat = torch.cat((user_embed, item_embed), -1)
            output = self.MLP_layers(concat)
        else:  # NeuMF-end, NeuMF-pre
            user_gmf = self.embed_user_GMF(user)
            item_gmf = self.embed_item_GMF(item)
            gmf_output = user_gmf * item_gmf
            user_mlp = self.embed_user_MLP(user)
            item_mlp = self.embed_item_MLP(item)
            mlp_input = torch.cat((user_mlp, item_mlp), -1)
            mlp_output = self.MLP_layers(mlp_input)
            output = torch.cat((gmf_output, mlp_output), -1)
        
        prediction = self.predict_layer(output)
        return prediction.view(-1)