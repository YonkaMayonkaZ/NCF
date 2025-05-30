import os
# dataset name 
dataset = 'u'

# Model settings (Ensure this line exists)
model = 'NeuMF-end'  # Options: ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths - use the mounted dataset
train_rating = "/app/u.train.rating"
test_rating = "/app/u.test.rating"
test_negative = "/app/u.test.negative"

model_path = "/app/models/"

# Ensure the directory exists
if not os.path.exists(model_path):
    os.makedirs(model_path)
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'

gpu = "-1"
