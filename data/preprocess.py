import numpy as np
import pandas as pd
import scipy.sparse as sp

# Input file
input_file = "u.data"

# Output files
train_output_file = "u.train.rating"
test_rating_output_file = "u.test.rating"  # Only positive samples
test_negative_output_file = "u.test.negative"  # 99 negatives per user

# Load dataset
columns = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv(input_file, sep='\t', names=columns)

# Convert explicit ratings into implicit feedback
df["interaction"] = 1  # Any rating > 0 is considered an interaction
df = df[["user_id", "item_id"]]  # Drop rating and timestamp

# Get user and item counts
num_users = df["user_id"].max() + 1  # Ensure zero-indexing
num_items = df["item_id"].max() + 1  # Ensure zero-indexing

print(f"Number of Users: {num_users}, Number of Items: {num_items}")

# Save train dataset
df.to_csv(train_output_file, sep='\t', index=False, header=False)

# Create user-item interaction matrix
interaction_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
for _, row in df.iterrows():
    interaction_matrix[row["user_id"], row["item_id"]] = 1.0

# Generate test.rating and test.negative
test_rating = []
test_negative = []
num_neg_samples = 99  # Fixed for evaluation

for user in range(num_users):
    user_items = set(df[df["user_id"] == user]["item_id"])
    if not user_items:
        continue  # Skip users with no interactions

    positive_item = np.random.choice(list(user_items))  # Select one positive item
    test_rating.append(f"{user}\t{positive_item}")

    # Generate 99 negative samples (ensuring they are not in training set)
    negative_samples = []
    while len(negative_samples) < num_neg_samples:
        neg_item = np.random.randint(num_items)
        if neg_item not in user_items:
            negative_samples.append(neg_item)

    # Format test.negative entry
    test_negative.append(f"({user},{positive_item})\t" + "\t".join(map(str, negative_samples)))

# Save test.rating
with open(test_rating_output_file, "w") as f:
    for line in test_rating:
        f.write(line + "\n")

# Save test.negative
with open(test_negative_output_file, "w") as f:
    for line in test_negative:
        f.write(line + "\n")

print(f"Preprocessing complete. Files saved:\n- {train_output_file}\n- {test_rating_output_file}\n- {test_negative_output_file}")


