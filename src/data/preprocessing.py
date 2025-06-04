import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from datetime import datetime

from src.utils.io import save_results
from src.utils.logging import get_experiment_logger

class LeaveOneOutPreprocessor:
    def __init__(self, raw_path="data/raw/u.data", processed_dir="data/processed", num_negatives=99):
        self.raw_path = Path(raw_path)
        self.processed_dir = Path(processed_dir)
        self.num_negatives = num_negatives
        self.logger = get_experiment_logger("preprocessing")
        self.results = {}

        self.train_file = self.processed_dir / "u.train.rating"
        self.test_rating_file = self.processed_dir / "u.test.rating"
        self.test_negative_file = self.processed_dir / "u.test.negative"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self):
        """Load raw data and prepare for temporal split."""
        self.logger.info("Loading and preparing raw data...")
        
        df = pd.read_csv(
            self.raw_path,
            sep='\t',
            names=["user_id", "item_id", "rating", "timestamp"],
            dtype={"user_id": int, "item_id": int, "rating": int, "timestamp": int}
        )
        
        self.logger.info(f"Loaded {len(df):,} interactions")
        self.logger.info(f"Users: {df['user_id'].nunique():,}, Items: {df['item_id'].nunique():,}")
        
        # Sort by user and timestamp to get temporal order
        df_sorted = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Keep only user_id, item_id (binarize - if rated, then interaction = 1)
        interactions_df = df_sorted[["user_id", "item_id", "timestamp"]].copy()
        
        return interactions_df

    def temporal_split(self, df):
        """
        Perform temporal leave-one-out split:
        - For each user, take their LAST interaction as test
        - Use all previous interactions as training
        """
        self.logger.info("Performing temporal leave-one-out split...")
        
        train_data = []
        test_data = []
        users_processed = 0
        users_skipped = 0
        
        # Group by user and process each user's interactions
        for user_id, user_group in df.groupby('user_id'):
            # Sort user's interactions by timestamp
            user_interactions = user_group.sort_values('timestamp')
            
            # Skip users with only one interaction
            if len(user_interactions) < 2:
                self.logger.debug(f"Skipping user {user_id} - only {len(user_interactions)} interaction(s)")
                users_skipped += 1
                # Add the single interaction to training
                if len(user_interactions) == 1:
                    row = user_interactions.iloc[0]
                    train_data.append([row['user_id'], row['item_id']])
                continue
            
            # Split: all but last for training, last for testing
            train_interactions = user_interactions.iloc[:-1]  # All except last
            test_interaction = user_interactions.iloc[-1]     # Last interaction
            
            # Add training interactions
            for _, row in train_interactions.iterrows():
                train_data.append([row['user_id'], row['item_id']])
            
            # Add test interaction
            test_data.append([test_interaction['user_id'], test_interaction['item_id']])
            users_processed += 1
        
        self.logger.info(f"Processed {users_processed} users for testing")
        self.logger.info(f"Skipped {users_skipped} users (insufficient interactions)")
        self.logger.info(f"Training interactions: {len(train_data):,}")
        self.logger.info(f"Test interactions: {len(test_data):,}")
        
        return train_data, test_data

    def generate_test_negatives(self, train_data, test_data, num_items):
        """Generate negative samples for each test user."""
        self.logger.info(f"Generating {self.num_negatives} negative samples per test user...")
        
        # Create set of all training interactions for fast lookup
        train_interactions = set()
        user_train_items = {}
        
        for user_id, item_id in train_data:
            train_interactions.add((user_id, item_id))
            if user_id not in user_train_items:
                user_train_items[user_id] = set()
            user_train_items[user_id].add(item_id)
        
        # Add test positives to user item sets (to exclude from negatives)
        for user_id, item_id in test_data:
            if user_id not in user_train_items:
                user_train_items[user_id] = set()
            user_train_items[user_id].add(item_id)
        
        test_negatives = []
        
        for user_id, pos_item_id in test_data:
            user_items = user_train_items.get(user_id, set())
            
            # Generate negative samples
            negatives = set()
            attempts = 0
            max_attempts = self.num_negatives * 10  # Prevent infinite loops
            
            while len(negatives) < self.num_negatives and attempts < max_attempts:
                neg_item = np.random.randint(num_items)
                if neg_item not in user_items:  # Ensure it's truly negative
                    negatives.add(neg_item)
                attempts += 1
            
            if len(negatives) < self.num_negatives:
                self.logger.warning(f"Could only generate {len(negatives)} negatives for user {user_id}")
            
            # Format: (user,pos_item)\tneg1\tneg2\t...
            neg_line = f"({user_id},{pos_item_id})\t" + "\t".join(map(str, sorted(negatives)))
            test_negatives.append(neg_line)
        
        return test_negatives

    def save_splits(self, train_data, test_data, test_negatives):
        """Save train/test splits to files."""
        self.logger.info("Saving train/test splits...")
        
        # Save training data
        train_df = pd.DataFrame(train_data, columns=['user_id', 'item_id'])
        train_df.to_csv(self.train_file, sep='\t', index=False, header=False)
        self.logger.info(f" Saved {len(train_data):,} training interactions to {self.train_file}")
        
        # Save test ratings
        test_df = pd.DataFrame(test_data, columns=['user_id', 'item_id'])
        test_df.to_csv(self.test_rating_file, sep='\t', index=False, header=False)
        self.logger.info(f" Saved {len(test_data):,} test ratings to {self.test_rating_file}")
        
        # Save test negatives
        with open(self.test_negative_file, "w") as f:
            f.write("\n".join(test_negatives))
        self.logger.info(f" Saved test negatives to {self.test_negative_file}")

    def verify_split(self, train_data, test_data):
        """Verify there's no data leakage between train and test."""
        self.logger.info("Verifying train/test split integrity...")
        
        # Convert to sets for fast intersection
        train_interactions = set(tuple(row) for row in train_data)
        test_interactions = set(tuple(row) for row in test_data)
        
        # Check for leakage
        leakage = train_interactions.intersection(test_interactions)
        
        if leakage:
            self.logger.error(f" DATA LEAKAGE DETECTED: {len(leakage)} interactions appear in both train and test!")
            self.logger.error(f"Example leaked interactions: {list(leakage)[:5]}")
            raise RuntimeError("Data leakage detected in train/test split!")
        else:
            self.logger.info(" No data leakage detected - train and test sets are properly separated")
        
        # Additional validation
        train_users = set(row[0] for row in train_data)
        test_users = set(row[0] for row in test_data)
        user_overlap = len(train_users.intersection(test_users))
        
        self.logger.info(f" Train users: {len(train_users):,}")
        self.logger.info(f" Test users: {len(test_users):,}")
        self.logger.info(f" User overlap: {user_overlap:,} ({user_overlap/len(test_users)*100:.1f}% of test users)")

    def build_interaction_matrix(self, train_data):
        """Build sparse interaction matrix from training data."""
        self.logger.info("Building interaction matrix...")
        
        # Get matrix dimensions
        all_users = [row[0] for row in train_data]
        all_items = [row[1] for row in train_data]
        num_users = max(all_users) + 1
        num_items = max(all_items) + 1
        
        # Create sparse matrix
        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        for user_id, item_id in train_data:
            mat[user_id, item_id] = 1.0
        
        self.logger.info(f" Matrix shape: {mat.shape}")
        self.logger.info(f" Matrix density: {mat.nnz / (mat.shape[0] * mat.shape[1]):.6f}")
        
        return mat, num_users, num_items

    def save_results(self):
        """Save preprocessing metadata and statistics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(self.results, f"preprocessing_{timestamp}")
        self.logger.info(" Saved preprocessing metadata to results/reports/")

    def run(self):
        """Execute the complete temporal leave-one-out preprocessing pipeline."""
        self.logger.info("="*60)
        self.logger.info("STARTING TEMPORAL LEAVE-ONE-OUT PREPROCESSING")
        self.logger.info("="*60)
        
        try:
            # Step 1: Load and prepare data
            df = self.load_and_prepare_data()
            
            # Step 2: Perform temporal split
            train_data, test_data = self.temporal_split(df)
            
            # Step 3: Build interaction matrix for item counting
            train_matrix, num_users, num_items = self.build_interaction_matrix(train_data)
            
            # Step 4: Generate test negatives
            test_negatives = self.generate_test_negatives(train_data, test_data, num_items)
            
            # Step 5: Verify split integrity
            self.verify_split(train_data, test_data)
            
            # Step 6: Save all splits
            self.save_splits(train_data, test_data, test_negatives)
            
            # Step 7: Save metadata
            self.results['preprocessing'] = {
                'num_users': int(num_users),
                'num_items': int(num_items),
                'total_original_interactions': int(len(df)),
                'train_interactions': int(len(train_data)),
                'test_interactions': int(len(test_data)),
                'users_with_test': int(len(test_data)),
                'test_coverage': float(len(test_data) / num_users * 100),
                'sparsity': float(1 - (len(train_data) / (num_users * num_items))),
                'split_method': 'temporal_leave_one_out'
            }
            
            self.save_results()
            
            # Step 8: Final summary
            self.logger.info("="*60)
            self.logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            self.logger.info(f" SUMMARY:")
            self.logger.info(f"   Users: {num_users:,}")
            self.logger.info(f"   Items: {num_items:,}")
            self.logger.info(f"   Original interactions: {len(df):,}")
            self.logger.info(f"   Training interactions: {len(train_data):,}")
            self.logger.info(f"   Test interactions: {len(test_data):,}")
            self.logger.info(f"   Test coverage: {len(test_data) / num_users * 100:.1f}% of users")
            self.logger.info(f"   Sparsity: {1 - (len(train_data) / (num_users * num_items)):.4f}")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f" Preprocessing failed: {e}")
            raise
