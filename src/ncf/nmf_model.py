import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp

class NMFRecommender:
    """
    Non-negative Matrix Factorization for recommendation.
    Adapted for top-k recommendation evaluation like NCF.
    """
    
    def __init__(self, n_components=10, random_state=42, max_iter=200):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.n_parameters = 0
        
    def fit(self, train_matrix):
        """
        Fit NMF model on training data.
        
        Args:
            train_matrix: scipy sparse matrix (users x items)
        """
        # Convert to dense for sklearn NMF
        train_dense = train_matrix.toarray()
        
        # Initialize NMF
        self.model = NMF(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            init='random'
        )
        
        # Fit the model
        self.user_factors = self.model.fit_transform(train_dense)
        self.item_factors = self.model.components_.T
        
        # Calculate number of parameters
        n_users, n_items = train_matrix.shape
        self.n_parameters = (n_users + n_items) * self.n_components
        
        return self
    
    def predict(self, user_ids, item_ids):
        """
        Predict ratings for given user-item pairs.
        
        Args:
            user_ids: array of user indices
            item_ids: array of item indices
            
        Returns:
            predictions: array of predicted ratings
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not fitted yet")
        
        # Compute predictions
        predictions = []
        for u, i in zip(user_ids, item_ids):
            pred = np.dot(self.user_factors[u], self.item_factors[i])
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_user(self, user_id):
        """
        Predict ratings for all items for a given user.
        
        Args:
            user_id: user index
            
        Returns:
            predictions: array of predicted ratings for all items
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not fitted yet")
        
        return np.dot(self.user_factors[user_id], self.item_factors.T)
    
    def get_top_k_items(self, user_id, k=10, exclude_seen=None):
        """
        Get top-k item recommendations for a user.
        
        Args:
            user_id: user index
            k: number of recommendations
            exclude_seen: set of item indices to exclude (already seen items)
            
        Returns:
            top_k_items: array of top-k item indices
            top_k_scores: array of corresponding prediction scores
        """
        # Get all predictions for this user
        all_predictions = self.predict_user(user_id)
        
        # Exclude already seen items if provided
        if exclude_seen is not None:
            for item_id in exclude_seen:
                all_predictions[item_id] = -np.inf
        
        # Get top-k items
        top_k_indices = np.argsort(all_predictions)[::-1][:k]
        top_k_scores = all_predictions[top_k_indices]
        
        return top_k_indices, top_k_scores
    
    def get_n_parameters(self):
        """Return the number of model parameters."""
        return self.n_parameters

class NMFEvaluator:
    """
    Evaluator for NMF model using NCF-style evaluation.
    """
    
    def __init__(self, model, test_data, train_matrix, top_k=10):
        self.model = model
        self.test_data = test_data  # List of [user, item] pairs from NCF test format
        self.train_matrix = train_matrix
        self.top_k = top_k
    
    def evaluate(self):
        """
        Evaluate the NMF model using HR@K and NDCG@K metrics.
        Following the same evaluation protocol as NCF.
        
        Returns:
            hr: Hit Ratio at K
            ndcg: NDCG at K
        """
        hits = []
        ndcgs = []
        
        # Set random seed for reproducible negative sampling
        np.random.seed(42)
        
        # Group test data by user
        user_test_items = {}
        for user, item in self.test_data:
            if user not in user_test_items:
                user_test_items[user] = []
            user_test_items[user].append(item)
        
        print(f"Evaluating {len(user_test_items)} users with test data...")
        
        for user_id, test_items in user_test_items.items():
            # Skip users not in training matrix
            if user_id >= self.train_matrix.shape[0]:
                continue
                
            # Get items this user has interacted with in training
            seen_items = set(self.train_matrix[user_id].nonzero()[1])
            
            # For each test item, evaluate in the context of 99 negative samples
            for test_item in test_items:
                # Skip if test item was seen in training (shouldn't happen but safety check)
                if test_item in seen_items:
                    continue
                
                # Create candidate set: test_item + 99 negative samples
                item_pool = set(range(self.train_matrix.shape[1])) - seen_items - {test_item}
                
                if len(item_pool) < 99:
                    # If not enough negatives, use all available
                    negative_items = list(item_pool)
                else:
                    negative_items = np.random.choice(list(item_pool), 99, replace=False)
                
                candidate_items = [test_item] + list(negative_items)
                
                # Get predictions for candidate items
                user_ids = [user_id] * len(candidate_items)
                try:
                    predictions = self.model.predict(user_ids, candidate_items)
                except Exception as e:
                    print(f"Error predicting for user {user_id}: {e}")
                    continue
                
                # Rank items by prediction scores
                item_score_pairs = list(zip(candidate_items, predictions))
                item_score_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Find position of test item
                ranked_items = [item for item, score in item_score_pairs]
                
                try:
                    position = ranked_items.index(test_item)
                except ValueError:
                    print(f"Test item {test_item} not found in ranked items for user {user_id}")
                    continue
                
                # Calculate metrics
                if position < self.top_k:  # 0-indexed, so < top_k means in top-k
                    hits.append(1)
                    # Calculate NDCG (position is 0-indexed)
                    ndcg = 1.0 / np.log2(position + 2)  # +2 because log2(1) = 0
                    ndcgs.append(ndcg)
                else:
                    hits.append(0)
                    ndcgs.append(0)
        
        print(f"Evaluated {len(hits)} test cases")
        print(f"Hits: {sum(hits)}/{len(hits)} = {sum(hits)/len(hits):.4f}")
        
        hr = np.mean(hits) if hits else 0
        ndcg = np.mean(ndcgs) if ndcgs else 0
        
        return hr, ndcg

def run_nmf_experiment(train_matrix, test_data, n_components_list, num_runs=10):
    """
    Run NMF experiments with different numbers of components.
    
    Args:
        train_matrix: Training interaction matrix
        test_data: Test data in NCF format
        n_components_list: List of component numbers to try
        num_runs: Number of runs for statistical analysis
        
    Returns:
        results: Dictionary with results for each component number
    """
    results = {}
    
    for n_comp in n_components_list:
        print(f"\nTesting NMF with {n_comp} components...")
        
        run_results = []
        
        for run in range(num_runs):
            # Train NMF model
            nmf_model = NMFRecommender(n_components=n_comp, random_state=42+run)
            nmf_model.fit(train_matrix)
            
            # Evaluate
            evaluator = NMFEvaluator(nmf_model, test_data, train_matrix)
            hr, ndcg = evaluator.evaluate()
            
            run_results.append({
                'hr': hr,
                'ndcg': ndcg,
                'parameters': nmf_model.get_n_parameters()
            })
        
        # Compute statistics
        hrs = [r['hr'] for r in run_results]
        ndcgs = [r['ndcg'] for r in run_results]
        params = run_results[0]['parameters']  # Same for all runs
        
        results[n_comp] = {
            'hr_mean': np.mean(hrs),
            'hr_std': np.std(hrs),
            'ndcg_mean': np.mean(ndcgs),
            'ndcg_std': np.std(ndcgs),
            'parameters': params
        }
        
        print(f"{n_comp} components: HR@10={np.mean(hrs):.4f}+/-{np.std(hrs):.4f}, "
              f"NDCG@10={np.mean(ndcgs):.4f}+/-{np.std(ndcgs):.4f}, Params={params:,}")
    
    return results