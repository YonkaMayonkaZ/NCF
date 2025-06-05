import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF

class NMFRecommender:
    def __init__(self, n_components=10, random_state=42, max_iter=300):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.n_parameters = 0
        
    def fit(self, train_matrix):
        if sp.issparse(train_matrix):
            train_dense = train_matrix.toarray()
        else:
            train_dense = train_matrix
        
        # Ensure non-negative values (NMF requirement)
        train_dense = np.maximum(train_dense, 0)
        
        self.n_users, self.n_items = train_dense.shape
        
        self.model = NMF(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            init='random',
            solver='mu',  # multiplicative update solver works better for sparse data
            beta_loss='frobenius'
        )
        
        self.user_factors = self.model.fit_transform(train_dense)
        self.item_factors = self.model.components_.T
        
        self.n_parameters = self.n_users * self.n_components + self.n_items * self.n_components
        
        # Debug: check if factorization is reasonable
        reconstruction_error = self.model.reconstruction_err_
        print(f"  NMF reconstruction error: {reconstruction_error:.4f}")
        
        return self
    
    def predict(self, user_ids, item_ids):
        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        
        predictions = []
        for u, i in zip(user_ids, item_ids):
            if u >= self.n_users or i >= self.n_items:
                pred = 0.0
            else:
                pred = np.dot(self.user_factors[u], self.item_factors[i])
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_n_parameters(self):
        return self.n_parameters

class NMFEvaluator:
    def __init__(self, model, test_data, train_matrix, top_k=10):
        self.model = model
        self.test_data = test_data
        self.train_matrix = train_matrix
        self.top_k = top_k
    
    def evaluate(self):
        hits = []
        ndcgs = []
        
        # NCF test data format: first item per user is positive, next 99 are negatives
        # Group by user and process in chunks of 100 (1 pos + 99 neg)
        user_data = {}
        for user, item in self.test_data:
            if user not in user_data:
                user_data[user] = []
            user_data[user].append(item)
        
        for user_id, items in user_data.items():
            if user_id >= self.train_matrix.shape[0]:
                continue
            
            # Items list has: [positive_item, neg1, neg2, ..., neg99]
            # Take all items (should be 100 items per user)
            candidate_items = items
            user_array = [user_id] * len(candidate_items)
            predictions = self.model.predict(user_array, candidate_items)
            
            # The first item is the positive test item
            positive_item = items[0]
            
            # Rank all items by prediction scores
            item_score_pairs = list(zip(candidate_items, predictions))
            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            ranked_items = [item for item, score in item_score_pairs]
            
            # Check if positive item is in top-k
            if positive_item in ranked_items[:self.top_k]:
                hits.append(1)
                position = ranked_items.index(positive_item)
                ndcg = 1.0 / np.log2(position + 2)
                ndcgs.append(ndcg)
            else:
                hits.append(0)
                ndcgs.append(0)
        
        print(f"    Evaluated {len(user_data)} users")
        hr = np.mean(hits) if hits else 0.0
        ndcg = np.mean(ndcgs) if ndcgs else 0.0
        return hr, ndcg