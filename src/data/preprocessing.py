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

    def load_and_binarize(self):
        df = pd.read_csv(
            self.raw_path,
            sep='\t',
            names=["user_id", "item_id", "rating", "timestamp"],
            dtype=int
        )
        df["interaction"] = 1
        return df[["user_id", "item_id"]]

    def build_interaction_matrix(self, df):
        num_users = df["user_id"].max() + 1
        num_items = df["item_id"].max() + 1
        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        for _, row in df.iterrows():
            mat[row["user_id"], row["item_id"]] = 1.0
        return mat, num_users, num_items

    def split_and_save(self, df, num_users, num_items):
        df.to_csv(self.train_file, sep='\t', index=False, header=False)
        self.logger.info(f"Saved training data to {self.train_file}")

        test_rating, test_negative = [], []
        grouped = df.groupby("user_id")["item_id"].apply(set).to_dict()

        for user in range(num_users):
            if user not in grouped or not grouped[user]:
                continue

            items = list(grouped[user])
            pos_item = np.random.choice(items)
            test_rating.append(f"{user}\t{pos_item}")

            negatives = set()
            while len(negatives) < self.num_negatives:
                neg_item = np.random.randint(num_items)
                if neg_item not in grouped[user]:
                    negatives.add(neg_item)

            neg_line = f"({user},{pos_item})\t" + "\t".join(map(str, negatives))
            test_negative.append(neg_line)

        with open(self.test_rating_file, "w") as f:
            f.write("\n".join(test_rating))
        with open(self.test_negative_file, "w") as f:
            f.write("\n".join(test_negative))

        self.logger.info(f"Saved test ratings to {self.test_rating_file}")
        self.logger.info(f"Saved test negatives to {self.test_negative_file}")

        self.results['preprocessing'] = {
            'num_users': int(num_users),
            'num_items': int(num_items),
            'interactions': int(len(df)),
            'test_users': int(len(test_rating))
        }

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(self.results, f"preprocessing_{timestamp}")
        self.logger.info("Saved preprocessing metadata and stats to results/reports/")

    def run(self):
        df = self.load_and_binarize()
        matrix, num_users, num_items = self.build_interaction_matrix(df)
        self.split_and_save(df, num_users, num_items)
        self.save_results()
