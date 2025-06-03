import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="configs/experiments/neumf.yaml"):  # Changed default to neumf.yaml
        """Initialize configuration by loading from a YAML file or using defaults."""
        self.config_path = Path(config_path)
        self._load_config()
        self._set_defaults()
        self._create_directories()

    def _load_config(self):
        """Load configuration from a YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f) or {}

    def _set_defaults(self):
        """Set default values for paths, model, and training parameters."""
        # Data paths
        self.raw_data = Path(self._config.get('data', {}).get('raw_data', 'data/raw/u.data'))
        self.train_rating = Path(self._config.get('data', {}).get('train_rating', 'data/processed/u.train.rating'))
        self.test_rating = Path(self._config.get('data', {}).get('test_rating', 'data/processed/u.test.rating'))
        self.test_negative = Path(self._config.get('data', {}).get('test_negative', 'data/processed/u.test.negative'))

        # Model parameters
        self.user_num = self._config.get('model', {}).get('user_num', 943)  # Matches dataset
        self.item_num = self._config.get('model', {}).get('item_num', 1682)  # Matches dataset
        self.factor_num = self._config.get('model', {}).get('factor_num', 32)
        self.num_layers = self._config.get('model', {}).get('num_layers', 3)
        self.dropout = self._config.get('model', {}).get('dropout', 0.0)
        self.model_type = self._config.get('model', {}).get('type', 'NeuMF-end')

        # Training parameters
        self.batch_size = self._config.get('training', {}).get('batch_size', 256)
        self.epochs = self._config.get('training', {}).get('epochs', 20)
        self.lr = self._config.get('training', {}).get('lr', 0.001)
        self.num_ng = self._config.get('training', {}).get('num_ng', 4)
        self.test_num_ng = self._config.get('training', {}).get('test_num_ng', 99)
        self.top_k = self._config.get('training', {}).get('top_k', 10)

        # Distillation parameters
        self.temperature = self._config.get('distillation', {}).get('temperature', 2.0)
        self.alpha = self._config.get('distillation', {}).get('alpha', 0.5)

        # Output paths
        self.output_dir = Path(self._config.get('output', {}).get('dir', 'results'))
        self.log_dir = self.output_dir / 'logs'
        self.model_dir = self.output_dir / 'models'
        self.figure_dir = self.output_dir / 'figures'

    def _create_directories(self):
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key, default=None):
        """Get a configuration value by key with a fallback default."""
        return getattr(self, key, default)

# Singleton instance
config = Config()