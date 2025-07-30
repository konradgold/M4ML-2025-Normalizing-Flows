import yaml


class Config:
    def __init__(self, path: str):
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {path}")
            config = {}
        # Set default values for configuration parameters
        # If a parameter is not found in the config file, it will use the default value.
        # Add to taste
        self.num_epochs = config.get("num_epochs", 100)
        self.batch_size = config.get("batch_size", 128)
        self.input_dim = config.get("input_dim", 2)
        self.num_layers = config.get("num_layers", 4)
        self.hidden_size = config.get("hidden_size", 32)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.samples = config.get("samples", 12000)
        self.log_interval = config.get("log_interval", 1)