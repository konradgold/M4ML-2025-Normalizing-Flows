import yaml


class Config:
    def __init__(self, path: str):
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {path}")
            config = {}
        self.num_epochs = config.get("num_epochs", 150)
        self.batch_size = config.get("batch_size", 128)
        self.input_dim = config.get("input_dim", 2)
        self.num_layers = config.get("num_layers", 8)