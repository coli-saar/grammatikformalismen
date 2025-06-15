from typing import List

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    debug: bool = False
    batchsize: int
    ignore_index: int = -100
    epochs: int
    learning_rate: float
    dropout: float = 0.1
    transformer_activation: str = "relu"
    betas: List[float] = [0.9, 0.999]
    limit_train: int = 1000000000
    limit_dev: int = 1000000000

    @staticmethod
    def load(filename) -> "Config":
        with open("config.yml", "r") as f:
            config = Config(**yaml.safe_load(f))
            return config
