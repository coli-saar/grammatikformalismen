from min_train import train
import simplemodel
from util import Config

UD_DATASET = "en_ewt"
config = Config.load("config.yml")
model = simplemodel.MlpParsingModel(roberta_id="xlm-roberta-base", dropout=config.dropout, activation=config.transformer_activation)

train(config, UD_DATASET, model)


