from embeddings import *
from utilities import *
import yaml
import os
import torch

module_dir = os.path.dirname(__file__)
config_path = os.path.join(module_dir, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = objectview(config)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')






