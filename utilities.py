import json
import pickle
import time
import os
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch

def load_file(path):
    extension = path.split('.')[-1]
    if extension not in [ "json", "pkl" ]:
        print("File type not supported")
        exit()


    if extension == "json":
        data_path = open(path)
        data = json.load( data_path )
    elif extension == "pkl":
        with open(path, 'rb') as f:
            data = pickle.load(f)
    
    print("Loading :", path)
    return data
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def setup_logging(config):
    run_id  = str(int(time.time())%1e7)
    save_path = os.path.join(config.results_folder, config.experiment_name, "runs", run_id)

    logs_path   = os.path.join(save_path, "logs")
    models_path = os.path.join(save_path, "models")
    os.makedirs(logs_path)
    os.makedirs(models_path)

    writer = SummaryWriter(logs_path)
    best_model = models_path +'best.pth'

    return writer, best_model

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



