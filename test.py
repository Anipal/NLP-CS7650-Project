from utilities import objectview, setup_logging, set_seed
from model import SimpleClassifier
from dataset import VQADataset
from torch.utils.data import DataLoader
import yaml
import os
import torch
from torch import nn
from sklearn.metrics import f1_score, accuracy_score
import argparse


parser = argparse.ArgumentParser(description='VQA-RAD testing')
parser.add_argument('--config_path', type=str, default='config.yaml',help='path to config file')
args = parser.parse_args()

set_seed(0)
# config settings -- later move to main.py where we can call test.py
module_dir = os.path.dirname(__file__)
config_path = os.path.join(module_dir, args.config_path)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = objectview(config)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#dataset
test_ds = VQADataset(config, type = "test")
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

# model and optimizer -- add these to config to run experiments easily
model = SimpleClassifier().to(config.device)
# setup logging and the results directory for the current run
writer, best_model_path = setup_logging(config)
# checkpoint = torch.load(best_model_path)
# model.load_state_dict(checkpoint['model_state_dict'])


# yet to edit after this

gtLabelsList = []
predictedLabelsList = []

model.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        inputImg, inputTxt, labels = data
        if not(labels.item() == -1):

            outputs = model(inputImg, inputTxt).squeeze()

            outputLabels = torch.argmax(outputs, dim=0)
            
            predictedLabelsList.append(int(outputLabels.detach().cpu().numpy()))
            gtLabelsList.append(int(labels.detach().cpu().numpy()))

f1_test = f1_score(predictedLabelsList, gtLabelsList, average = 'weighted')
acc_test = accuracy_score(predictedLabelsList, gtLabelsList)

print('Testing finished')

print(f"F1 Test score:{f1_test:.2f}")
print(f"Acc Test score:{acc_test:.2f}")
