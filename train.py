from utilities import objectview, setup_logging, set_seed
from model import SimpleClassifier
from dataset import VQADataset
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import torch
from torch import nn
from sklearn.metrics import f1_score, accuracy_score



parser = argparse.ArgumentParser(description='VQA-RAD training')
parser.add_argument('--config_path', type=str, default='config.yaml',help='path to config file')
args = parser.parse_args()

set_seed(0)
# config settings -- later move to main.py where we can call train.py
module_dir = os.path.dirname(__file__)
config_path = os.path.join(module_dir, args.config_path)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = objectview(config)
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#dataset
train_ds = VQADataset(config, type = "train")
val_ds = VQADataset(config, type = "val")
test_ds = VQADataset(config, type = "test")

train_dataloader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dataloader = DataLoader(test_ds, batch_size=128, shuffle=False)

# model and optimizer -- add these to config to run experiments easily
model = SimpleClassifier().to(config.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = config.epochs
run_id = config.run_id
# setup logging and the results directory for the current run
writer, best_model_path = setup_logging(config)
best_f1_val = 0.0
best_acc_val = 0.0
best_f1_train = 0.0
best_acc_train = 0.0
# yet to edit after this
for epoch in range(num_epochs):
    running_loss_train = 0.0
    gtLabelsList = []
    predictedLabelsList = []
    model.train()
    for i, data in enumerate(train_dataloader, 0):
        inputImg, inputTxt, labels , idx= data

        optimizer.zero_grad()

        outputs = model(inputImg, inputTxt).squeeze()

        loss_train = criterion(outputs, labels)
        loss_train.backward()
        optimizer.step()

        running_loss_train += loss_train.item()

        outputLabels = torch.argmax(outputs, dim=1)
        
        predictedLabelsList.extend(list(outputLabels.detach().cpu().numpy()))
        gtLabelsList.extend(list(labels.detach().cpu().numpy()))

    # print("pred",predictedLabelsList )
    # print("Gt" , gtLabelsList)

    f1_train = f1_score(predictedLabelsList, gtLabelsList, average = 'weighted')
    accuracy_train = accuracy_score(predictedLabelsList, gtLabelsList)

    if(f1_train>best_f1_train):
      best_f1_train = f1_train

    if(accuracy_train>best_acc_train):
      best_acc_train = accuracy_train

    writer.add_scalar("Loss/train", running_loss_train/len(train_dataloader), epoch)
    writer.add_scalar("Accuracy/train", accuracy_train , epoch)
    writer.add_scalar("F1/train", f1_train , epoch)

    print(f"EPOCH:{epoch} TRAIN f1:{f1_train:.2f}, accuracy:{accuracy_train:.2f}")

    if(epoch%2==0):
        running_loss_val = 0.0
        gtLabelsList = []
        predictedLabelsList = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                inputImg, inputTxt, labels , idx= data

                # print(labels)

                outputs = model(inputImg, inputTxt).squeeze()
                outputLabels = torch.argmax(outputs, dim=1)
                # loss_val = criterion(outputs, labels)
                # running_loss_val += loss_val.item()
                predictedLabelsList.extend(list(outputLabels.detach().cpu().numpy()))
                gtLabelsList.extend(list(labels.detach().cpu().numpy()))

            
            f1_val = f1_score(predictedLabelsList, gtLabelsList, average = 'weighted')
            accuracy_val = accuracy_score(predictedLabelsList, gtLabelsList)

            
            if(accuracy_val > best_acc_val):
        
                best_acc_val = accuracy_val
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_train':running_loss_train/len(train_dataloader),
                    'accuracy_train':accuracy_train,
                    'f1_train':f1_train, 
                    'loss_val':running_loss_val/len(val_dataloader),
                    'accuracy_val':accuracy_val,
                    'f1_val':f1_val
                    }, best_model_path)
                    
            writer.add_scalar("Loss/val", running_loss_val/len(val_dataloader), epoch)
            writer.add_scalar("Accuracy/val", accuracy_val , epoch)
            writer.add_scalar("F1/val", f1_val , epoch)
            print(f"\n VAL f1:{f1_val:.2f}, accuracy:{accuracy_val:.2f}\n")

            if(f1_val>best_f1_val):
                best_f1_val = f1_val

print('Training finished')

print(f"Best F1 Train score:{best_f1_train:.2f}")
print(f"Best Acc Train score:{best_acc_train:.2f}")
print(f"Best F1 Test score:{best_f1_val:.2f}")
print(f"Best Acc Test score:{best_acc_val:.2f}")
