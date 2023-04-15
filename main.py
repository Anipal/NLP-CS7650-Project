import json 
import pickle
import torch
from tqdm import tqdm
import timm
from PIL import Image
import os
import re
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertModel, BertTokenizer
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter       # a_embedding = torch.mean(a_encoded_sequence, dim=1)

q_path = 'vqa_rad/baseline/embeddings/questions'
img_path = 'vqa_rad/baseline/embeddings/images'

################## READ DATA #######################

with open('data/cache/train_target.pkl', 'rb') as f:
    data_train = pickle.load(f)
print('Number of instances of train', len(data_train))
with open('data/cache/train_target.pkl', 'rb') as f:
    data_test = pickle.load(f)
print('Number of instances of val', len(data_test))

f_train = open('med-vqa/data/trainset.json')
data_train_qs = json.load(f_train)
f_test = open('med-vqa/data/testset.json')
data_test_qs = json.load(f_test)

with open('med-vqa/data/cache/trainval_ans2label.pkl', 'rb') as f:
    label_train = pickle.load(f)
print('Total classes', len(label_train.keys()))

################## CREATE DIRS #######################
try:
   os.makedirs(q_path, exist_ok=True)
except:
    pass
try:
   os.makedirs(img_path, exist_ok=True)
except:
    pass
################## INITIALIZE TEXT PRETRSINED MOEL #######################

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name) # Initializing tokenizer
model = BertModel.from_pretrained(model_name) # Initializing model
model = model.cuda()

################## SAVE EMBEDDINGS #######################
#Train
for dataPoint in tqdm(data_train): #Make sure to run with dataTrain, dataVal & dataTest

  question_id = dataPoint['qid']
  for data in data_train_qs:
    if data['qid'] == question_id:
      qSent = data['question']

  q_ids = tokenizer.encode(qSent, max_length=512, truncation=True)

  q_ids = torch.tensor([q_ids]).cuda()

  q_outputs = model(q_ids)
  q_encoded_sequence = q_outputs[0]

  q_embedding = torch.mean(q_encoded_sequence, dim=1)

  torch.save(q_embedding, f'./vqa_rad/baseline/embeddings/questions/{question_id}.pt')

  #Val

for dataPoint in tqdm(data_test): #Make sure to run with dataTrain, dataVal & dataTest

  question_id = dataPoint['qid']
  for data in data_test_qs:
    if data['qid'] == question_id:
      qSent = data['question']

  q_ids = tokenizer.encode(qSent, max_length=512, truncation=True)
 
  q_ids = torch.tensor([q_ids]).cuda()

  q_outputs = model(q_ids)
  q_encoded_sequence = q_outputs[0]

  q_embedding = torch.mean(q_encoded_sequence, dim=1)

  torch.save(q_embedding, f'./vqa_rad/baseline/embeddings/questions/{question_id}.pt')

################## IMAGE EMBEDDINGS #######################

# Load pre-trained ConvNeXt model
model = timm.create_model('convnext_base', pretrained=True)
model = model.cuda()

# Remove final classification layer
model.reset_classifier(0)

################## SAVE EMBEDDINGS #######################
#Train
count = 0
for image_path in tqdm(data_train):
  image_name = image_path['image_name']
  # print(image_path)

  # Load and preprocess the image
  image = Image.open("./med-vqa/data/images/" + image_name).convert('RGB')
  image = image.resize((224, 224))
  image = transforms.ToTensor()(image).cuda()
  image = image.unsqueeze(0)


  # Extract features using the ConvNeXt model
  with torch.no_grad():
      features = model(image)

  # Save the extracted features
  torch.save(features, f'./vqa_rad/baseline/embeddings/images/{image_name}.pt')
#Val

count = 0
for image_path in tqdm(data_test):
  image_name = image_path['image_name']
  # print(image_path)

  # Load and preprocess the image
  image = Image.open("./med-vqa/data/images/" + image_name).convert('RGB')
  image = image.resize((224, 224))
  image = transforms.ToTensor()(image).cuda()
  image = image.unsqueeze(0)


  # Extract features using the ConvNeXt model
  with torch.no_grad():
      features = model(image)

  # Save the extracted features
  torch.save(features, f'./vqa_rad/baseline/embeddings/images/{image_name}.pt')

################## DATALOADER #######################
class VQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        dataPoint = self.data[idx]
        # print(dataPoint)
        img_id = dataPoint['image_name']
        q_id = dataPoint['qid']

        q_feat = torch.load(f"./vqa_rad/baseline/embeddings/questions/{q_id}.pt")
        img_feat = torch.load(f"./vqa_rad/baseline/embeddings/images/{img_id}.pt")
        label = dataPoint['labels'][0]


        return img_feat.cuda(), q_feat.cuda(), torch.tensor(label, dtype=torch.long).cuda()

train_ds = VQADataset(data_train)
val_ds = VQADataset(data_test)

################## MODEL #######################
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()

        self.image = nn.Sequential(
         nn.Linear(in_features=1024, out_features=512)   
        )

        self.txt = nn.Linear(in_features=768, out_features=512)
        
        self.linstack = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.Linear(in_features=128, out_features=458),
        )

    def forward(self, xImage, xText):
        
        xImage = self.image(xImage)
        xText = self.txt(xText)

        x = xImage*xText

        x = self.linstack(x)

        return x

################## TRAINING #######################

# Instantiate the model, loss function, and optimizer
best_acc = 0.0
model = SimpleClassifier().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
run_id = 0
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)
writer = SummaryWriter(f"./vqa_rad/baseline/runs/{run_id}/logs/")
save_path = f"./vqa_rad/baseline/runs/{run_id}/models/"
try:
  os.makedirs(f"./vqa_rad/baseline/runs/{run_id}/logs")
except:
  pass
try:
  os.makedirs(f"./vqa_rad/baseline/runs/{run_id}/models")
except:
  pass  
save_path_model = save_path +'best.pth'
# Training loop
for epoch in range(num_epochs):
    running_loss_train = 0.0
    running_loss_val = 0.0

    gtLabelsList = []
    predictedLabelsList = []

    for i, data in enumerate(train_dl, 0):
        inputImg, inputTxt, labels = data

        optimizer.zero_grad()

        outputs = model(inputImg, inputTxt).squeeze()

        loss_train = criterion(outputs, labels)
        loss_train.backward()
        optimizer.step()

        running_loss_train += loss_train.item()

        outputLabels = torch.argmax(outputs, dim=1)

        predictedLabelsList.extend(list(outputLabels.detach().cpu().numpy()))
        gtLabelsList.extend(list(labels.detach().cpu().numpy()))

    
    f1_train = f1_score(predictedLabelsList, gtLabelsList, average = 'micro')
    accuracy_train = accuracy_score(predictedLabelsList, gtLabelsList)
    writer.add_scalar("Loss/train", running_loss_train/len(train_dl), epoch)
    writer.add_scalar("Accuracy/train", accuracy_train , epoch)
    writer.add_scalar("F1/train", f1_train , epoch)

    print(f"TRAIN f1:{f1_train}, accuracy:{accuracy_train}")

    if(epoch%2==0):
      with torch.no_grad():

        gtLabelsList = []
        predictedLabelsList = []

        for i, data in enumerate(val_dl, 0):
            inputImg, inputTxt, labels = data

            outputs = model(inputImg, inputTxt).squeeze()
            outputLabels = torch.argmax(outputs, dim=1)
            loss_val = criterion(outputs, labels)
            running_loss_val += loss_val.item()
            predictedLabelsList.extend(list(outputLabels.detach().cpu().numpy()))
            gtLabelsList.extend(list(labels.detach().cpu().numpy()))

        
        f1_val = f1_score(predictedLabelsList, gtLabelsList, average = 'micro')
        accuracy_val = accuracy_score(predictedLabelsList, gtLabelsList)
        if(accuracy_val > best_acc):
          
          best_acc = accuracy_val
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss_train':running_loss_train/len(train_dl),
              'accuracy_train':accuracy_train,
              'f1_train':f1_train, 
              'loss_val': running_loss_val/len(val_dl),
              'accuracy_val':accuracy_val,
              'f1_val':f1_val
              }, save_path_model)
        writer.add_scalar("Loss/val", running_loss_val/len(val_dl), epoch)
        writer.add_scalar("Accuracy/val", accuracy_val , epoch)
        writer.add_scalar("F1/val", f1_val , epoch)
        print(f"\n VAL f1:{f1_val}, accuracy:{accuracy_val}\n")


    

print('Training finished')

############################################################################################