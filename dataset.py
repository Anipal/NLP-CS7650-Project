from torch.utils.data import Dataset, DataLoader
from embeddings import *
import json
import pickle


def load_file(path):
    extension = path.split('.')[-1]
    if extension not in [ "json", "pkl" ]:
        print("File type not supported")
        exit()

    data_path = open(path)

    if extension == "json":
        data = json.load( data_path )
    elif extension == "pkl":
        data = pickle.load( data_path )
    
    print("Loading :", path)
    return data
    

class VQADataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        generate_language_embeddings(config)
        generate_vision_embeddings(config)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        img_id = data['image_name']
        q_id = data['qid']

        q_feat = torch.load(f"./vqa_rad/biomed_roberta_convnext/embeddings/questions/{q_id}.pt")
        img_feat = torch.load(f"./vqa_rad/biomed_roberta_convnext/embeddings/images/{img_id}.pt")
        label = data['labels'][0]

        return img_feat.cuda(), q_feat.cuda(), torch.tensor(label, dtype=torch.long).cuda()