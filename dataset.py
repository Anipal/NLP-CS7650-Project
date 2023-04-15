from torch.utils.data import Dataset
from embeddings import *
from utilities import *    

class VQADataset(Dataset):
    def __init__(self, config, type = "train"):
        if type not in ["train", "val", "test"]:
            print("Incorrect type for dataloader.")
            exit()
        
        if type=="train":
            self.data = load_file(config.train_dataloader)
        elif type = "val":
            elf.data = load_file(config.val_dataloader) 
        elif type=="test":
            self.data = load_file(config.test_dataloader)

        self.config = config
        generate_language_embeddings(config)
        generate_vision_embeddings(config)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data   = self.data[idx]
        img_id = data['image_name']
        q_id   = data['qid']

        q_feat   = torch.load(os.path.join( self.config.language_embeddings_folder, self.config.language_model_name, f'{q_id}.pt' ))
        img_feat = torch.load(os.path.join( self.config.vision_embeddings_folder, self.config.vision_model_name, f'{img_id}.pt' ))
        label = data['labels'][0]

        return img_feat.to(self.config.device), q_feat.to(self.config.device), torch.tensor(label, dtype=torch.long).to(self.config.device)