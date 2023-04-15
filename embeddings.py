from transformers import AutoTokenizer, AutoModel
import torch
from dataset import load_file
from tqdm import tqdm
import yaml
import os
import timm
from PIL import Image
import torchvision.transforms as transforms


def generate_language_embeddings( config ):
    # ./embeddings/language/<language_model_name>/<question_id.pt>
    save_path = os.path.join( config.language_embeddings_folder, config.language_model_name )
    if os.path.exists(save_path):
        print("Language model embeddings already exist")
        return
    else:
        os.makedirs(save_path)


    tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)
    model = AutoModel.from_pretrained(config.language_model_name)
    model = model.to(config.device)


    for dataset in [config.train, config.test]:
        dataset_file = load_file(dataset)
        for data in tqdm( dataset_file ): 
            q_tokenized = tokenizer.encode(data['question'], max_length=512, truncation=True)
            q_tokenized = torch.tensor([q_tokenized]).to(config.device)
            q_encoded_sequence = model(q_tokenized)[0]
            q_embedding = torch.mean(q_encoded_sequence, dim=1)
            torch.save(q_embedding, os.path.join(save_path, f"{ data['qid'] }.pt") )

def generate_vision_embeddings( config ):
    # ./embeddings/vision/<vision_model_name>/<image_id.pt>
    save_path = os.path.join( config.vision_embeddings_folder, config.vision_model_name )
    if os.path.exists(save_path):
        print("Vision model embeddings already exist")
        return
    else:
        os.makedirs(save_path)
    
    # Load pre-trained ConvNeXt model
    model = timm.create_model(config.vision_model_name, pretrained=True)
    model = model.to(config.device)
    # Remove final classification layer
    model.reset_classifier(0)

    images_path = os.listdir(config.images) 
    for image_id in tqdm(images_path):
        # Load and preprocess the image
        image = Image.open(os.path.join(config.images, image_id)).convert('RGB')
        image = image.resize((224, 224))
        image = transforms.ToTensor()(image).to(config.device)
        image = image.unsqueeze(0)
        # Extract features using the ConvNeXt model
        with torch.no_grad():
            features = model(image)
        # Save the extracted features
        torch.save(features, os.path.join(save_path, f"{ image_id }.pt") )