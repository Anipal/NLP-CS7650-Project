# NLP-CS7650-Project

## Setting up the environment
```
   conda env create -f environment.yml
``` 

## Changing the config
```
    # parameters that can be changed
    language_model_name : "emilyalsentzer/Bio_ClinicalBERT"
    vision_model_name : "convnext_base"
    experiment_name: "bioclinicalBERT_convnext"
```

## Training
``` 
    python train.py
```

## Testing
```
    python test.py
```