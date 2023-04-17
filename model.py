import torch
from torch import nn

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()

        self.image = nn.Sequential(
         nn.Linear(in_features=1024, out_features=512)   
        )

        self.txt = nn.Linear(in_features=768, out_features=512)
        
        # for run_0 and run_2
        # self.linstack = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=256),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=256, out_features=128),
        #     nn.Linear(in_features=128, out_features=458),
        # )

        # for run_1 and run_3
        self.linstack = nn.Sequential(
            nn.Linear(512,256),
            nn.Dropout(p=0.2),
            nn.Linear(256,128),
            nn.Linear(128,458),
        )

    def forward(self, xImage, xText):
        
        xImage = self.image(xImage)
        xText = self.txt(xText)

        x = xImage*xText

        x = self.linstack(x)

        return x
