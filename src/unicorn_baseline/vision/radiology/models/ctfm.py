from lighter_zoo import SegResNet
from monai.transforms import (
    Compose,
    EnsureType,
    ScaleIntensityRange,
)
from monai.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch

class SegResNetEncoder:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegResNet.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def load_data(self, data): 
        train_transforms = Compose([ 
            EnsureType(),                         
            ScaleIntensityRange(
                a_min=-1024,    
                a_max=2048,    
                b_min=0,        
                b_max=1,        
                clip=True       
            ),
        ])

        train_ds = Dataset(data=data, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        return train_loader


    def encode(self, patches):
        self.model.to(self.device)

        # expand patch to match encoder input requirements
        patch_array = np.expand_dims(patches, axis=(0, 1))
        train_loader = self.load_data(patch_array)

        self.model.eval()
        with torch.no_grad():
            for input in train_loader:
                input = input.to(self.device)

                output = self.model.encoder(input)
                # average pool and flatten the output to fit feature vector requirements
                #output_flat = self.adaptive_pool(output[-1])
                out_flat = output[-1].flatten(start_dim=0)

        return out_flat.cpu().detach().numpy().tolist()
