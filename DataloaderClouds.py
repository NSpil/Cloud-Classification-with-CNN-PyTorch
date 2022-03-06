import os
import torch
# from skimage import io
from torch.utils.data import Dataset
from torch import load
import numpy
import pandas as pd
from torchvision import transforms,io


data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize((420,420)),
        transforms.RandomHorizontalFlip(),
         #transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.485], [0.229])
        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),
    'val': transforms.Compose([
        transforms.Resize((420,420)),
        # transforms.CenterCrop(224),
         #transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.485], [0.229])
        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])}


class Datasetclouds(Dataset):
    def __init__(self,csv_file, root_dir,transform=None):
        self.annotations=pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)  #5000
    
    def __getitem__(self, index):
        
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0]).replace("\\","/")
        x = img_path.split(";")
        img_path = x[0]
        image = io.read_image(img_path).float()
        #y_label = torch.tensor(int(self.annotations.iloc[index,1]))
        y_label = int(x[1])
        if self.transform:
            image=self.transform(image)
            
        return(image, y_label)
    
    