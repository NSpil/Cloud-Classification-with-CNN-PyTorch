import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import DataloaderClouds
from DataloaderClouds import Datasetclouds
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cuda:0'
num_classes = 7
batch_size = 16
#text_file = open("confusion_matrix_30.txt", "w")


dataset_val = Datasetclouds(csv_file = 'C:/Users/acer/Desktop/GRSCD/test_texts/GRSCD_test.csv', root_dir ='C:/Users/acer/Desktop/GRSCD/test_images',transform = DataloaderClouds.data_transforms['val'] )
val_set = dataset_val
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle= False)


model = torchvision.models.resnet18(pretrained=True)
model.fc =  torch.nn.Linear(model.fc.in_features, num_classes)

model.to(device)
model.load_state_dict(torch.load('C:/Users/acer/Desktop/CLOUDS/models/model_GRSCD.pkl'))
model.eval()


confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(val_loader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

conf_mat=str(confusion_matrix)
#text_file.write(conf_mat)

print(confusion_matrix.diag()/confusion_matrix.sum(1))

#text_file.close()   