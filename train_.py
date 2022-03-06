import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import DataloaderClouds
from DataloaderClouds import Datasetclouds
import copy
from torch.optim import lr_scheduler

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cuda:0'
im_channel = 3
num_classes = 5
learnig_rate= 0.1
batch_size = 32
num_epochs=60
text_file = open("text.txt", "w")

def check_accuracy(loader, model):
    num_correct =0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.to(device)
            
            scores = model(x)
            _, predictions =scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
        text_file.write(f'{float(num_correct)/float(num_samples)*100} ')
        print('')
    result = float(num_correct)/float(num_samples)*100 
    return result
        
    model.train()

dataset_train=Datasetclouds(csv_file ='C:/Users/ellab2/Desktop/CLOUDS/csv_clouds_train.csv', root_dir = 'C:/Users/ellab2/Desktop/CLOUDS/DataClouds', transform = DataloaderClouds.data_transforms['train'])
dataset_val = Datasetclouds(csv_file = 'C:/Users/ellab2/Desktop/CLOUDS/csv_clouds_validation.csv', root_dir ='C:/Users/ellab2/Desktop/CLOUDS/DataClouds',transform = DataloaderClouds.data_transforms['val'] )

train_set = dataset_train
val_set = dataset_val

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle= True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle= False)

model = torchvision.models.resnet18(pretrained=True)
model.fc =  torch.nn.Linear(model.fc.in_features, num_classes)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=learnig_rate)

scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
epoch_acc=0.0

for epoch in range(num_epochs):
    losses = 0
    model.train()
    for  batch_idx, (data, targets) in enumerate(train_loader):
        data= data.to(device)
        targets= targets.to(device)
        
        scores=model(data)
        loss= criterion(scores,targets)
       
        losses += float(loss)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        # print(f'Train iteration loss at epoch {epoch} is {loss}')
    scheduler.step()    
    print(f'Train cost at epoch {epoch} is {losses/(batch_idx+1)}')
    text_file.write(f'{losses/(batch_idx+1)} ')
    print("Checking accuracy on Training Set")
    check_accuracy(train_loader, model)
    model.eval()
    losses = 0
    for  batch_idx, (data, targets) in enumerate(val_loader):
        data= data.to(device=device)
        targets= targets.to(device=device)
        
        scores=model(data)
        loss= criterion(scores,targets)
        losses += float(loss) 
        
        
    print(f'Validation cost at epoch {epoch} is {losses/(batch_idx+1)}')
    text_file.write(f'{losses/(batch_idx+1)} ')
    print("Checking accuracy on Validation Set")
    epoch_acc = check_accuracy(val_loader, model)
    text_file.write('\n')
    print('-----'*10)
    print('')
    epoch_acc=float(epoch_acc)
    if  float(epoch_acc) > float(best_acc):
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'C:/Users/ellab2/Desktop/CLOUDS' + '/model.pkl')

text_file.close()                               