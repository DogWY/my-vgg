import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np

from tqdm import tqdm
import wandb

import random
import datetime
import itertools

from train import train_epoch
from val import val_epoch
from VGG import VGG

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device:{device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.Flowers102(root="../data", split="train", download=True, transform=transform)
val_dataset = datasets.Flowers102(root="../data", split="val", download=True, transform=transform)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)


def main():
    lr_list = [1e-3]
    num_epochs_list = [1000]
    momentum_list = [0.9]
    batch_size_list = [256]
    seed_list = [42]
    
    for lr, num_epochs, momentum, batch_size, seed in itertools.product(lr_list, num_epochs_list, momentum_list, batch_size_list, seed_list):
        
        print(f"lr:{lr}, num_epochs:{num_epochs}, momentum:{momentum}, batch_size:{batch_size}, seed:{seed}")
        
        current = datetime.datetime.now()
        current_str = current.strftime('%Y-%m-%d %H:%M:%S')
        
        wandb.init(
            project="VGG",
            name=f"ITX {current_str}",
            config={
                "learning rate": lr,
                "num_epochs": num_epochs,
                "momentum": momentum,
                "batch_size": batch_size,
                "seed": seed
            }
        )
        
        set_seed(seed)
        
        model = VGG(in_channels=3, out_channels=102).to(device)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        
        print(model)
        
        for epoch in range(num_epochs):
            train_loss, train_correct = train_epoch(model, train_loader, optimizer, criterion, epoch+1, num_epochs, device)
            wandb.log({
                'train_loss':train_loss,
                'train_acc':train_correct/len(train_dataset)
            }, step=epoch) 
            
            val_loss, val_correct = val_epoch(model, val_loader, optimizer, criterion, epoch+1, num_epochs, device)
            wandb.log({
                'val_loss': val_loss,
                'val_acc': val_correct/len(val_dataset)
            }, step=epoch)
        
        wandb.finish()
        

if __name__ == '__main__':
    main()
    pass
