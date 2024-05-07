import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import torch
from torchvision import models
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from fastprogress import progress_bar


class Ethiopic(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.image_filenames = os.listdir(data_dir)
        self.transform = transform
        self.char_to_id_mapping = self._create_char_to_id_mapping(csv_path)
    
    def _create_char_to_id_mapping(self, csv_path):
        df = pd.read_csv(csv_path)
        char_to_id_mapping = {char: i for i, char in enumerate(df['Character'].unique())}
        return char_to_id_mapping
    
    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image = Image.open(os.path.join(self.data_dir, filename))
        if self.transform:
          image = self.transform(image)      
          
        file_id = int(filename.split('.')[0][:3]) 
        label_part = filename.split('.')[0][3:]  
        
        label = file_id - 1  
        
        return image, label
    
    def __len__(self):
        return len(self.image_filenames)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

dataset = Ethiopic(data_dir ="/Users/yeabsiramohammed/classes/neuro140/dataset1",csv_path = "/Users/yeabsiramohammed/classes/neuro140/supported_chars.csv", transform = train_transform)

from torch.utils.data import DataLoader, random_split
total_size = len(dataset)
train_size = int(0.8 * total_size)  
val_size = total_size - train_size  


train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)



# Assuming you have defined train_loader, val_loader, and num_epochs
# Assuming dataset and loaders setup

model_dict = {
    # 'AlexNet': models.alexnet(),
    # 'ResNet': models.resnet18(),
    # 'DenseNet': models.densenet121(),
    'VGG_net': models.vgg16_bn(),
}

results = {
    'model_name': [],
    'epoch': [],
    'train_accuracy': [],
    'val_accuracy': [],
    'train_loss': [],
    'val_loss': [],
}

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")  # Fallback to CPU if necessary
device = 'cpu'

for model_name, model in model_dict.items():
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  

    for epoch in range(20):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_accum = 0

        for inputs, labels in progress_bar(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * inputs.size(0)  # Aggregate the loss
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total
        train_loss = train_loss_accum / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_accum = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss_accum += loss.item() * inputs.size(0) 
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        val_loss = val_loss_accum / val_total

        print(f'Epoch {epoch} - Model: {model_name} Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save results
        results['model_name'].append(model_name)
        results['epoch'].append(epoch)
        results['train_accuracy'].append(train_accuracy)
        results['val_accuracy'].append(val_accuracy)
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)

        torch.save(model.state_dict(), f'{model_name}_epoch_{epoch}.pth')

# Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{model_name}_training_results.csv', index=False)


