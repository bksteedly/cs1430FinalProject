import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from learning3d.losses import ClassificationLoss
from learning3d.models import PointNet, Classifier
import os
import numpy as np
import json

def load_data(load_train=True):
    with open('../data/modelnet40/encoding_metadata.json', 'r') as f:
        encoding_metadata = json.load(f)
    
    if load_train:
        data_dir = '../data/modelnet40/train/'
        np_files = os.listdir(data_dir)
        train_data = []
        for file in np_files:
            pointcloud = np.load(data_dir+file)
            label = file.split('.')[0]
            train_data.append({'pointcloud': pointcloud, 'label': np.array(encoding_metadata[label])})
        return encoding_metadata, train_data
    else:
        data_dir = '../data/modelnet40/test/'
        np_files = os.listdir(data_dir)
        test_data = []
        for file in tqdm(np_files, desc='Loading model net 40 numpy files'):
            pointcloud = np.load(data_dir+file)
            label = file.split('.')[0]
            test_data.append({'pointcloud': pointcloud, 'label': np.array(encoding_metadata[label])})
        return encoding_metadata, test_data
    
        
class Data(Dataset):
    def __init__(self, data):
        self.pcs = []
        self.labels = []

        for c in data:
            l = c['label']
            for ex in c['pointcloud']:
                self.pcs.append(ex)
                self.labels.append(l)

        self.pcs = np.array(self.pcs)
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.pcs)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.pcs[index]), torch.from_numpy(self.labels[index])
    

def train():
    pnet = PointNet(global_feat=True)
    model = Classifier(feature_model=pnet)

    _, train_data = load_data()
    trainset = Data(train_data)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, drop_last=True)

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(learnable_params)

    loss_fn = ClassificationLoss()

    epochs = 300
    for i in range(epochs):
        total_loss = 0
        for j, data in enumerate(tqdm(trainloader, desc=f"Epoch {i+1}/{epochs}")):
            points, target = data
            target = target.squeeze(-1)
            output = model(points.float())
            target_indices = target.argmax(dim=1)
            loss = loss_fn(output, target_indices)

            correct = (target_indices == output.argmax(dim=1)).sum().item()
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Accuracy: " + str(correct/len(points)))
        print("Loss: " + str(total_loss/len(points)))

        torch.save(model.state_dict(), f"pointnet_classification_model_{i}.pth")

def test():
    pnet = PointNet(global_feat=True)
    model = Classifier(feature_model=pnet)
    model.load_state_dict(torch.load('pointnet_classification_model_66.pth', map_location=torch.device('cpu')))
    model.eval()

    _, test_data = load_data(load_train=False)
    testset = Data(test_data)
    testloader = DataLoader(testset, batch_size=32, shuffle=True, drop_last=True)

    accuracy = 0
    total = 0
    for j, data in enumerate(tqdm(testloader)):
        points, target = data
        target = target.squeeze(-1)
        output = model(points.float())
        target_indices = target.argmax(dim=1)
        predictions = output.argmax(dim=1)

        accuracy += (target_indices == predictions).sum().item()
        total += len(points)

    print('Accuracy:', round(accuracy/total, 4))

def classify(verts):
    pnet = PointNet(global_feat=True)
    model = Classifier(feature_model=pnet)
    model.load_state_dict(torch.load('pointnet_classification_model_66.pth', map_location=torch.device('cpu')))
    model.eval()

    d, _ = load_data()

    data = [{'pointcloud': np.array([verts]), 'label': [np.zeros((1,40))]}]
    testloader = DataLoader(Data(data), batch_size=1, shuffle=True, drop_last=True)

    for i in range(1):
        for j, data in enumerate(testloader):
            points, target = data
            target = target.squeeze(-1)
            output = model(points.float())
            return list(d.keys())[output.argmax(dim=1)]



