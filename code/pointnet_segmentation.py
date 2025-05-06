import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from learning3d.losses import ClassificationLoss
from learning3d.models import PointNet, Segmentation, Classifier

import os
from torchvision import transforms
import numpy as np
import random
import json

def read_off_file(file_path):
    with open(file_path, 'r') as file:
        off_header = file.readline().strip()
        if 'OFF' == off_header:
            n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        else:
            n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        
        pointcloud = default_transforms()((verts, faces))
        return pointcloud
    
def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize()
                              ])

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return sampled_points

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud

def load_data(load_from_raw=False, load_train=True):
    with open('../data/modelnet40/encoding_metadata.json', 'r') as f:
        encoding_metadata = json.load(f)
    # encoding_metadata = {}

    if load_train:
        if load_from_raw:
            data_dir = '/Users/amulya/Desktop/modelnet40-princeton-3d-object-dataset/ModelNet40'
            classes = os.listdir(data_dir)
            train_data = []
            for c in classes:
                np_file = []
                print(f'Working on class {c} ...')
                
                for file in os.listdir(data_dir+'/'+c+'/train'):
                    pointcloud = read_off_file(data_dir+'/'+c+'/train/'+file)
                    np_file.append(pointcloud)
                    train_data.append({'pointcloud': pointcloud, 'label': c})
                
                np.save(f'../data/modelnet40/train/{c}.npy', np.array(np_file))
                print('Saved training data')
            return train_data
        else:
            data_dir = '../data/modelnet40/train/'
            np_files = os.listdir(data_dir)
            train_data = []
            for file in tqdm(np_files, desc='Loading model net 40 numpy files'):
                pointcloud = np.load(data_dir+file)
                label = file.split('.')[0]
                train_data.append({'pointcloud': pointcloud, 'label': np.array(encoding_metadata[label])})
            return train_data
    else:
        if load_from_raw:
            data_dir = '/Users/amulya/Desktop/modelnet40-princeton-3d-object-dataset/ModelNet40'
            classes = os.listdir(data_dir)
            test_data = []
            for c in classes:
                np_file = []
                print(f'Working on class {c} ...')
                
                for file in os.listdir(data_dir+'/'+c+'/test'):
                    pointcloud = read_off_file(data_dir+'/'+c+'/test/'+file)
                    np_file.append(pointcloud)
                    test_data.append({'pointcloud': pointcloud, 'label': c})
                
                np.save(f'../data/modelnet40/test/{c}.npy', np.array(np_file))
                print('Saved testing data')
            return test_data
        else:
            data_dir = '../data/modelnet40/test/'
            np_files = os.listdir(data_dir)
            test_data = []
            for file in tqdm(np_files, desc='Loading model net 40 numpy files'):
                pointcloud = np.load(data_dir+file)
                label = file.split('.')[0]
                test_data.append({'pointcloud': pointcloud, 'label': np.array(encoding_metadata[label])})
            return test_data
    
        

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
    # TODO: training this as a classifier for now because that is how are labels are
    pnet = PointNet(global_feat=True)
    model = Classifier(feature_model=pnet)

    train_data = load_data()
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

    torch.save(model.state_dict(), "pointnet_segmentation_model.pth")

def test():
    pnet = PointNet(global_feat=True)
    model = Classifier(feature_model=pnet)
    model.load_state_dict(torch.load('pointnet_segmentation_model.pth', map_location=torch.device('cpu')))
    model.eval()

    test_data = load_data(load_train=False)
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


if __name__ == "__main__":
    train()
    # load_train_data(load_from_raw=True)

    # model = torch.load('/Users/amulya/Desktop/learning3d/pretrained/exp_classifier/models/best_ptnet_model.t7', map_location=torch.device('cpu'))
    # print(model)
    # model.eval()
    # test_data = load_data(load_train=False)
    # print(type(test_data))
    # print(type(test_data[0]))
    # # testset = Data(test_data)
    # testloader = DataLoader(testset, batch_size=1, shuffle=True, drop_last=True)
    
    # # ex, label = next(iter(testloader))
    # # print("label:", label)
    # # print(type(model))
    # # output = model(ex.float())
    # # print(output)
    
    # pnet = PointNet(global_feat=True)
    # model = Classifier(feature_model=pnet)
    # model.load_state_dict(torch.load('pointnet_segmentation_model.pth', map_location=torch.device('cpu')))
    # model.eval()

    # for i in range(1):
    #     for j, data in enumerate(tqdm(testloader)):
    #         points, target = data
    #         target = target.squeeze(-1)
    #         output = model(points.float())
    #         target_indices = target.argmax(dim=1)
    #         print(f'target: {target_indices} and output: {output.argmax(dim=1)}')
            




