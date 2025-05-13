import os
from torchvision import transforms
import numpy as np
import random

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

def load_data(load_train=True):
    if load_train:
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
    

if __name__ == '__main__':
    load_data()