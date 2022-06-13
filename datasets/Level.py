import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor
from UCM.ucm import ucm
import ipdb


def loadData(img_dir, image_dir):
    image_list = []
    label_list = []
    with open(img_dir, "r") as f:
        label_data = f.readlines()
        for line in label_data:
            line = line.strip('\n')
            image = image_dir + line.split(' ')[0]
            label = line.split(' ')[1]
            image_list.append(image)
            label_list.append(int(label))
    return image_list, label_list


def ucmImage(image_path, labels, crop_size, filters, method, category):
    image_list = []
    label_list = []
    for index in range(len(image_path)):
        seg_img, org_img = ucm(image_path[index], crop_size, filters, method)
        image = np.concatenate((seg_img, org_img), axis=2)
        image_list.append(image)
        label_list.append(labels)
    np.save('./data/hazel-level/' + category + '/image.npy', np.array(image_list))
    np.save('./data/hazel-level/' + category + '/label.npy', np.array(label_list))


class Haze_Level(data.Dataset):
    def __init__(self, data_path, crop_size, filters, method=0, data_name='original_64', mode='train'):
        super().__init__()
        self.crop_size = crop_size
        self.filters = filters
        self.method = method
        data_dir = data_path + 'hazel-level'
        img_dir = data_dir + '/' + mode + '/' + data_name + '/image.npy'
        label_dir = data_dir + '/' + mode + '/' + data_name + '/label.npy'
        self.image_path, self.labels = np.load(img_dir), np.load(label_dir)

    def __getitem__(self, index):
        method = self.method
        # seg_img, org_img = ucm(self.image_path[index], self.crop_size, self.filters, method)
        image = self.image_path[index]
        label = self.labels[index]

        transform = Compose([ToTensor()])
        image = transform(image)
        return image, label
        # seg = transform(seg_img)
        # org = transform(org_img)
        # if method == 0:
        #     return org, label
        # else:
        #     return torch.cat((seg, org)), label

    def __len__(self):
        return len(self.image_path)
