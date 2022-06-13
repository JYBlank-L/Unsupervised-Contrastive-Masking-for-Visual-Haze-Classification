# --- Imports --- #
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor
from UCM.ucm import ucm


# --- Haze-Wild dataset --- #
class Haze_Wild(data.Dataset):

    def __init__(self, images_path, labels, crop_size, filters, method=0, mode='train'):

        if mode == 'train':
            self.image_path = images_path
            print("Total training examples:", len(self.image_path))
        elif mode == 'val':
            self.image_path = images_path
            print("Total validation examples:", len(self.image_path))
        else:
            self.image_path = images_path
            print("Total test examples:", len(self.image_path))

        self.labels = labels
        self.crop_size = crop_size
        self.method = method
        self.filters = filters

    def get_images(self, index):
        method = self.method
        seg_img, org_img = ucm(self.image_path[index], self.crop_size, self.filters, method)
        label = self.labels[index]

        transform = Compose([ToTensor()])
        seg = transform(seg_img)
        org = transform(org_img)
        if method == 0:
            return org, label
        else:
            return torch.cat((seg, org)), label

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.image_path)
