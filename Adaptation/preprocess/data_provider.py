import numpy as np
from .data_list import ImageList
import torch.utils.data as util_data
from torchvision import transforms
from PIL import Image, ImageOps


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

def load_images(images_file_path, batch_size, resize_size=256, is_train=True, crop_size=224, is_cen=False, split_noisy=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if not is_train:
        start_center = (resize_size - crop_size - 1) / 2
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])
        if "Office-31" in images_file_path:
            images_file_path=images_file_path.replace('Office-31','office_list')
        if "Office-Home" in images_file_path:
            images_file_path=images_file_path.replace('Office-Home','officehome_list')
        domain=images_file_path.split('list/')[1].split('.')[0]
        images_file_path=images_file_path.replace(domain,domain+'_list')
        images = ImageList(open(images_file_path+'').readlines(), transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=2)
        return images_loader
    else:
        if is_cen:
            transformer = transforms.Compose([ResizeImage(resize_size),
                transforms.Scale(resize_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize])
        else:
            transformer = transforms.Compose([ResizeImage(resize_size),
                  transforms.RandomResizedCrop(crop_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize])
        if split_noisy:
            if "noisy_" in images_file_path:
                images_file_path=images_file_path.replace('noisy_','')
            if "Office-31" in images_file_path:
                images_file_path=images_file_path.replace('Office-31','office_list')
            if "Office-Home" in images_file_path:
                images_file_path=images_file_path.replace('Office-Home','officehome_list')
            if "feature_uniform" in images_file_path:
                images_file_path=images_file_path.replace('feature_uniform','noisycorrupted')
            if "feature" in images_file_path:
                images_file_path=images_file_path.replace('feature','corrupted')
            if "uniform" in images_file_path:
                images_file_path=images_file_path.replace('uniform','noisy')
            domain=images_file_path.split('list/')[1].split('_')[0]
            images_file_path=images_file_path.replace(domain,domain+'_list')
 
            clean_images = ImageList(open(images_file_path.split('.t')[0]+'_Relabel_0.8.txt').readlines(), transform=transformer)
            noisy_images = ImageList(open(images_file_path.split('.t')[0]+'_left_0.8.txt').readlines(), transform=transformer)
            clean_loader = util_data.DataLoader(clean_images, batch_size=batch_size, shuffle=True, num_workers=2)
            noisy_loader = util_data.DataLoader(noisy_images, batch_size=int(batch_size), shuffle=True, num_workers=2)
            return clean_loader, noisy_loader
        else:
            if "Office-31" in images_file_path:
                images_file_path=images_file_path.replace('Office-31','office_list')
            if "Office-Home" in images_file_path:
                images_file_path=images_file_path.replace('Office-Home','officehome_list')
            domain=images_file_path.split('list/')[1].split('.')[0]
            images_file_path=images_file_path.replace(domain,domain+'_list')
            images = ImageList(open(images_file_path+'').readlines(), transform=transformer)
            images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last)
            return images_loader

