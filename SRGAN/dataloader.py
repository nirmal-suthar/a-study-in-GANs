#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T


# In[2]:


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


# In[3]:


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


# In[16]:


def train_HR_transform(crop_size):
    return T.Compose([ T.RandomCrop(crop_size, pad_if_needed=True), T.ToTensor() ])
    
        


# In[17]:


def train_LR_transform(crop_size, upscale_factor):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(crop_size // upscale_factor, interpolation = Image.BICUBIC),
        T.ToTensor(),
    ])
    
    return transform

def display_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize(400),
        T.CenterCrop(400),
        T.ToTensor()
    ])


# In[22]:


class dataset_train_from_folder(Dataset):
    
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        
        super().__init__()
        
        self.image_filename = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.HR_transform = train_HR_transform(crop_size) 
        self.LR_transform = train_LR_transform(crop_size, upscale_factor)
        
    
    def __getitem__(self, idx):
        
        HR_image = self.HR_transform(Image.open(self.image_filename[idx]))
        LR_image = self.LR_transform(HR_image)
        
        return HR_image, LR_image
    
    def __len__(self):
        return len(self.image_filename)



class dataset_val_from_folder(Dataset):
    
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        
        super().__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        
        self.upscale_factor = upscale_factor
        
    def __getitem__(self, idx):
        
        image = Image.open(self.image_filenames[idx])
        w, h = image.size
        crop_size = calculate_valid_crop_size(min(w,h) , self.upscale_factor)
        LR_scale = T.Resize(crop_size // self.upscale_factor, interpolation = Image.BICUBIC)
        HR_scale = T.Resize(crop_size, interpolation = Image.BICUBIC)
        HR_image = T.CenterCrop(crop_size)(image) 
        LR_image = LR_scale(HR_image)
        HR_restore_image = HR_scale(LR_image)
        
        return T.ToTensor()(LR_image), T.ToTensor()(HR_restore_image), T.ToTensor()(HR_image)
    
    def __len__(self):
        return len(self.image_filenames)
        
    
    
    
# In[ ]:


# class ValDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(ValDatasetFromFolder, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

#     def __getitem__(self, index):
#         hr_image = Image.open(self.image_filenames[index])
#         w, h = hr_image.size
#         crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
#         lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
#         hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
#         hr_image = CenterCrop(crop_size)(hr_image)
#         lr_image = lr_scale(hr_image)
#         hr_restore_img = hr_scale(lr_image)
#         return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

#     def __len__(self):
#         return len(self.image_filenames)


# class TestDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(TestDatasetFromFolder, self).__init__()
#         self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
#         self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
#         self.upscale_factor = upscale_factor
#         self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
#         self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

#     def __getitem__(self, index):
#         image_name = self.lr_filenames[index].split('/')[-1]
#         lr_image = Image.open(self.lr_filenames[index])
#         w, h = lr_image.size
#         hr_image = Image.open(self.hr_filenames[index])
#         hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
#         hr_restore_img = hr_scale(lr_image)
#         return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

#     def __len__(self):
#         return len(self.lr_filenames)

