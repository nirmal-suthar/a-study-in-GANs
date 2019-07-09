#!/usr/bin/env python
# coding: utf-8

# In[16]:


import argparse
import os
from math import log10

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import vgg19
from tqdm import tqdm
import pytorch_ssim

from dataloader import*
from fastprogress import master_bar, progress_bar

# import pytorch_ssim
# from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform


# In[17]:


class ResnetBlock(nn.Module):
    
    def __init__(self, channels=64):
        
        super().__init__()
        
        self.identity = nn.Sequential(
            
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

        
    def forward(self, x):
        
        identity = self.identity(x)
        x = identity + x
        return x
    


# In[18]:


class UpSampleBlock(nn.Module):
    
    def __init__(self, up_scale = 2, channels = 64):
        
        super().__init__()
        
        self.model = nn.Sequential(
                
          
        
            nn.Conv2d(channels, channels*(up_scale**2), kernel_size=3, padding=1),
            nn.PixelShuffle(up_scale),    
          
        
            nn.PReLU()
        )
        
        
    def forward(self, x):
        
        x = self.model(x)
        return x


# In[19]:


class generator(nn.Module):
    
    def __init__(self, B_numResnetBlock = 4, in_channels = 3, step_channels = 64):
        
        super().__init__()
        
        self.init_model = nn.Sequential(
            nn.Conv2d(in_channels, step_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        mid_model = []
        
        for i in range(B_numResnetBlock):
            mid_model.append(ResnetBlock(step_channels))
            
        mid_model.append(
            nn.Sequential(
                nn.Conv2d(step_channels, step_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(step_channels)
            )
        )
        
        self.mid_model = nn.Sequential(*mid_model)
        
        end_model = []
        
        self.r = 2
        for i in range(self.r):
            end_model.append(UpSampleBlock(up_scale = 2, channels = step_channels))
        
        end_model.append(
            nn.Sequential(
                nn.Conv2d(step_channels, in_channels, kernel_size=9, padding=4)
            )
        )
        
        self.end_model = nn.Sequential(*end_model)
        
        self._weight_initializer()
    
    
    def forward(self, x):
        
        x = self.init_model(x)
        skip_connection = self.mid_model(x)
        x = skip_connection + x
        x = self.end_model(x)
        
        return x
            
            
            
    def _weight_initializer(self):
        r"""Default weight initializer for all generator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        


# In[20]:


class discriminator(nn.Module):
    
    def __init__(self, in_channels = 3, step_channels = 64):
        
        super().__init__()
        
        model = []
        
        model.append(
            nn.Sequential(
                nn.Conv2d(in_channels, step_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(.2),
                nn.Conv2d(step_channels, step_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(step_channels),
                nn.LeakyReLU(.2)
            )
        )
        
        self.expansion = step_channels
        
        for i in range(3):
            
            model.append(
                nn.Sequential(
                    nn.Conv2d(self.expansion, self.expansion*2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.expansion*2),
                    nn.LeakyReLU(.2),
                    nn.Conv2d(self.expansion*2, self.expansion*2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(self.expansion*2),
                    nn.LeakyReLU(.2)    
          
        
                )
            )
            
            self.expansion = self.expansion*2
            
        
        model.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.expansion, self.expansion*2, kernel_size=1),
                nn.LeakyReLU(.2),
                nn.Conv2d(self.expansion*2, 1, kernel_size=1),
                nn.Sigmoid()
            )
        )
        
        
        self.model = nn.Sequential(*model)
        
        self._weight_initializer()
        

    
    def forward(self, x):
        
        x = self.model(x)
        return x.view(-1)
        
        
        
    def _weight_initializer(self):
        r"""Default weight initializer for all generator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        


# In[21]:


class discriminatorLoss(nn.Module):
    
    def __init__(self, generator, discriminator, device):
        
        super().__init__()
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.bceloss = nn.BCELoss().to(device)
        
    def forward(self, LR_image, HR_image):
        
        HR_pred = self.discriminator(HR_image)
        SR_image = self.generator(LR_image)
        SR_pred = self.discriminator(SR_image)
        real_ = torch.ones(HR_pred.shape).to(self.device)
        fake_ = torch.zeros(SR_pred.shape).to(self.device)
        
        HR_loss = self.bceloss(HR_pred, real_)
        SR_loss = self.bceloss(SR_pred, fake_)
        loss = HR_loss + SR_loss
        
        return loss
        


# In[22]:


class generatorLoss(nn.Module):
    
    def __init__(self, generator, discriminator, device):
        
        super().__init__()
        
        self.generator = generator
        self.discriminator = discriminator
        
        vgg = vgg19(pretrained=True, progress=True)
        
        vgg_loss = nn.Sequential(*(list(vgg.features)[:9])).eval()
        for param in vgg_loss.parameters():
            param.requires_grad = False
        
        self.vgg_features = vgg_loss.to(device)
        self.mseloss = nn.MSELoss.to(device)
        self.bceloss = nn.BCELoss.to(device)
        
        
    def forward(self, LR_image, HR_image):
        
        SR_image = self.generator(LR_image)
        SR_pred = self.discriminator(SR_image)
        real_ = torch.ones(SR_pred.shape).to(self.device)
        
        adversial_loss = self.bceloss(SR_pred, real_)
        perceptual_loss = self.mseloss(self.vgg_features(HR_image), self.vgg_features(SR_image))
        content_loss = self.mseloss(HR_image, SR_image)
        
        return content_loss + 0.001*adversial_loss + 0.006*perceptual_loss
    


# In[23]:


class parser():
    
    def __init__(self):
        
        self.crop_size = 88
        self.upscale_factor = 4 
        self.num_epochs = 200


# In[24]:


opt = parser()
UPSCALE_FACTOR =4


# In[25]:


path = '/data/nirmalps/VOC2012/VOC2012/JPEGImages/'
valpath = '/data/nirmalps/VOC2012/VOC2012/Val/'


# In[26]:


train_set = dataset_train_from_folder(path, crop_size=opt.crop_size, upscale_factor=opt.upscale_factor)
val_set = dataset_val_from_folder(valpath, upscale_factor=UPSCALE_FACTOR, crop_size=opt.crop_size)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)


# In[27]:


# if torch.cuda.is_available():
#     device = ['cuda:0']
#     # Use deterministic cudnn algorithms
#     torch.backends.cudnn.deterministic = True
#     epochs = 100
# else:
#     device = ['cpu']
#     epochs = 5

# print("Device: {}".format(device))
# print("Epochs: {}".format(epochs))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(device)




# In[28]:


generator = generator().to(device)
discriminator = discriminator().to(device)

# generator.load_state_dict(torch.load('./generator_multi'))



# In[29]:
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    generator = nn.DataParallel(generator)
    discriminator  = nn.DataParallel(discriminator)

# generatorLoss = generatorLoss(generator, discriminator, device)
# discriminatorLoss = discriminatorLoss(generator, discriminator, device)


# In[30]:




optimizerG = optim.Adam(generator.parameters(), lr = 0.001)
optimizerD = optim.Adam(discriminator.parameters(), lr = 0.001)

schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, 15, gamma=0.2, last_epoch=-1)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, 15, gamma=0.2, last_epoch=-1)


# In[31]:



vgg = vgg19(pretrained=True, progress=True)
        
vgg_loss = nn.Sequential(*(list(vgg.features)[:9])).eval()
for param in vgg_loss.parameters():
    param.requires_grad = False
        
vgg_features = vgg_loss.to(device)
mseloss = nn.MSELoss().cuda()
bceloss = nn.BCELoss().cuda()


# In[32]:


print(len(train_loader))
UPSCALE_FACTOR = 4


# checkpoint = torch.load('/data/nirmalps/SRGAN/training_results/pretrainedResnet/epochs/netG_epoch_4_50.pth')
# try:
#   generator = generator.load_state_dict(checkpoint)
# except:
#   print("******generator warning*******")
# # generator = generator.load_state_dict(checkpoint)


# In[48]:



results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
mb = master_bar(range(1, opt.num_epochs+1))
print('training started')
for epoch in mb:
    
    
    generator.train()
    discriminator.train()

    schedulerG.step()
    schedulerD.step()  
    
    running_results = {'batch_sizes': 1, 'd_loss': 0, 'g_loss': 0}
    for i, data in zip(progress_bar(range(len(train_loader)), parent=mb), train_loader):
        
        HR_image, LR_image = data
        HR_image = HR_image.to(device)
        LR_image = LR_image.to(device)
        
        m = HR_image.size(0)
        batch_size = m
        running_results['batch_sizes'] += batch_size
        
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        
        discriminator.zero_grad()
        
        
        HR_pred = discriminator(HR_image)
        SR_image = generator(LR_image)
        SR_pred = discriminator(SR_image)
        real_ = torch.ones(HR_pred.shape).to(device)
        fake_ = torch.zeros(SR_pred.shape).to(device)
        
        HR_loss = bceloss(HR_pred, real_)
        SR_loss = bceloss(SR_pred, fake_)
        d_loss = HR_loss + SR_loss
        running_results['d_loss'] += d_loss.item() * batch_size
        
#         d_loss = discriminatorLoss(LR, HR)
        d_loss.backward(retain_graph = True)
        optimizerD.step()
        
        ############################
        # (2) Update G network: minimize adversial loss + Perception Loss + content Loss
        ###########################
        
        generator.zero_grad()
        
        SR_pred = discriminator(SR_image)
        real_ = torch.ones(SR_pred.shape).to(device)
        
        adversial_loss = bceloss(SR_pred, real_)
        perceptual_loss = mseloss(vgg_features(HR_image), vgg_features(SR_image))
        content_loss = mseloss(HR_image, SR_image)
        
        g_loss = content_loss + 0.001*adversial_loss + 0.003*perceptual_loss
        
        running_results['g_loss'] += g_loss.item() * batch_size
        
#         g_loss = generatorLoss(LR, HR)
        g_loss.backward()
        optimizerG.step()
          
    ############################
    # (3) post epoch Summary
    ###########################
    
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
                  epoch, opt.num_epochs, running_results['d_loss'] / running_results['batch_sizes'],
                  running_results['g_loss'] / running_results['batch_sizes']))
    
    generator.eval()
    
    
    #directory for result
    out_path = 'training_results/images/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    #validation starts
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    
    for val_lr, val_hr_restore, val_hr in val_bar:
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size
        with torch.no_grad():
            LR = Variable(val_lr)
            HR = Variable(val_hr)
        if torch.cuda.is_available():
            LR = LR.to(device)
            HR = HR.to(device)
        
        SR = generator(LR)

        batch_mse = ((SR - HR) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(SR, HR).item()
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        val_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))

        val_images.extend(
            [display_transform()(val_hr_restore.squeeze(0)), display_transform()(HR.data.cpu().squeeze(0)),
             display_transform()(SR.data.cpu().squeeze(0))])
    
    #save val image (sr, lr, hr)
    val_images = torch.stack(val_images)
    val_images = torch.chunk(val_images, val_images.size(0) // 15)
    val_save_bar = tqdm(val_images, desc='[saving training results]')
    index = 1
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
        index += 1

    # save model parameters
    out_path = 'training_results/epochs/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if epoch %5 ==0 and epoch !=0:
        torch.save(generator.state_dict(), 'training_results/epochs/generator_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(discriminator.state_dict(), 'training_results/epochs/discriminator_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    
    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    # results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    # results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    #displaying statistics
    
    out_path = 'training_results/statistics/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if epoch % 10 == 0 and epoch != 0:
        
        out_path = 'statistics/'
              
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'],
                  'PSNR': results['psnr'], 'SSIM': results['ssim']}, index=range(1, epoch + 1))

data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
        
 
# In[45]:


def load_checkpoint(generator,discriminator, optimizerD, optimizerG, filename=resume):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['g_state_dict'])
        optimizerG.load_state_dict(checkpoint['g_optimizer'])
        discriminator.load_state_dict(checkpoint['d_state_dict'])
        optimizerD.load_state_dict(checkpoint['d_optimizer'])
        
#         losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return generator,discriminator, optimizerD, optimizerG

