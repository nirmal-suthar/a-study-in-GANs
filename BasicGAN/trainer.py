import tqdm
from tensorboardX import SummaryWriter
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils


class GANTrainer(object):
    def __init__(self,device,G,D,dataset,batch_size,epochs,lr,model,images):
        self.G = G
        self.D = D
        self.device = device
        self.G.to(self.device)
        self.D.to(self.device)
        self.dataset = dataset
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=4)
        self.batch_size = batch_size
        self.lr = lr
        self.path = model
        self.images = images
        self.epochs = epochs   
        self.start_epoch = 0
        self.optimG = optim.Adam(self.G.parameters(), self.lr)
        self.optimD = optim.Adam(self.D.parameters(),self.lr)
        self.loss = nn.BCELoss()
        self.target_real_g = torch.ones(self.batch_size,device=self.device)
        self.target_fake_g = torch.zeros(self.batch_size,device=self.device)
        self.target_real = 0.7 * torch.rand(self.batch_size,device=self.device) #See Soumith Chintala ganhacks NIPS2016 : Having soft labels instead of hard helps
        self.target_fake = torch.zeros(self.batch_size,device=self.device)
        self.test_noise = torch.randn(32,64,device=self.device) #While evaluating during training, the noise input for image generation is fixed
        self.losses_g = []
        self.losses_d = []
        self.writer = SummaryWriter()

    def save_model(self, epoch):
        print("Saving Model at '{}'".format(self.path))
        try:
            model = {
                    'epoch': epoch+1,
                    'generator': self.G.state_dict(),
                    'discriminator': self.D.state_dict(),
                    'g_optimizer': self.optimG.state_dict(),
                    'd_optimizer': self.optimD.state_dict()
                    }
            torch.save(model, self.path)
        except:
            print("Unable to load model from '{}'. Training from scratch.".format(self.path))
    
    def load_model(self):
        model = torch.load(self.path)
        self.G.load_state_dict(model['generator'])
        self.D.load_state_dict(model['discriminator'])
        self.optimG.load_state_dict(model['g_optimizer'])
        self.optimD.load_state_dict(model['d_optimizer'])
        self.start_epoch = model['epoch']

    def sample_images(self,epoch):
        with torch.no_grad():
            images = self.G(self.test_noise)
            img = torchvision.utils.make_grid(images)
            self.writer.add_image("Epoch {}".format(epoch),img,epoch)
            torchvision.utils.save_image(images,"%s/epoch%d.png" % (self.images,epoch+1),nrow=8,normalize=True)

    def train(self):
        self.G.train()
        self.D.train()
        for epoch in range(self.start_epoch, self.epochs+1):
            print("Epoch %d of %d" % (epoch,self.epochs))
            running_G_loss = 0.0
            running_D_loss = 0.0
            for i, data in tqdm.tqdm(enumerate(self.loader, 1)):
                images,_ = data
                images = images.to(self.device) 
                #Update weights of the discriminator D(x)
                self.optimD.zero_grad()
                d_real = self.D(images)
                p = torch.rand(1).item() >= 0.1
                loss_real = 0;
                if p is True:
                    loss_real = self.loss(d_real,self.target_real) #Gradient descent on -log(D(x))
                else:
                    loss_real = self.loss(d_real,self.target_fake) #Gradient descent on -log(D(x))
                loss_real.backward()
                noise = torch.randn(self.batch_size,64,device=self.device) 
                fake = self.G(noise) #G(z) for training the discriminator
                d_fake = self.D(fake.detach()) #Since we are just training the discriminator, computing gradients wrt G(z) has no point hence we can treat G(Z) as fixed input
                loss_fake = 0
                if p is True:
                    loss_fake = self.loss(d_fake,self.target_fake) # Gradient descent on -log(1-D(G(z)) 
                else:
                    loss_fake = self.loss(d_fake,self.target_real) # Gradient descent on -log(1-D(G(z)) 
                loss_fake.backward()
                self.optimD.step()
                #Update weights of the generator G(z)
                self.optimG.zero_grad()
                noise_g = torch.randn(self.batch_size,64,device=self.device)
                fake_g = self.G(noise_g)
                d_fake_g = self.D(fake_g) #Here we actually update the weights of G(z) hence we cannot treat G(z) as a fixed input and detach it from computational graph
                loss_fake_g = self.loss(d_fake_g,self.target_real_g) # Gradient descent on log(1 - D(G(z)) is the same as gradient descent on -log(D(G(z))
                loss_fake_g.backward()
                self.optimG.step()
                #Log the generator and the discriminator losses 
                self.writer.add_scalar('Generator Loss',loss_fake_g.item(),i*(epoch+1))
                self.writer.add_scalar('Discriminator Loss',loss_real.item() + loss_fake.item(),i*(epoch+1))
                running_G_loss += loss_fake_g.item()
                running_D_loss += loss_real.item() + loss_fake.item()
                if i%100==0:
                    print("Generator Loss : {} Discriminator Loss : {}".format(running_G_loss/i,running_D_loss/i))
                    self.losses_g.append(running_G_loss)
                    self.losses_d.append(running_D_loss)
                    running_D_loss = running_G_loss = 0.0
            self.save_model(epoch)
            self.G.eval()
            self.D.eval()
            print("Sampling and saving images")
            self.sample_images(epoch)
            self.G.train()
            self.D.train()
