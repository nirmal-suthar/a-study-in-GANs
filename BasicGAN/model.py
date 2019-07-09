import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

#Generate an image from a 128*1 vector sampled from the noise prior


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024,2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fc6 = nn.Linear(2048,28*28)

        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.xavier_normal_(self.fc4.weight,gain=torch.nn.init.calculate_gain('tanh')) #c.f. Bengio and Glorot for Sigmoid

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)),0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)),0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)),0.2)
        x = F.leaky_relu(self.bn4(self.fc4(x)),0.2)
        x = F.leaky_relu(self.bn5(self.fc5(x)),0.2)
        x = F.tanh(self.fc6(x))#Chintala NIPS2016 : Normalize inputs in -1 to 1 and then use tanh layer in generator 
        x = x.reshape(-1,28,28)
        return x.unsqueeze(1)

##Instead of using a scalar for classification: i.e. 0 for fake and 1 for real, most popular
#implementations use a vector of a given size n where [0]n is for fake and [1]n is for real
#We use n=128


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28,256)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128,32)
        self.bn3 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(32,1)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.xavier_normal_(self.fc4.weight,torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = x.reshape(-1,28*28)
        x = self.drop1(F.leaky_relu(self.fc1(x),0.2))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)),0.2))
        x = self.drop3(F.leaky_relu(self.bn3(self.fc3(x)),0.2))
        x = F.sigmoid(self.fc4(x))
        return torch.squeeze(x,1)
