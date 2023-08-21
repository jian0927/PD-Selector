import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from PD_Selector import build_samples_classes,PDselector
# 定义 device
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#device = torch.device("cpu")


class Encoder_fig(nn.Module):
    def __init__(self,n_dim=10):
        super(Encoder_fig, self).__init__()
        self.n_dim = n_dim
        self.dropout1 = torch.nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.encode_figure = nn.Linear(7 * 7 * 32, n_dim)
        
    def forward(self, x_fig):
        model_device = next(self.parameters()).device
        x_fig = x_fig.to(model_device)
        y_fig = self.relu(self.conv1(x_fig))
        y_fig = self.maxpool(y_fig)
        y_fig = self.relu(self.conv2(y_fig))
        y_fig = self.maxpool(y_fig)
        y_fig = y_fig.view(y_fig.size(0), -1)
        y_fig = self.encode_figure(y_fig)  
        
        y_fig = y_fig/(torch.norm(y_fig,p=2,dim=-1,keepdim=True)+1e-8)  # (batch, n_dim)

        return y_fig


class Decoder_fig(nn.Module):
    def __init__(self, n_dim=10):
        super(Decoder_fig, self).__init__()
        self.n_dim = n_dim
        
        self.decode_figure = nn.Linear(n_dim, 7 * 7 * 32)
        
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(16, 8, kernel_size=5, padding=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(8, 1, kernel_size=5, padding=2)
        
    def forward(self, y_fig):
        model_device = next(self.parameters()).device
        y_fig = y_fig.to(model_device)
        y_fig = self.decode_figure(y_fig)
        y_fig = y_fig.view(y_fig.size(0), 32, 7, 7)
        
        y_fig = self.relu(self.conv1(y_fig))
        y_fig = self.upsample1(y_fig)
        y_fig = self.relu(self.conv2(y_fig))
        y_fig = self.upsample2(y_fig)
        y_fig = self.conv3(y_fig)
        
        return y_fig

    def loss_reconstruction(self, z_fig, x_fig):
        model_device = next(self.parameters()).device
        criterion = nn.MSELoss()
        loss = criterion(z_fig.to(model_device), x_fig.to(model_device))
        return loss
        
        

class Autoencoder_fig(nn.Module):
    def __init__(self, n_dim=10):
        super(Autoencoder_fig, self).__init__()
        self.n_dim = n_dim
        
        self.encoder = Encoder_fig(n_dim)
        self.selector = PDselector(n_dim = n_dim,n_preference = (20,20),end_residual_rate = 0.1)
        self.decoder = Decoder_fig(n_dim)
        
    def forward(self, x_fig):
        model_device = next(self.parameters()).device
        x_fig = x_fig.to(model_device)
        encoded = self.encoder(x_fig)
        w_selected = self.selector(encoded.unsqueeze(0))
        decoded = self.decoder(encoded)
        return decoded, w_selected



