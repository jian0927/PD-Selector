import torch
import torch.nn as nn
from torch.nn import functional as F
from PD_Selector import PDselector


EPS = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.n_fft = 512
        self.n_feature = int(self.n_fft/2)+1
        self.n_hidden = 512
        self.out_channels = 32
        self.n_dim = int(self.out_channels*4) # 需要乘上一个正整数
        
        
        # pre-coding
        self.pre_encoder = nn.Sequential(
            nn.Linear(self.n_feature,self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden,self.n_hidden),
            nn.ReLU(inplace=True),
            
        )
        
        # vocal tract coding
        self.vocal_lstm = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        # speaker trait encode
        self.trait_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n_dim)
        )
        # Standard phoneme coding
        self.phoneme_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n_dim)
        )
        
        # glottis coding
        self.glottis_lstm = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        self.glottis_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.LayerNorm(self.n_dim),
            nn.Sigmoid(),
        )
        
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.n_dim, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_feature),
        )
        
        self.selector = PDselector(n_dim = self.n_dim,n_preference = (20,20),end_residual_rate = 0.1)
        
        
        self.onehot_encoder = nn.Sequential(
            nn.Linear(self.n_dim,self.n_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_dim,10),
            
            )
        
    def forward(self, x):
        batch_size, n_sample, feature, time_step = x.size()
        model_device = next(self.parameters()).device
        x_in = (torch.reshape(x,[batch_size*n_sample,feature, time_step])).clone().to(model_device)
        x_in = x_in.transpose(1,2) #编排好输入模型的格式 (batch*n_sample,time,feature)
        
        y = self.pre_encoder(x_in)
        
        y_phoneme,y_trait = self.vocal_lstm(y)
        y_trait   = self.trait_encoder(y_trait[1])
        y_phoneme = self.phoneme_encoder(y_phoneme)
        
        y_glottis,_ = self.glottis_lstm(y)
        y_glottis   = self.glottis_encoder(y_glottis)
        
        z_speaker = y_glottis*y_trait.transpose(0,1)
        z_vocal   = y_glottis*y_phoneme
        
        x_out = 0
        x_out = self.decoder(z_speaker+z_vocal).transpose(1,2)
        x_out = torch.reshape(x_out,[batch_size, n_sample, feature, time_step]) # (batch,n_sample,feature,time)
        
        z_speaker_R1 = torch.reshape(torch.mean(z_speaker,dim=1),[batch_size*n_sample,-1]).unsqueeze(0) # (1,batch*n_sample,n_dim)
        z_speaker_R1 = z_speaker_R1/(torch.norm(z_speaker_R1,2,dim=-1)+EPS).unsqueeze(-1)
        w = self.selector(z_speaker_R1)
        #w = self.onehot_encoder(z_speaker_R1)
        #w = F.softmax(w,dim=-1)
        return x_out,w
    
        

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.n_fft = 512
        self.n_feature = int(self.n_fft/2)+1
        self.n_hidden = 512
        self.out_channels = 32
        self.n_dim = int(self.out_channels*4) # 需要乘上一个正整数
        
        
        # pre-coding
        self.pre_encoder = nn.Sequential(
            nn.Linear(self.n_feature,self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden,self.n_hidden),
            nn.ReLU(inplace=True),
            
        )
        
        # vocal tract coding
        self.vocal_lstm = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        # speaker trait encode
        self.trait_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n_dim)
        )
        
        # glottis coding
        self.glottis_lstm = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        self.glottis_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.LayerNorm(self.n_dim),
            nn.Sigmoid(),
        )
        
        self.selector = PDselector(n_dim = self.n_dim,n_preference = (20,20),end_residual_rate = 0.1)
        
        
        self.onehot_encoder = nn.Sequential(
            nn.Linear(self.n_dim,self.n_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_dim,10),
            
            )
        
    def forward(self, x):
        batch_size, n_sample, feature, time_step = x.size()
        model_device = next(self.parameters()).device
        x_in = (torch.reshape(x,[batch_size*n_sample,feature, time_step])).clone().to(model_device)
        x_in = x_in.transpose(1,2) #编排好输入模型的格式 (batch*n_sample,time,feature)
        
        y = self.pre_encoder(x_in)
        
        y_phoneme,y_trait = self.vocal_lstm(y)
        y_trait   = self.trait_encoder(y_trait[1])
        
        y_glottis,_ = self.glottis_lstm(y)
        y_glottis   = self.glottis_encoder(y_glottis)
        
        z_speaker = y_glottis*y_trait.transpose(0,1)
        
        x_out = 0
                
        z_speaker_R1 = torch.reshape(torch.mean(z_speaker,dim=1),[batch_size*n_sample,-1]).unsqueeze(0) # (1,batch*n_sample,n_dim)
        z_speaker_R1 = z_speaker_R1/(torch.norm(z_speaker_R1,2,dim=-1)+EPS).unsqueeze(-1)
        w = self.selector(z_speaker_R1)
        #w = self.onehot_encoder(z_speaker_R1)
        #w = F.softmax(w,dim=-1)
        return x_out,w


class Encoder_onehot(nn.Module):
    def __init__(self):
        super(Encoder_onehot, self).__init__()
        self.n_fft = 512
        self.n_feature = int(self.n_fft/2)+1
        self.n_hidden = 512
        self.out_channels = 32
        self.n_dim = int(self.out_channels*4) # 需要乘上一个正整数
        
        
        # pre-coding
        self.pre_encoder = nn.Sequential(
            nn.Linear(self.n_feature,self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden,self.n_hidden),
            nn.ReLU(inplace=True),
            
        )
        
        # vocal tract coding
        self.vocal_lstm = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        # speaker trait encode
        self.trait_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n_dim)
        )
        
        # glottis coding
        self.glottis_lstm = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        self.glottis_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.LayerNorm(self.n_dim),
            nn.Sigmoid(),
        )
        
        self.selector = PDselector(n_dim = self.n_dim,n_preference = (20,20),end_residual_rate = 0.1)
        
        
        self.onehot_encoder = nn.Sequential(
            nn.Linear(self.n_dim,self.n_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_dim,10),
            
            )
        
    def forward(self, x):
        batch_size, n_sample, feature, time_step = x.size()
        model_device = next(self.parameters()).device
        x_in = (torch.reshape(x,[batch_size*n_sample,feature, time_step])).clone().to(model_device)
        x_in = x_in.transpose(1,2) #编排好输入模型的格式 (batch*n_sample,time,feature)
        
        y = self.pre_encoder(x_in)
        
        y_phoneme,y_trait = self.vocal_lstm(y)
        y_trait   = self.trait_encoder(y_trait[1])
        
        y_glottis,_ = self.glottis_lstm(y)
        y_glottis   = self.glottis_encoder(y_glottis)
        
        z_speaker = y_glottis*y_trait.transpose(0,1)
        
        x_out = 0
                
        z_speaker_R1 = torch.reshape(torch.mean(z_speaker,dim=1),[batch_size*n_sample,-1]).unsqueeze(0) # (1,batch*n_sample,n_dim)
        #z_speaker_R1 = z_speaker_R1/(torch.norm(z_speaker_R1,2,dim=-1)+EPS).unsqueeze(-1)
        #w = self.selector(z_speaker_R1)
        w = self.onehot_encoder(z_speaker_R1)
        w = F.softmax(w,dim=-1)
        return x_out,w



class Autoencoder_onehot(nn.Module):
    def __init__(self):
        super(Autoencoder_onehot, self).__init__()
        self.n_fft = 512
        self.n_feature = int(self.n_fft/2)+1
        self.n_hidden = 512
        self.out_channels = 32
        self.n_dim = int(self.out_channels*4) # 需要乘上一个正整数
        
        
        # pre-coding
        self.pre_encoder = nn.Sequential(
            nn.Linear(self.n_feature,self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden,self.n_hidden),
            nn.ReLU(inplace=True),
            
        )
        
        # vocal tract coding
        self.vocal_lstm = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        # speaker trait encode
        self.trait_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n_dim)
        )
        # Standard phoneme coding
        self.phoneme_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.n_dim)
        )
        
        # glottis coding
        self.glottis_lstm = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        self.glottis_encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_dim),
            nn.LayerNorm(self.n_dim),
            nn.Sigmoid(),
        )
        
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.n_dim, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.n_feature),
        )
        
        self.selector = PDselector(n_dim = self.n_dim,n_preference = (20,20),end_residual_rate = 0.1)
        
        
        self.onehot_encoder = nn.Sequential(
            nn.Linear(self.n_dim,self.n_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_dim,10),
            
            )
        
    def forward(self, x):
        batch_size, n_sample, feature, time_step = x.size()
        model_device = next(self.parameters()).device
        x_in = (torch.reshape(x,[batch_size*n_sample,feature, time_step])).clone().to(model_device)
        x_in = x_in.transpose(1,2) #编排好输入模型的格式 (batch*n_sample,time,feature)
        
        y = self.pre_encoder(x_in)
        
        y_phoneme,y_trait = self.vocal_lstm(y)
        y_trait   = self.trait_encoder(y_trait[1])
        y_phoneme = self.phoneme_encoder(y_phoneme)
        
        y_glottis,_ = self.glottis_lstm(y)
        y_glottis   = self.glottis_encoder(y_glottis)
        
        z_speaker = y_glottis*y_trait.transpose(0,1)
        z_vocal   = y_glottis*y_phoneme
        
        x_out = 0
        x_out = self.decoder(z_speaker+z_vocal).transpose(1,2)
        x_out = torch.reshape(x_out,[batch_size, n_sample, feature, time_step]) # (batch,n_sample,feature,time)
        
        z_speaker_R1 = torch.reshape(torch.mean(z_speaker,dim=1),[batch_size*n_sample,-1]).unsqueeze(0) # (1,batch*n_sample,n_dim)
        #z_speaker_R1 = z_speaker_R1/(torch.norm(z_speaker_R1,2,dim=-1)+EPS).unsqueeze(-1)
        #w = self.selector(z_speaker_R1)
        w = self.onehot_encoder(z_speaker_R1)
        w = F.softmax(w,dim=-1)
        return x_out,w




class conv_test(nn.Module):
    def __init__(self):
        super(conv_test, self).__init__()
        self.n_fft = 512
        self.n_feature = int(self.n_fft/2)+1
        self.out_channels = 32
        self.n_dim = int(self.out_channels*4) # 需要乘上一个正整数
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.n_feature,self.n_feature-1),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_feature-1,self.n_feature-1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Linear(int(256/2**4),int(self.n_dim/self.out_channels)),
            nn.ReLU(inplace=True),
            
            #nn.LayerNorm()
            
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(int(self.n_dim/self.out_channels),int(256/2**4)),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_feature-1,self.n_feature),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_feature,self.n_feature)
            
        )
        
        self.selector = PDselector(n_dim = self.n_dim,n_preference = (20,20),end_residual_rate = 0.1)
        
        
        self.onehot_encoder = nn.Sequential(
            nn.Linear(self.n_dim,self.n_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_dim,10),
            
            )
        
    def forward(self, x):
        batch_size, n_sample, feature, time_step = x.size()
        model_device = next(self.parameters()).device
        x_in = (torch.reshape(x,[batch_size*n_sample,feature, time_step])).clone().to(model_device)
        x_in = x_in.transpose(1,2).unsqueeze(1) #编排好输入模型的格式 (batch*n_sample,1,time,feature)
        y = self.encoder(x_in)
        z=0
        z = self.decoder(y).transpose(2,3)
        z = torch.reshape(z,[batch_size, n_sample, feature, time_step])
        
        y_R1 = torch.reshape(torch.mean(y,dim=2),[batch_size*n_sample,-1]).unsqueeze(0) # (batch1,n_sample~b*n,n_dim)
        y_R1 = y_R1/(torch.norm(y_R1,2,dim=-1)+EPS).unsqueeze(-1)
        w = self.selector(y_R1)
        #w = self.onehot_encoder(y_R1)
        #w = F.softmax(w,dim=-1)
        return z,w


