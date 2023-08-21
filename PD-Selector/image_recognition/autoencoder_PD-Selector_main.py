import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

import sys
sys.path.append('./image_recognition')

from PD_Selector import build_samples_classes,PDselector
from image_model import Autoencoder_fig
# 定义 device
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
device = torch.device("cpu")




# 设置随机种子，以便结果可复现
torch.manual_seed(0)

# 定义超参数


n_dim = 4
n_sample = 30
batch_size = 60
n_batchToSample = 6

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 对图像进行标准化
])

# 加载 Fashion MNIST 训练集和测试集, 路径选择要根据设置的根目录决定
#train_dataset = FashionMNIST(root='./image_recognition/image_data', train=True, transform=transform, download=True)
#test_dataset = FashionMNIST(root='./image_recognition/image_data', train=False, transform=transform)
train_dataset = FashionMNIST(root='./image_data', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='./image_data', train=False, transform=transform)

#%%
# debug
# 模型测试


#%%
n_dim = 10
# 自编码模式中途有聚类
autoencoder = Autoencoder_fig(n_dim = n_dim).to(device)

# 创建数据加载器
batch_size = 200
n_batchToSample = 200 # 必须能整除 batch_size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(list(autoencoder.parameters()),lr = 1e-3) 


loss_record = []
acc_record = []
# 下面的计算有值得简化的地方
for epoch_idx in range(20):
    loss_avg = 0
    acc_avg = 0
    for idx, (images, labels_value) in enumerate(train_loader):
        optimizer.zero_grad()
        
        labels = torch.zeros([labels_value.size(0),10])
        labels.scatter_(1, labels_value.unsqueeze(1), 1)
        
        autoencoder.selector.candidate=20
        z_fig, w_selected = autoencoder(images) # (batch,n_dim)
        
        with torch.no_grad():
            classes_cluster = torch.split(labels, labels.size(0)//n_batchToSample, dim=0)
            classes_cluster = torch.stack(classes_cluster, dim=2) # (batch, n_class, n_sample)
            
        autoencoder.selector.candidate=20
        loss_selector = autoencoder.selector.loss_supervised(w_selected,classes_cluster)
        loss_fig = autoencoder.decoder.loss_reconstruction(z_fig,images)
        loss = loss_selector + loss_fig
        loss_avg += loss.item()
        loss.backward() # 反向传播算梯度
        optimizer.step() # 反向传播改参数
        acc_avg += autoencoder.selector.clustering_accuracy(w_selected, classes_cluster)
        
        #if idx==200:
        #    break
        
    loss_avg /= (idx+1)
    acc_avg /= (idx+1)
    print("loss_avg:",loss_avg,"   accuracy:",acc_avg)
    loss_record.append(loss_avg)
    acc_record.append(acc_avg)
    torch.cuda.empty_cache()


    
