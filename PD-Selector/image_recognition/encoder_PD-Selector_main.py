import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

import sys
sys.path.append('./image_recognition')

from PD_Selector import build_samples_classes,PDselector
from image_model import Encoder_fig
# 定义 device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")




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

# 有监督学习 （即输入中含有标签信息）

n_dim = 20

encoder = Encoder_fig(n_dim=n_dim)
selector = PDselector(n_dim = n_dim,n_preference = (20,20),end_residual_rate = 0.1)
#encoder.load_state_dict(torch.load("FashionMNIST_encoder.pth"))
#selector.load_state_dict(torch.load("FashionMNIST_selector.pth"))

# 创建数据加载器
batch_size = 200
n_batchToSample = 200 # 必须能整除 batch_size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(selector.parameters()),lr = 1e-3) 

loss_record = []
acc_record = []
# 下面的计算有值得简化的地方
for epoch_idx in range(1):
    loss_avg = 0
    acc_avg = 0
    for idx, (images, labels_value) in enumerate(train_loader):
        optimizer.zero_grad()
        
        labels = torch.zeros([labels_value.size(0),10])
        labels.scatter_(1, labels_value.unsqueeze(1), 1)
        
        y_fig = encoder(images) # (batch,1,w,h)
        random_coef = 1#torch.rand(y_fig.size(0), 1)# *0.2 +0.8
        y = y_fig/(torch.norm(y_fig,p=2,dim=-1,keepdim=True)+1e-8)
        y = torch.split(y, y.size(0)//n_batchToSample, dim=0)  # 在第一维度上进行切割
        y = torch.stack(y,dim=1)  # 在第二维度上进行拼接 (batch, n_sample, n_dim)
        
        with torch.no_grad():
            classes_cluster = torch.split(labels, labels.size(0)//n_batchToSample, dim=0)
            classes_cluster = torch.stack(classes_cluster, dim=2) # (batch, n_class, n_sample)
            
        selector.candidate=20
        w = selector(y,residual_cut=True)
        loss = selector.loss_supervised(w,classes_cluster)
        loss_avg += loss.item()
        loss.backward() # 反向传播算梯度
        optimizer.step() # 反向传播改参数
        acc_avg += selector.clustering_accuracy(w, classes_cluster)
        
        #if idx==200:
        #    break
        
    loss_avg /= (idx+1)
    acc_avg /= (idx+1)
    print("loss_avg:",loss_avg,"   accuracy:",acc_avg)
    loss_record.append(loss_avg)
    acc_record.append(acc_avg)
    torch.cuda.empty_cache()

torch.save(encoder.state_dict(), "./FashionMNIST_encoder.pth")
torch.save(selector.state_dict(), "./FashionMNIST_selector.pth")

#%%
# 模型测试

encoder = Encoder_fig(n_dim=n_dim)
selector = PDselector(n_dim = n_dim,n_preference = (20,20),end_residual_rate = 0.1)
encoder.load_state_dict(torch.load("./FashionMNIST_encoder.pth"))
selector.load_state_dict(torch.load("./FashionMNIST_selector.pth"))

# 创建数据加载器
batch_size = 200
n_batchToSample = 200 # 必须能整除 batch_size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 测试
with torch.no_grad():
    loss_avg = 0
    acc_avg = 0
    for idx, (images, labels_value) in enumerate(test_loader):
        
        labels = torch.zeros([labels_value.size(0),10])
        labels.scatter_(1, labels_value.unsqueeze(1), 1)
        
        y_fig = encoder(images) # (batch,n_dim)
        y = y_fig/(torch.norm(y_fig,p=2,dim=-1,keepdim=True)+1e-8)
        y = torch.split(y, y.size(0)//n_batchToSample, dim=0)  # 在第一维度上进行切割
        y = torch.stack(y,dim=1)  # 在第二维度上进行拼接 (batch, n_sample, n_dim)
        
        with torch.no_grad():
            classes_cluster = torch.split(labels, labels.size(0)//n_batchToSample, dim=0)
            classes_cluster = torch.stack(classes_cluster, dim=2) # (batch, n_class, n_sample)
            
        selector.candidate=20
        w = selector(y,residual_cut=True)
        acc_avg += selector.clustering_accuracy(w, classes_cluster)
    acc_avg /= (idx+1)
    print("accuracy:",acc_avg)

#%%
# 绘图测试
import matplotlib.pyplot as plt

encoder = Encoder_fig(n_dim=n_dim)
selector = PDselector(n_dim = n_dim,n_preference = (20,20),end_residual_rate = 0.1)
encoder.load_state_dict(torch.load("./FashionMNIST_encoder.pth"))
selector.load_state_dict(torch.load("./FashionMNIST_selector.pth"))

n_batchToSample = 200
# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=n_batchToSample, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=n_batchToSample, shuffle=False)

encoder.eval()
selector.eval()
with torch.no_grad():
    acc_avg = 0
    for idx, (images, labels_value) in enumerate(train_loader):
        labels = torch.zeros([labels_value.size(0),10])
        labels.scatter_(1, labels_value.unsqueeze(1), 1)
        
        y_fig = encoder(images) # (batch,n_dim)
        y = y_fig/(torch.norm(y_fig,p=2,dim=-1,keepdim=True)+1e-8)
        y = torch.split(y, y.size(0)//n_batchToSample, dim=0)  # 在第一维度上进行切割
        y = torch.stack(y,dim=1)  # 在第二维度上进行拼接 (batch, n_sample, n_dim)
        
        
        y0 = y[0].detach().numpy()
        plt.figure(figsize=(8,8))
        
        for class_idx in range(10):
            class_index = labels[:,class_idx]==1
            plt.scatter(y0[class_index,0], y0[class_index,1],marker=".")
            
        plt.xlim((-1.2,1.2))
        plt.ylim((-1.2,1.2))
        plt.title("distrbution")
        plt.show()
        plt.close()
        
        if idx == 0:
            break


    
