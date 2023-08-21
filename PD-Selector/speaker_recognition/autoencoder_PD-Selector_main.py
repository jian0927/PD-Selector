import os
import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt

from PD_Selector import PDselector
from speaker_model import Autoencoder, conv_test
from utils import SpeechDataset, build_speaker_list, read_arrays_from_txt, retrieve_speaker_list, save_list_to_txt

EPS = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
    
#%%
# 训练模型

seed = 42  # 你可以选择任意整数作为种子
torch.manual_seed(seed)  # 设置PyTorch随机种子
np.random.seed(seed)  # 设置NumPy的随机种子


# 语音文件路径
data_dir = "./speech_data/ST-AEDS-20180100_1-OS"
batch_size = 6 # 每次反向传播训练时，用了几个 batch,即有多少段语言
n_class = 10 # 本数据集中共有多少个类别

# 创建数据集和数据加载器
train_dataset = SpeechDataset(data_dir)
val_dataset   = SpeechDataset(data_dir)
traini,vali,testi = train_dataset.random_sample_list(train_dataset.__len__(), 
                                    train_rate=0.8,val_rate=0.2,test_rate=0.) # 在原数据集上建训练集和验证集映射列表

train_dataset.file_list = [train_dataset.file_list[i] for i in traini]       # 用更改列表的方法实现
val_dataset.file_list   = [val_dataset.file_list[i] for i in vali]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# 构建说话人列表，用于标签映射
sorted_categories = build_speaker_list(data_dir)

model = Autoencoder().to(device)
model.load_state_dict(torch.load("speaker_recognition.pth")) # *****直接读取训练好的
optimizer = torch.optim.Adam(list(model.parameters()),lr = 1e-3)
criterion = nn.MSELoss()

n_sample = train_dataset.n_sample

# L_tar1 代表仅仅知晓某个样本归属于哪个语音段
L_tar1 = torch.zeros([1,n_class,batch_size*n_sample]) 
for si in range(batch_size):
    L_tar1[0,si,si*n_sample:(si+1)*n_sample]=1

loss_record=[]
acc_record = []
acc_record1 = []
acc_record_train = []
#acc_record = read_arrays_from_txt('./acc_record.txt')  #******** 继承之前的 acc记录
for epochi in range(400):
    
    # 训练
    model.train()
    loss_avg = 0
    acc_avg_train=0
    for idx, (batch,fileName_list) in enumerate(train_loader):
        batch = batch.to(device) # 保证输入数据的设备与模型一致
        # L_tar2 代表知晓同属于一个说话人的语音样本应属于同以内
        positions = retrieve_speaker_list(fileName_list,sorted_categories) # 获得本次训练语音是哪个说话人（即映射在sorted_categories）
        L_tar20 = torch.zeros([1,n_class,batch_size*n_sample])
        for si in range(batch_size):
            L_tar20[0,positions[si],si*n_sample:(si+1)*n_sample]=1
            L_tar20_sum = torch.sum(L_tar20,dim=-1).squeeze(0)
            nonZero = L_tar20_sum!=0
            L_tar2 = L_tar20[0,nonZero,:].unsqueeze(0)


        optimizer.zero_grad()
        z,w = model(torch.log(batch**2+EPS)) # log 将输入变得相对平滑点
        loss_autoencode = criterion(z,torch.log(batch**2+EPS)) # autoencoder
        loss_selector = model.selector.loss_supervised(w,L_tar2) # PD-Selector 训练使用 L_tar1 或者 L_tar2
        loss = loss_selector + loss_autoencode # autoencoder & PD-Selector
        
        loss.backward() # 反向传播算梯度
        optimizer.step() # 反向传播改参数
        
        acc_avg_train += model.selector.clustering_accuracy(w, L_tar2)  #验证计算 acc 使用 L_tar1 或者 L_tar2
        
        loss_avg += loss.item()
        if idx == min(np.inf,len(val_loader)-2):
            break
    
    acc_avg_train/=(idx+1)
    acc_record_train.append(acc_avg_train)
    #save_list_to_txt("./acc_record_train.txt", acc_record_train)
    loss_avg /= (idx+1)
    loss_record.append(loss_avg)
    torch.cuda.empty_cache()
    
    #保存 模型
    torch.save(model.state_dict(), "speaker_recognition.pth")
    
    # 验证
    model.eval()
    with torch.no_grad():
        acc_avg = 0
        acc_avg1 = 0
        for idx, (batch,fileName_list) in enumerate(val_loader):
            batch = batch.to(device) # 保证输入数据的设备与模型一致
            
            # L_tar2 代表知晓同属于一个说话人的语音样本应属于同以内
            positions = retrieve_speaker_list(fileName_list,sorted_categories) # 获得本次训练语音是哪个说话人（即映射在sorted_categories）
            L_tar20 = torch.zeros([1,n_class,batch_size*n_sample])
            for si in range(batch_size):
                L_tar20[0,positions[si],si*n_sample:(si+1)*n_sample]=1
                L_tar20_sum = torch.sum(L_tar20,dim=-1).squeeze(0)
                nonZero = L_tar20_sum!=0
                L_tar2 = L_tar20[0,nonZero,:].unsqueeze(0)
                
            z,w = model(torch.log(batch**2+EPS)) # log 将输入变得相对平滑点
            
            acc_avg += model.selector.clustering_accuracy(w, L_tar2)  #验证计算 acc 使用 L_tar1 或者 L_tar2
            acc_avg1 += model.selector.clustering_accuracy(w, L_tar1)  # 验证计算 acc 使用 L_tar1 或者 L_tar2
            
            if idx == min(np.inf,len(val_loader)-2):
                break

        acc_avg /= (idx+1)
        acc_avg1 /= (idx + 1)
        acc_record.append(acc_avg)
        acc_record1.append(acc_avg1)
        
        save_list_to_txt("./acc_record.txt", acc_record)
        save_list_to_txt("./acc_record1.txt", acc_record1)
        
        
        torch.cuda.empty_cache()
    print("loss_avg:",loss_avg,"acc_avg_train:",acc_avg_train,"acc_avg:",acc_avg,"acc_avg1:",acc_avg1)
    








