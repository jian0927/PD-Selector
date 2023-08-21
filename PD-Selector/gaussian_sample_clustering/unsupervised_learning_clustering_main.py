import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from PD_Selector import build_samples_classes,PDselector
# 当前定义了 device, 这里使用 cpu 速度可能快点
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 构建数据集
n_dim=8
samples, classes = build_samples_classes(n_dim=n_dim,n_centers=100,n_class_random = [5,6,7,8,9,10],\
                                       input_sample_n = 100,input_sample_n_least = 10,\
                                       radius=0.1,n_batch =1000,n_rands = 10000,random_seed=43)



#%%
# debug test



#%%
# 无监督学习
TrainSet = TensorDataset(samples[:200], classes[:200])

# 进行训练
selector = PDselector(n_dim = n_dim,n_preference = (20,20),end_residual_rate = 0.1).to(device)
#selector.load_state_dict(torch.load("./gaussian_sample_clustering/selector.pth")) # 直接读取训练好的, 需要注意读取的模型参数要一致
trainData = DataLoader(TrainSet, batch_size=10, shuffle=False)


loss_record = []
acc_record = []
optimizer = torch.optim.Adam(selector.parameters(),lr = 1e-2) 
selector.train()
for epochi in range(30):
    loss_avg = 0
    acc_avg = 0
    for idx, (X_in,L_tar) in enumerate(trainData):
        
        optimizer.zero_grad()
        selector.candidate = 20
        # 不再需要 residual_cut=False 和设置 selector.candidate = k 了，算法会自动识别有多少个类别
        w = selector(X_in,residual_cut=True) 

        loss_selector = selector.loss_unsupervised(X_in,w)
        loss_avg += loss_selector.item()
        loss_selector.backward() # 反向传播算梯度
        optimizer.step() # 反向传播改参数
        acc_avg += selector.clustering_accuracy(w, L_tar)
        
        
    loss_avg /= (idx+1)
    acc_avg /= (idx+1)
    print("loss_avg:",loss_avg,"   accuracy:",acc_avg)
    loss_record.append(loss_avg)
    acc_record.append(acc_avg)
    torch.cuda.empty_cache()

#保存
torch.save(selector.state_dict(), "./selector.pth")
#%%
# 测试 
TestSet = TensorDataset(samples[500:], classes[500:])

# 读取模型
selector = PDselector(n_dim = n_dim,n_preference = (20,20),end_residual_rate = 0.1).to(device)
selector.load_state_dict(torch.load("./selector.pth")) # 直接读取训练好的, 需要注意读取的模型参数要一致
testData = DataLoader(TestSet, batch_size=10, shuffle=True)

# 测试
selector.eval()
with torch.no_grad():
    loss_avg = 0
    acc_avg = 0
    for idx, (X_in,L_tar) in enumerate(testData):
        
        w = selector(X_in,residual_cut=True)
        acc_avg += selector.clustering_accuracy(w, L_tar)
        

    acc_avg /= (idx+1)
    print("accuracy:",acc_avg)
    torch.cuda.empty_cache()


#%%
# 先后显示输出 源, 硬分类 和 软分类 (与原点距离代表信心)
'''   '''
import matplotlib.pyplot as plt

TestSet = TensorDataset(samples[500:], classes[500:])

selector = PDselector(n_dim = n_dim,n_preference = (20,20),end_residual_rate = 0.1).to(device)
selector.load_state_dict(torch.load("./selector.pth")) 

TestData = DataLoader(TestSet, batch_size=1, shuffle=True)
for idx, (X_in,L_tar) in enumerate(TestData):
    X_tar = X_in.unsqueeze(1)*L_tar.unsqueeze(-1)
    if idx == 1:
        break
    # 输出 源
    plt.figure(figsize=(8,8))
    plt.xlim((-1.2,1.2))
    plt.ylim((-1.2,1.2))
    for i in range(X_tar.size(1)):
        plt.scatter(X_tar[0,i,:,0].detach().numpy(), X_tar[0,i,:,1].detach().numpy(),marker="x")
    plt.title("Sources")
    plt.show()
    plt.close()
    
    
    # 先后输出 硬分类 和 软分类  (与原点距离代表信心)
    with torch.no_grad():
        selector.candidate = 10
        w = selector(X_in, residual_cut=True)
        w0_selected = w[0,selector.selected_pid[0]] #(preference,n_sample)
        x0_w_soft = X_in[0].unsqueeze(0)*w0_selected.unsqueeze(-1) #(preference,n_sample,n_dim)
        
        w0_maxIndex = torch.argmax(w0_selected,dim=0)
        w0_belong = torch.zeros_like(w0_selected)
        w0_belong.scatter_(0, w0_maxIndex.unsqueeze(0), 1)
        x0_w_hard = X_in[0].unsqueeze(0)*w0_belong.unsqueeze(-1) #(preference,n_sample,n_dim)
    
    # 显示硬分类
    plt.figure(figsize=(8,8))
    for p in range(x0_w_hard.size(0)):
        plt.scatter(x0_w_hard[p,:,0], x0_w_hard[p,:,1],marker=".")
        
    plt.xlim((-1.2,1.2))
    plt.ylim((-1.2,1.2))
    plt.title("Hard Classification")
    plt.show()
    plt.close()
    
    # 显示软分类
    plt.figure(figsize=(8,8))
    for p in range(x0_w_soft.size(0)):
        plt.scatter(x0_w_soft[p,:,0], x0_w_soft[p,:,1],marker=".")
        
    plt.xlim((-1.2,1.2))
    plt.ylim((-1.2,1.2))
    plt.title("Soft Classification")
    plt.show()
    plt.close()
    



































