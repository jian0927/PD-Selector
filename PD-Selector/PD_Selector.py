import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

# 角度代表不同的分布
# 点到中心的距离代表模型的这个偏好对这一点属于这个分布的信心
# distribution preference selector
class PDselector(nn.Module):
    # Distributed preference selector
    def __init__(self,n_dim = 3,n_preference = (20,20),end_residual_rate = 0.05):
        super(PDselector, self).__init__()
        # 下面定义模型的一些参数
        self.n_dim = n_dim
        self.n_preference = n_preference
        n_preferenceN = self.n_preference[0]*self.n_preference[1]
        self.kernel_size = 3
        self.padding = self.kernel_size // 2
        self.candidate = n_preferenceN # 被候选偏好数量的最大值, 默认全选
        self.inhibition_coef = 0.1 #抑制系数
        self.end_residual_rate = end_residual_rate
        
        self.selected_pid = []
        
        # 下面定义模型的各个层
        
        self.fc1 = nn.Linear(self.n_dim, n_preferenceN)
        self.ln1 = torch.nn.LayerNorm(n_preferenceN)
        self.dropout1 = torch.nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(n_preferenceN, n_preferenceN)
        self.ln2 = torch.nn.LayerNorm(n_preferenceN)
        self.dropout2 = torch.nn.Dropout(0.1)
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=5,kernel_size=self.kernel_size,stride=1)
        self.conv2 = nn.Conv2d(in_channels=5,out_channels=10,kernel_size=self.kernel_size,stride=1)

    def forward(self, x0,residual_cut=True):
        # x0: (batch,n_sample,n_dim)
        # w: (batch,preference,n_sample)
        model_device = next(self.parameters()).device
        x0 = x0.to(model_device)
        
        batch_size, n_sample, n_dim = x0.size()
        w = F.relu(self.fc1(x0))
        w = self.dropout1(self.ln1(w))
        
        w = F.relu(self.fc2(w))
        w = self.dropout2(self.ln2(w))
        '''   '''
        w = w.view(batch_size*n_sample,1, self.n_preference[0], self.n_preference[1])
        
        w = F.pad(w, [self.padding, self.padding, self.padding, self.padding], mode='circular') # 形成对循环边界的卷积（球面）
        w = self.conv1(w)
        w = F.pad(w, [self.padding, self.padding, self.padding, self.padding], mode='circular')
        w = self.conv2(w)
        w = w.mean(dim=1)
        #w = w.sum(dim=1)
        w = w.view(batch_size, n_sample, self.n_preference[0]*self.n_preference[1])
        
        
        w = F.sigmoid(w)
        w = w.permute(0,2,1) # (batch,preference,n_sample)
        
        self.selected_pid = self.select_preference(x0,w,residual_cut=residual_cut) # selected_pid: [batch](pid,)
        
        bursting = torch.zeros_like(w)+self.inhibition_coef # (batch,preference,n_sample)
        for n in range(batch_size):
            bursting[n,self.selected_pid[n],:]=1
        w = w*bursting
        
        return w
    
    def sigmoid(self, x, k ,x_mid=0.5):
        with torch.no_grad():
            a = 1/(1 + np.exp(-k*(0 - x_mid)))
            b = 1/(1 + np.exp(-k*(1 - x_mid)))
            output = (1 / (1 + torch.exp(-k*(x.detach() - x_mid))) - a)/(b- a)
        
        return output
    
    def select_preference(self,x0,w,residual_cut=True):
        # x0: (batch,n_sample,n_dim) |  w:(batch,preference,n_sample)
        # 选择标准：活性
        # 截至模式 residual_cut=True 表示用样本残留量截止; residual_cut=True 表示用直到满足 self.candidate
        # 输出 selected_pid 即被选中的 偏好id, [batch](pid,) 即列表存储的记录的偏好id的张量
        model_device = next(self.parameters()).device
        with torch.no_grad():
            selected_pid = []
            w_c = w.clone() # 避免影响到原 w
            for n in range(x0.size(0)):
                pid = []
                x0n_residual = torch.ones([x0.size(1),]).to(model_device) # (n_sample,)
                for pi in range(self.candidate):
                    wn_residual = w_c[n]*x0n_residual.unsqueeze(0) # (preference,n_sample)
                    x0n_w = (x0[n].unsqueeze(0) * wn_residual.unsqueeze(2)) #(preference,sample,n_dim_wighted)
                    
                    # 计算活跃度
                    x0n_w_mean = x0n_w.mean(dim = 2) # (preference,n_dim_wight)
                    x0n_activity = torch.norm(x0n_w_mean,p=2,dim=1)   # (preference)
                    #x0n_w_norm = x0n_w_mean/x0n_activity.unsqueeze(1)  # (preference,n_dim_wight)
                    
                    # 选出当前最佳，并计算样本残留
                    pid.append(torch.argmax(x0n_activity)) # [pid]
                    x0n_residual -= w_c[n,pid[-1],:]
                    w_c[n,pid[-1],:]=0
                    #x0n_residual -= self.sigmoid(w_c[n,pid[-1],:],k=1,x_mid=0.5)
                    x0n_residual[x0n_residual<0]=0
                    
                    
                    
                    if residual_cut==True and torch.mean(x0n_residual)<self.end_residual_rate:
                        break
                selected_pid.append(torch.stack(pid,dim=0))
            
        return selected_pid
    
    
    
    def match_classes(self,similar_matrix, max_order=True):
        # 两矢量集 (n1) (n2) 依据 similar_matrix， 逐次按照 max_order 的顺序做最佳匹配，返回匹配列表
        # similar_matrix (n1,n2), 记录两矢量集之间的距离
        # max_order， 按照从大到小逐次匹配，默认为 True
        # 输出 index,  row 对应 n1, col 对应 n2
        with torch.no_grad(): 
            if max_order:
                out_value = torch.min(similar_matrix)-1 # 用于对已经被匹配上的索引所在行列涂黑的值
            else:
                out_value = torch.max(similar_matrix)+1
            match_index = []
            dist = similar_matrix.clone() # 避免后续的赋值对比操作污染了原 similar_matrix
            for sp in range(min(similar_matrix.shape[0],similar_matrix.shape[1])):
                if max_order:
                    selected_index = torch.argmax(dist)  # 找到最大值的索引位置
                else:
                    selected_index = torch.argmin(dist)  # 找到最小值的索引位置
                # 将一维索引位置转换为二维坐标
                row = selected_index // similar_matrix.size(1)  # 行索引
                col = selected_index % similar_matrix.size(1)  # 列索引
                match_index.append(torch.stack((row,col),dim=0).view(1,2)) # row 对应target的source, col 对应 outputs 的 preference
                dist[row,:] = out_value # 将已经被选择的偏好和有偏好的 targets 涂黑
                dist[:,col] = out_value
            
            match_index = torch.cat(match_index,dim=0)
            
        return match_index
    
    
    def loss_supervised(self, w, classes):
        # w: (batch,preference,samples)
        # classes: (batch,sources,samples)
        criterion = nn.MSELoss()  # 定义选择使用的损失函数
        loss = 0
        model_device = next(self.parameters()).device
        classes = classes.to(model_device) # 保证规范到设备上
        for n in range(w.size(0)):
            wn_selected = w[n,self.selected_pid[n]]
            with torch.no_grad(): 
                # 计算 wn_selected 与 聚类样本间的距离，并做匹配
                non_zero = torch.sum(classes[n],dim=1) != 0
                dist = torch.sum(torch.abs((wn_selected.view(1,-1,w.size(2)) - classes[n,non_zero].unsqueeze(1))), dim=2)
                match_index = self.match_classes(dist,max_order=False)
                
                # 不确定是否要添加对未匹配的标签补充的计算，可以添加作为一个改进算法的方向
                if wn_selected.size(0) < torch.sum(non_zero):
                    pass
                
                 # row 对应 classesn 的source, col 对应 wn 的 preference
                classes_matched = classes[n,match_index[:,0],:]
            w_matched = wn_selected[match_index[:,1],:]
            
            loss += criterion(w_matched, classes_matched)

                
        return loss/(n+1)
    
    def loss_supervised_center(self, x0, w, classes):
        # x0: (batch,samples,dim)
        # w: (batch,preference,samples)
        # classes: (batch,sources,samples)
        class PairwiseLoss(nn.Module):
            # Pairwise Loss
            def __init__(self):
                super(PairwiseLoss, self).__init__()

            def forward(self, x0, w_matched, classes_matched):
                diff = torch.mean((w_matched-classes_matched)**2)
                x0_w_matched = x0[n].unsqueeze(0)*classes_matched.unsqueeze(-1)
                var  = torch.mean(torch.var(x0_w_matched,dim=1))
                
                loss = diff + var*10
                return loss
        criterion = PairwiseLoss()  # 定义选择使用的损失函数
        loss = 0
        for n in range(w.size(0)):
            wn_selected = w[n,self.selected_pid[n]]
            with torch.no_grad(): 
                # 计算 wn_selected 与 聚类样本间的距离，并做匹配
                non_zero = torch.sum(classes[n],dim=1) != 0
                dist = torch.sum(torch.abs((wn_selected.view(1,-1,w.size(2)) - classes[n,non_zero].unsqueeze(1))), dim=2)
                match_index = self.match_classes(dist,max_order=False)
                
                # 不确定是否要添加对未匹配的标签补充的计算，可以添加作为一个改进算法的方向
                if wn_selected.size(0) < torch.sum(non_zero):
                    pass
                
                 # row 对应 classesn 的source, col 对应 wn 的 preference
                classes_matched = classes[n,match_index[:,0],:]
            w_matched = wn_selected[match_index[:,1],:] # (preference, n_sample)

            loss += criterion(x0, w_matched, classes_matched)

        return loss/(n+1)
    
    
    def loss_unsupervised(self, x0, w, ):
        # x0: (batch,samples,dim)
        # w: (batch,preference,samples)
        criterion = nn.MSELoss()  # 定义选择使用的损失函数
        loss = 0
        for n in range(w.size(0)):
            wn_selected = w[n,self.selected_pid[n]] #(preference,samples)
            
            with torch.no_grad(): 
                # 计算各个被选偏好的中心
                xn_w = x0[n,:,:].unsqueeze(0)* wn_selected.unsqueeze(-1) #(preference,n_sample,n_dim)
                xn_w_mean = torch.mean(xn_w,dim=1) # (preference,n_dim)
                xn_w_mean_norm = xn_w_mean/(torch.norm(xn_w_mean,p=2,dim=-1,keepdim=True)+1e-8) # (preference,n_dim)
                
                # 计算各个样本距离偏好中心的欧氏距离，然后根据距离将样本分配给偏好,从而构建训练目标
                dist = torch.sum((x0[n].unsqueeze(0)-xn_w_mean_norm.unsqueeze(1))**2,dim=-1) # (preference,n_sample)
                matchList = torch.argmin(dist,dim=0) # 得到每个样本距离哪个偏好中心更近 #(n_sample,)
                wn_target = torch.zeros_like(wn_selected)
                for p in range(wn_selected.size(0)):
                    wn_target[p,matchList==p] = 1 # (preference,n_sampel)
                    
            loss += criterion(wn_selected,wn_target)
        
        return loss/(n+1)
    
    
    def clustering_accuracy(self, w, classes):
        # w: (batch, preference, n_sample)
        # classes: (batch,sorces,n_sample)
        # 计算规则：从 outputs 硬分类后 的每个偏好内的样本的角度看，找哪个 source 内正确的量大就把这个值放入分子
        # 分母是 n_samples, 输出结果是所有batch 的平均值
        model_device = next(self.parameters()).device
        with torch.no_grad():
            classes = classes.to(model_device) # 确保输入符合设备
            numerator = 0
            for n in range(w.size(0)):
                # 在样本的角度，每个样本，应该归属于哪个偏好，归属则设为1
                wn_selected = w[n,self.selected_pid[n]]
                wn_maxIndex = torch.argmax(wn_selected,dim=0) 
                wn_belong = torch.zeros_like(wn_selected)
                wn_belong.scatter_(0, wn_maxIndex.unsqueeze(0), 1)
                
                # 计算匹配列表
                non_zero = torch.sum(classes[n],dim=1) != 0
                classes_selected = classes[n,non_zero]
                dist = torch.sum(torch.abs((wn_belong.view(1,-1,w.size(2)) - classes_selected.unsqueeze(1))), dim=2)
                match_index = self.match_classes(dist,max_order=False)
                
                numerator += torch.sum(wn_belong[match_index[:,1],:]*classes_selected[match_index[:,0],:])
                
        return (numerator/(classes.size(0)*classes.size(2))).item()
    


# 构建数据
def build_samples_classes(n_dim=3,n_centers=100,n_class_random = [4],\
                         input_sample_n = 300,input_sample_n_least = 50,\
                         radius=0.1,n_batch =1000,n_rands = 10000,random_seed = 43):
    # 构造随机数据，每个类的样本都是标准差为 radius 高斯分布，样本点都在半径为1的 n 维球的球面
    # 输出 inputs (batch,n_sample,n_dim); targets (batch,sources,n_sample,n_dim,),注意 targets 中不是本 source的样本的 n_dim 为0
    # n_dim = 3       # 生产的样本是几个维度的
    # n_centers = 100 # 共生产多少个聚类中心
    # n_class_random = [4]     # 1 个batch 有多少个类（聚类中心），创建数据时做选择
    # input_sample_n = 300 # 每个batch 放多少个样本点
    # input_sample_n_least = 50 # 每个batch 的每个聚类中心放最少多少个样本点
    # radius=0.1 # 样本扩散程度 1标准差作为半径默认设置为 0.1，注意 n 维球的半径为 1 
    # n_rands = 10000 # 每个聚类中心有多少个样本点
    # n_batch =1000   # 有多少个batch 
    # random_seed = 43 随机数种子
    torch.manual_seed(random_seed) 
    np.random.seed(random_seed)
    
    samples = []
    classes = []
    n_class_max = max(n_class_random) # 1 个batch 最多有多少个类， 多余的部分直接用 0 表示,保证张量的形态
    input_sample_n_least = min(input_sample_n_least,int(input_sample_n/n_class_max))
    radius_dim = radius/np.sqrt(n_dim)
    
    centers = torch.randn([n_centers,n_dim]) # 定义了几个数据中心
    centers = centers/(torch.sqrt(torch.sum(centers**2,dim=(1))).unsqueeze(1))
    rands = torch.randn([n_centers,n_rands,n_dim])*radius_dim 
    dataSets = centers.unsqueeze(1) + rands
    dataSets = dataSets/torch.norm(dataSets,2,dim=2).unsqueeze(2)
    
    for n in range(n_batch): # 设定 batch 大小
        # 随机挑选源
        n_class = np.random.choice(n_class_random)
        source = dataSets[np.random.choice(np.linspace(0,n_centers-1,num=n_centers,dtype=np.int32),size=(n_class,),replace=False)]
        
        #随机挑选输入样本
        n_sample_in = np.random.rand(n_class)
        n_sample_in = np.floor(n_sample_in/np.sum(n_sample_in)*(input_sample_n-input_sample_n_least*n_class))+input_sample_n_least
        n_sample_in[-1] += input_sample_n - np.sum(n_sample_in)
        n_sample_in = np.asarray(n_sample_in,dtype=np.int32)
        
        source_n_random = []
        target = torch.zeros([n_class_max,input_sample_n])
        i = 0
        for s in range(source.shape[0]):
            source_n_random.append(source[s,torch.randint(0,n_rands,size=[n_sample_in[s],])])
            target[s,i:i+int(n_sample_in[s])] = 1
            i += int(n_sample_in[s])
            
        samples.append(torch.cat(source_n_random,dim=0))
        classes.append(target)
        
    samples = torch.stack(samples,dim=0)
    classes = torch.stack(classes,dim=0)
    
    return samples, classes


def build_inputs_targets2(n_dim=3,n_centers=100,n_class_random = [4],\
                         input_sample_n = 300,input_sample_n_least = 50,\
                         radius=0.1,n_batch =1000,n_rands = 10000,random_seed = 43):
    # 构造随机数据，每个类的样本都是标准差为 radius 高斯分布，样本点都在半径为1的 n 维球的球面
    # 输出 inputs (batch,n_sample,n_dim); targets (batch,sources,n_sample,n_dim,),注意 targets 中不是本 source的样本的 n_dim 为0
    # n_dim = 3       # 生产的样本是几个维度的
    # n_centers = 100 # 共生产多少个聚类中心
    # n_class_random = [4]     # 1 个batch 有多少个类（聚类中心），创建数据时做选择
    # input_sample_n = 300 # 每个batch 放多少个样本点
    # input_sample_n_least = 50 # 每个batch 的每个聚类中心放最少多少个样本点
    # radius=0.1 # 样本扩散程度 1标准差作为半径默认设置为 0.1，注意 n 维球的半径为 1 
    # n_rands = 10000 # 每个聚类中心有多少个样本点
    # n_batch =1000   # 有多少个batch 
    # random_seed = 43 随机数种子
    torch.manual_seed(random_seed) 
    np.random.seed(random_seed)
    
    inputs = []
    targets = []
    n_class_max = max(n_class_random) # 1 个batch 最多有多少个类， 多余的部分直接用 0 表示,保证张量的形态
    input_sample_n_least = min(input_sample_n_least,int(input_sample_n/n_class_max))
    radius_dim = radius/np.sqrt(n_dim)
    
    centers = torch.randn([n_centers,n_dim]) # 定义了几个数据中心
    centers = centers/(torch.sqrt(torch.sum(centers**2,dim=(1))).unsqueeze(1))
    rands = torch.randn([n_centers,n_rands,n_dim])*radius_dim 
    
    centers_samples = centers.unsqueeze(1) + rands
    centers_samples = centers_samples/torch.norm(centers_samples,2,dim=2).unsqueeze(2)
    
    dataSets = torch.cat([centers_samples[:50,:int(n_rands/2)],centers_samples[50:,:int(n_rands/2)]],dim=1)
    
    
    for n in range(n_batch): # 设定 batch 大小
        # 随机挑选源
        n_class = np.random.choice(n_class_random)
        source = dataSets[np.random.choice(np.linspace(0,dataSets.size(0)-1,num=dataSets.size(0),dtype=np.int32),size=(n_class,),replace=False)]
        
        #随机挑选输入样本
        n_sample_in = np.random.rand(n_class)
        n_sample_in = np.floor(n_sample_in/np.sum(n_sample_in)*(input_sample_n-input_sample_n_least*n_class))+input_sample_n_least
        n_sample_in[-1] += input_sample_n - np.sum(n_sample_in)
        n_sample_in = np.asarray(n_sample_in,dtype=np.int32)
        
        source_n_random = []
        target = torch.zeros([n_class_max,input_sample_n,n_dim])
        i = 0
        for s in range(source.shape[0]):
            source_n_random.append(source[s,torch.randint(0,n_rands,size=[n_sample_in[s],])])
            target[s,i:i+int(n_sample_in[s]),:] = source_n_random[-1]
            i += int(n_sample_in[s])
            
        inputs.append(torch.cat(source_n_random,dim=0))
        targets.append(target)
        
    inputs = torch.stack(inputs,dim=0)
    targets = torch.stack(targets,dim=0)
    
    return inputs, targets

