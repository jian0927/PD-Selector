import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn import functional as F
import matplotlib.pyplot as plt



class SpeechDataset(Dataset):
    # 构建从文件夹中读取 .wav 格式的语音，然后经过 STFT, 无语音部分删除，标准化裁剪后输出数据
    # 输出数据格式(batch,n_sample,f,t)
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [filename for filename in os.listdir(data_dir) if filename.endswith('.wav')] # 需要注意，仅收录后缀 .wav 文件
        
        self.n_fft = 512 # 傅里叶变换窗大小
        self.hop_length = 256  # 傅里叶变换步长
        self.n_sample = 30 # 每段语音中随机抽出的语音片段数量
        self.t_clip = 32 # 每个语音片段的持续时长
        self.kernel_wide = 9  # 保留可能的无语音片段范围 奇数
        self.std_k = 0.2 # 在计算活动阈值的时候，针对标准差设置的系数，默认0.2

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        audio_file = os.path.join(self.data_dir, self.file_list[idx])
        
        audio, sample_rate = torchaudio.load(audio_file)
        
        audio = self.preprocess_waveform(
            audio,n_sample=self.n_sample,t_clip=self.t_clip,n_fft=self.n_fft,hop_length=self.hop_length)
        
        return audio, self.file_list[idx]
    
    def random_sample_list(self, len_dataset, train_rate=0.8,val_rate=0.2,test_rate=0.):
        # 如果数据集没有分割训练集，验证集和测试集时，使用本函数构建一个对原数据的随机映射
        # len_dataset 是原数据集的样本数量
        # train_rate, val_rate, test_rate 分别时瓜分原数据集 dataset 的比例
        # 返回 3个数组，其数值是 训练集，验证集和测试集 在原数据集上的映射位置
            
        # 生成0到len_dataset-1的一维整数数组x
        x = np.arange(len_dataset)
        
        # 将数组x打乱顺序
        np.random.shuffle(x)
        
        # 将打乱后的数组分成三个部分，每个部分的长度
        # 可以根据需要进行调整
        boundary1 = min(int(train_rate*len_dataset),len_dataset)
        boundary2 = min(int((train_rate+val_rate)*len_dataset),len_dataset)
        boundary3 = min(int((train_rate+val_rate+test_rate)*len_dataset),len_dataset)
        
        part1 = x[:boundary1]
        part2 = x[boundary1:boundary2]
        part3 = x[boundary2:boundary3]
        
        return part1, part2, part3

    # 预处理函数
    def preprocess_waveform(self,waveform,n_sample=20,t_clip=32,n_fft=512,hop_length=256):
        # 预处理数据，将wav格式的语音转为声谱图，然后裁剪掉无语音片段，最后为了让
        # 张量保持一致，从裁剪的声谱图中按标准随机抽选 n_sample 个语音片段，
        # 每个片段长度为 t_clip (t_clip*hop_length/sr (s))
        
        # 这里使用 STFT 获得特征
        spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(waveform)
        
        spec_clipped = self.clip_spec(spec, kernel_wide = self.kernel_wide, in_rate = 0.1,std_k=0.2)
        
        random_int = torch.randint(low=0, high=spec.size(-1)-t_clip, size=(n_sample,), dtype=torch.int)
        spec_clip = []
        for i in range(n_sample):
            spec_clip.append(spec[0,:,random_int[i]:random_int[i]+t_clip])
        spec_clip = torch.stack(spec_clip,dim=0)
        
        return spec_clip

    def clip_spec(self,spec, kernel_wide = 15, in_rate = 0.1,std_k=0.2):
        # 将声谱图中无语音的片段裁剪掉
        # spec，声谱图，(batch=1,f=nfft/2+1,t)
        # kernel_wide,用于在有语音前后保留的片段长度，用卷积的方法实现，数值必须设为奇数
        # in_rate, 在卷积内，至少要有语音的比例
        # std_k, 在计算活动阈值的时候，针对标准差设置的系数，默认0.2就有比较好的效果
        
        # 计算阈值
        spec_mean = torch.mean(spec**2,dim=1)
        mean_value = torch.mean(spec_mean)
        std_value = torch.std(spec_mean)
        threshold = mean_value - std_value*std_k
        
        # 标记高于阈值的部分为1，否则为0
        voice_sign = torch.where(spec_mean > threshold, torch.tensor(True), torch.tensor(False))
        
        # 在有语音前后保留的一定的可能的无语音片段
        conv_kernel = torch.ones((1, kernel_wide))
        voice_sign_conv = F.conv1d(voice_sign.float(), conv_kernel.unsqueeze(0).float(),padding=kernel_wide//2)
        voice_sign_conv = voice_sign_conv[:,:]
        
        # 标记卷积结果中值大于的部分为True，否则为False
        voice_sign_conv = torch.where(voice_sign_conv > int(kernel_wide*in_rate), torch.tensor(True), torch.tensor(False))
        
        spec_clipped = spec*voice_sign_conv[:,:].unsqueeze(1) # 裁剪
        
        return spec_clipped





def build_speaker_list(data_dir):
    # 将路径内的文件，根据文件名，归纳得到所有说话人的索引列表
    
    # 文件名的列表, 注意，只针对 wav 格式
    file_names = [filename for filename in os.listdir(data_dir) if filename.endswith('.wav')]
    
    # 提取类别部分并存储到一个列表中， 文件名格式 "classId_---.wav", 即下划线前部分表明说话人
    categories = [file_name.split('_')[0] for file_name in file_names]
    
    # 使用set去除重复项，并转换成一个列表
    unique_categories = list(set(categories))
    
    # 对列表进行排序
    sorted_categories = sorted(unique_categories)
    
    return sorted_categories

def retrieve_speaker_list(file_name_list,sorted_categories):
    # 以排序后的类别作为标准，而列表file_name_list，内部也是装载的文件名，且属于之前的类别
    # file_name_list 内文件名根据标准, 查询每个文件名对应的位置
    
    positions = [sorted_categories.index(file_name.split('_')[0]) for file_name in file_name_list]
    return positions


# 显示声谱图
def plot_spectrogram(spectrogram, title, ylabel, xlabel,magType="DB"):
    if magType=="DB":
        specgram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    else:
        specgram = spectrogram
    
    plt.figure(figsize=(10, 8))
    plt.imshow(specgram, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    
    plt.show()


def one_hot_accuracy(output,label):
    # 在one-hot 模式下计算准确度
    # output(1,n_sample,n_class)
    # label(1,n_sample,n_class)
    
    output_argmax = torch.argmax(output,dim=-1).squeeze(0)
    #output_select = output[0,:,output_argmax]
    
    max_indices = torch.argmax(output, dim=2)

    # 创建一个全零张量，形状和 a 一样
    one_hot = torch.zeros_like(output)
    
    # 将最大值的位置设为1
    one_hot.scatter_(2, max_indices.unsqueeze(2), 1)
    
    correct_sum = torch.sum(one_hot * label)
    acc = (correct_sum/label.size(1)).item()
    
    return acc

def save_list_to_txt(file_path, data_list):
    with open(file_path, 'w') as file:
        file.write('\n'.join(map(str, data_list)))
    
def read_arrays_from_txt(file_path):
    arrays = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                arrays.append(float(line))
    except :
        print("读取文件",file_path, "出现错误，若无此文件则将会被创建。")

    return arrays
  












