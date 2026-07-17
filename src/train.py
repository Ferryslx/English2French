import re
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

#设备选择
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#指定特殊的token
#起始标记
SOS_token =0
#结束标记
EOS_token =1
#最大句子长度不能超过10（包括标点）
MAX_LENGTH=10

with open('../data/eng-fra-v2.txt','r') as f:
    content = f.readline()
    print(content)