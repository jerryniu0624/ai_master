import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # dropout
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc1 = nn.Linear(in_features=32 * 32 * 3, out_features=10)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        '''
        # try to define different layers 
        '''

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(-x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        '''
        # try to forward with different predefined layers 
        '''
        return x
        
class Net2(nn.Module):
	# init定义网络中的结构
    def __init__(self):
        super().__init__()
        # 3输入，16输出，卷积核(7, 7)，膨胀系数为2
        self.conv1 = nn.Conv2d(3,6,5,padding=2)  
        self.conv2 = nn.Conv2d(6,16,5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        # dropout
        self.conv2_drop = nn.Dropout2d()
        # 全连接层
        self.fc1 = nn.Linear(8*8*16,200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 10)
	
	# forward定义数据在网络中的流向
    def forward(self, x):
    	# 卷积之后做一个最大池化，然后RELU激活
        x = self.pool(F.relu(self.conv1(-x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # 整形
        #x = x.view(-1, 55696*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return x
        
        
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,3)
        self.conv3=nn.Conv2d(16,32,3)
        self.fc1=nn.Linear(32*5*5,400)
        self.fc2=nn.Linear(400,200)
        self.fc3=nn.Linear(200,100)
        self.fc4=nn.Linear(100,10)
    def forward(self, x):
        x=self.pool(self.conv1(x))
        x=self.conv2(x)
        x=self.pool(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        x=self.fc4(x)
        return x
        
        

#net = Net1()
#net.to(device)
#print(net)
#!!!!!!!!
#net.cuda()
#print(net.cuda())