import time
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model import Net1,Net2,Net3
from ResNet import ResNet18
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from AlexNet import AlexNet
import draw as d
from PIL import Image
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model",type=int,default=8)


args = parser.parse_args()

#import sys.argv
#net=argv[1]
loss1=[]
acc=[]


accuary=[]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def draw_loss(loss1):
#     x=range(0,loss1)
#     plt.title("loss") 
#     plt.xlabel("training times") 
#     plt.ylabel("loss") 
#     plt.plot(x,len(loss1))
#     plt.show()
#     plt.savefig('../image/loss.jpg')
    

  


def evaluate(model, dataloder):
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloder:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the networ
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %' .format(len(testloader),
            100 * correct / total))
    return 100 * correct / total

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize([227,227]),
         transforms.ToTensor(),
         transforms.RandomCrop(224,padding = 2 ,pad_if_needed = True,fill = 0,padding_mode ='constant'),
         transforms.RandomHorizontalFlip(p=0.5), # 表示进行左右的翻转
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    '''
    # try to set different batch_size
    '''
    batch_size = args.batch_size

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #trainset=Variable(trainset).cuda()
    #testset=Variable(testset).cuda()

    if args.model==0:
        net = Net1()
    elif args.model==1:
        net = Net2()
    elif args.model==2:
        net = Net3()
    elif args.model==3:
        net= ResNet18()
    elif args.model==8:
        net=AlexNet()
    
    print(net)
    net.cuda()
    
    
    criterion = F
    '''
    # try to change the learning rate
    '''
    optimizer1 = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer2 = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
    start = time.time()

    '''
    # try to change the number of total epoch 
    '''
    
    for epoch in range(64):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
        
            if epoch<20:
                optimizer=optimizer1
            else:
                optimizer=optimizer2
                
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) # 图像作为输入，经过net做正向传播，得到输出（分类/框/。。。）
            loss = criterion.cross_entropy(outputs, labels)
            
            loss.backward()
            optimizer.step() 
            

            # print statistics
            running_loss += loss.item()
            #print(type(loss.item()))
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                
                loss1.append(running_loss/100)
            
                running_loss = 0.0

        # evaluate at the end of every epoch
        acc.append(evaluate(net, testloader))

    end = time.time()

    print('Finished Training {}s'.format(end-start))

    PATH = './cifar_net'+'model: '+str(args.model)+'.pth'
    torch.save(net.state_dict(), PATH)
    #draw_loss(loss1)
    
    name1='loss'+'_model:'+str(args.model)+'.npy'
    name2='acc'+'_model:'+str(args.model)+'.npy'
    np.save(name1,loss1)
    np.save(name2,acc)
    d.draw_loss(loss1,name1)
    d.draw_acc(acc,name2)
