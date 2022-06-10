from PIL import Image
import os
import torch
import torchvision
from torchvision import transforms
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from model import Net1,Net2,Net3
from ResNet import ResNet18
from AlexNet import AlexNet
import pylab
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model",type=int,default=0)


args = parser.parse_args()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    # pylab.show()

if __name__ == '__main__':

    PATH = './cifar_net'+'model: '+str(args.model)+'.pth'
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
    net.load_state_dict(torch.load(PATH))

    os.makedirs('./image/', exist_ok=True)

    '''
    # change the url
    # test your model using a new online image 
    '''

    IMAGE_URL = "https://img1.baidu.com/it/u=43031557,4242621975&fm=26&fmt=auto&gp=0.jpg"
    filename = './image/img1.jpg'

    # calling urlretrieve function to get resource
    urllib.request.urlretrieve(IMAGE_URL, filename)

    image = Image.open('./image/img1.jpg')

    test_transform = transforms.Compose(
        [transforms.Resize([32, 32]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = test_transform(image)
    print(image.shape)
    image = image.unsqueeze(0)
    print(image.shape)

    outputs = net(image)
    _, predicted = torch.max(outputs, 1)
    imshow(torchvision.utils.make_grid(image))
    # print predicted labels
    print('Predicted: {}'.format(classes[predicted[0]]))