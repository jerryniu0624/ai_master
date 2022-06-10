import torch
import torchvision
from torchvision import transforms
from model import Net1,Net2,Net3
from ResNet import ResNet18
from AlexNet import AlexNet
#net=argv[1]


import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model",type=int,default=0)


args = parser.parse_args()


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    '''
    # try to set different batch_size
    '''
    batch_size = args.batch_size

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {}%'.format(
        100 * correct / total))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))