import numpy as np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model",type=int,default=0)
args = parser.parse_args()


def draw_loss(train_ls_sum_l,test_ls_sum_l):
    x=range(0,len(train_ls_sum_l))
    plt.title("loss of the Rnn") 
    plt.xlabel("training times") 
    plt.ylabel("loss") 
    plt.plot(x,train_ls_sum_l)
    plt.plot(x,test_ls_sum_l)
    plt.legend(['y=train loss','y=test loss']) # 添加图例
    plt.savefig('./loss.jpg')
    plt.show()
    plt.close()


def draw_acc(train_acc_sum_l,test_acc_sum_l):
    x=range(0,len(train_acc_sum_l))
    plt.title("accuracy of the Rnn") 
    plt.xlabel("training times") 
    plt.ylabel("accuracy") 
    plt.plot(x,train_acc_sum_l)
    plt.plot(x,test_acc_sum_l)
    plt.legend(['y=train accuracy','y=test accuracy']) # 添加图例
    
    plt.savefig('./accuracy.jpg')
    plt.show()
    plt.close()


if __name__ == '__main__':

    name1='loss'+'_model:'+str(args.model)
    name2='acc'+'_model:'+str(args.model)
    loss1=np.load(name1+'.npy')
    print(loss1)
    
    acc=np.load(name2+'.npy')
    print(acc)
    draw_loss(loss1,name1)
    draw_acc(acc,name2)
