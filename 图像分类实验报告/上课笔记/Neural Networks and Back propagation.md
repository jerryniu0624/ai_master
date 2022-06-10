# Neural Networks and Back propagation

## 前向传播

## 反向传播

求导->不停沿着梯度的反方向调参数

## Learning Rate

choose proper step size

![IMG_20210707_165147](D:\周董\MobileFile\IMG_20210707_165147.jpg)



too large to get the 全局最小值



![1625648091053](D:\周董\MobileFile\1625648091053.jpg)

![1625648170464](D:\周董\MobileFile\1625648170464.jpg)

Loss function

n: the number of samples

addition: add each image's loss



n too large to update

![1625648382182](D:\周董\MobileFile\1625648382182.jpg)

左图：虽然每次不准，但是快

![1625648550742](D:\周董\MobileFile\1625648550742.jpg)

velocity: 历史梯度

gradient:当前梯度

按权重求和更新actual step

![1625648725367](D:\周董\MobileFile\1625648725367.jpg)

加momemtum以避免进入局部最小值



如果反向梯度由多于1的数据组成，则可以求和作为反向梯度



若sample非线性可分

![1625649521423](D:\周董\MobileFile\1625649521423.jpg)

## Activation functions

e.g. sigmoid, relu , tanh



饱和函数：本身梯度很可能0，之前的层的权重无用

恒正，所以update方向固定

![1625650022298](D:\周董\MobileFile\1625650022298.jpg)



![1625650177376](D:\周董\MobileFile\1625650177376.jpg)

0点不可导

![1625650467830](D:\周董\MobileFile\1625650467830.jpg)



![1625650542425](D:\周董\MobileFile\1625650542425.jpg)

层数是hyperparameter

一般来说，任务越简单，层数越少



