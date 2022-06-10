# machine learning

written by 牛远卓

## supervised learning

data with label 

classification

regression



## unsupervised learning

data without label

clustering（聚类）

e.g. K-means



## semi

first use the data with label to train the model

then label the data without label

and train the model with both of them



## basic concept

underfitting 欠拟合

overfitting 过拟合



hyperparameter 超参数

validation set 验证集

用来选择超参数



regularization正则化



### classifier: nearest neighbor

memorize

predict

```python
import numpy as np

class NearestNeighbor:
	def _init_(self):
        pass
    
    def train(self,X,y):#memorize
        self.Xtr = X
        self.Ytr = Y
   	
    def predict(self,X):#predict
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        
        for i in xrange(num_test):#离中点最近的
		distances = np.sum(np.abs(self.Xtr - X[i,:]),axis = 1)
        min_index = np.argmin(distance)
        Ypred[i] = self.ytr[min_index]
        
    return Ypred
```

![IMG_20210705_150830](D:\周董\MobileFile\IMG_20210705_150830.jpg)

![IMG_20210705_152831](D:\周董\MobileFile\IMG_20210705_152831.jpg)

### k-neighbor





## loss function

### e.g. SVM

Loss = max{0,sj-si+1}



![IMG_20210705_154212](D:\周董\MobileFile\IMG_20210705_154212.jpg)



### e.g. softmax classifier

want to interpret raw classifier scores as probabilities



![IMG_20210705_161816](D:\周董\MobileFile\IMG_20210705_161816.jpg)



### regularization（正则化）

lower the loss function

Meanwhile, to prevent the model from doing too well on training data we need regularization

- express preference over weights
- make the model simple so it works on test data
- improve optimization by adding curvature(not taught about)

![IMG_20210705_161418(1)(1)](D:\周董\MobileFile\IMG_20210705_161418(1)(1).jpg)

![](D:\周董\MobileFile\IMG_20210705_161418(1).jpg

![IMG_20210705_161548](D:\周董\MobileFile\IMG_20210705_161548.jpg)



## optimization

to minimize the loss

e.g. choose parameter randomly and choose the best parameter by the lest loss

![IMG_20210705_162534](D:\周董\MobileFile\IMG_20210705_162534.jpg)

this method is too simple and function poorly





## full connect layer

convolutional  layer（卷积层）

Filters always extend the full depth of the input volume

convolve the filter with the image

e.g. slide over the image spatially, computing dot products

![IMG_20210705_170132](D:\周董\MobileFile\IMG_20210705_170132.jpg)





```python
import torch
import numpy as np
arr=np.ones((4,3))
print("arr的数据类型为："+str(arr.dtype))
t=torch.tensor(arr)
print(t)
```



## pooling

池化

e.g. maxpooling



## padding

由于输入输出的数据大小要求，又是要求在原输入数据的周围一圈补数据
