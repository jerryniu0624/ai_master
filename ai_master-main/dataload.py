# import numpy as np
# import h5py
  
# mat = h5py.File('./data2/Dataset.mat')
  
# print(mat['images'].shape)#查看mat文件中images的格式
#   #(2284, 3, 640, 480)
  
# images = np.transpose(mat['images'])
#  #转置，images是numpy.ndarray格式的
 
# print(images)#控制台输出数据
# print(images.shape)#输出数据格式
#  #(480, 640, 3, 2284)
 
# np.save('./images', images)#保存数据，会生成一个images.npy文件
import scipy.io as sio
import numpy as np

yFile = './data2/Dataset.mat'    #相对路径
datay=sio.loadmat(yFile)

print (datay)
# print (datay['train'],datay['train'].shape)
# # (200, 15, 28, 28)
# print (datay['test'],datay['test'].shape)
# # (200, 5, 28, 28)
# print (datay['images'],datay['images'].shape)
np.save('',datay)