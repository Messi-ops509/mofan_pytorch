#神经网络代码  激活函数部分
# nn代表神经网络部分，
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

#假定数据 从-5到5的200个数据 张量-变量-numpy numpy是为了画图
x = torch.linspace(-5,5,200)
x = Variable(x)
x_np = x.data.numpy()

#激活函数 softmax也是激活函数，但是用于概率方面
y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

#画图
plt.figure(1,figsize=(6,8))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((1,5))
plt.legend(loc='best')

plt.show()