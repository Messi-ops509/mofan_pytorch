#pytorch与numpy对比
#1加载torch,numpy
import torch
import numpy as np

#2numpy形式数据，0-5 两行三列    转换为torch数据    再转换为numpy数据
np_data = np.arange(6).reshape(2,3)
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy

#3打印
print(
    '\nnumpy',np_data,
    '\ntorch',torch_data,
    '\ntensor2array',tensor2array,
)

#4torch数据用法,abs(绝对值) sin mean
data = (-1,-2,1,2)
tensor = torch.FloatTensor(data)


print(
    '\nabs',
    '\nnumpy',np.abs(data),
    '\ntorch',torch.abs(tensor),
)

data1 = (-1,-2,1,2)
tensor1 = torch.FloatTensor(data1)


print(
    '\nsin',
    '\nnumpy',np.sin(data1),
    '\ntorch',torch.sin(tensor1),
)

data2 = (-1,-2,1,2)
tensor2 = torch.FloatTensor(data2)


print(
    '\nmean',
    '\nnumpy',np.mean(data2),
    '\ntorch',torch.mean(tensor2),
)


#5矩阵形式 矩阵相乘
data4 = [(1,2),(1,2)]
tensor4 = torch.FloatTensor(data4)

print(
    '\n',
    '\nnumpy',np.matmul(data4,data4),
    '\ntorch',torch.mm(tensor4,tensor4),
)

#矩阵中dot  np中dot是相乘，tensor不行（错误）
data5 = [(1,2),(1,2)]
tensor5 = torch.FloatTensor(data5)
data5 = np.array(data5)

print(
    '\n',
    '\nnumpy',data5.dot(data5),
    '\ntorch',tensor5.dot(tensor5),
)
