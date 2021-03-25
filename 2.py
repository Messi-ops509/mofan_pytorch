# 有关variable
# 张量与变量，目前版本一致了，
import torch
from torch.autograd import Variable

#定义一个张量数据, 中括号加中括号,外面还有一个大括号   将张量变为变量形式 requires_grad = True与反向传播有关
tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad = True)

print(tensor)
print(variable)

# v=x*x
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

#反向传播 X（variable）相当于参数w，然后(grad)求导，求梯度
v_out.backward()
print(variable.grad)

#打印变量，打印张量，打印numpy数据
print(variable)
print(variable.data)
print(variable.data.numpy())

