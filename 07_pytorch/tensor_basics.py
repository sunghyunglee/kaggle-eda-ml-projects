import torch
import numpy as np

#torch 기본
# x = torch.tensor([[1., 2., 3], [4., 5., 6.], [7., 8., 9.]])
# y = torch.tensor([[11., 12., 13.], [14., 15., 16.], [17., 18., 19.]])
#
# print(x+y)
# print(x*y)
# print(x.mean())
# print(x.sum())
#
#
# a = torch.zeros(2, 3)
# b = torch.rand(2,3)
# print(a)
# print(b)



#텐서를 GPU 보내기
# abc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# x = torch.rand(2, 3)
# y = torch.rand(2, 3)
# print(x.device)
# print(y.device)
#
#
# x = x.to(abc)
# y = y.to(abc)
# print(x.device)
# print(y.device)
#
# print(x+y)




#뉴런
# x = torch.tensor([1., 2., 3.])
# w = torch.tensor([0.2, 0.8, -0.5])
# b = torch.tensor(0.1)
# print(w.shape)
#
# z = (x*w).sum()+b
# print(z.item())
#
# y = torch.sigmoid(z)
# print(z)
# print(z.shape)
# print(y.item())


#뉴런(층) 쌓기
# x = torch.tensor(([1., 2., 3.]))
# w = torch.rand(3,4)
# b = torch.rand(4)
#
# z = x @ w +b
#
# y = torch.relu(z)
#
# print(z)
# print(z.shape)
# print(y)
# print(w.shape)


#활성화 함수
z = torch.tensor([-2.,-1.,0,1.,2.])
print("sigmoid: ", torch.sigmoid(z))
print("relu: ", torch.relu(z))
print("tanh: ", torch.tanh(z))

