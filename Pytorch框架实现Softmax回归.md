通过深度学框架的高级API能够使实现softmax回归变得更加容易。通常采用pytorch的nn.Moudule实现。
## 导入包和定义数据流


```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

    /home/tangger/anaconda3/envs/pt/lib/python3.8/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448278899/work/torch/csrc/utils/tensor_numpy.cpp:180.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


## 初始化模型参数
softmax回归的输出层是一个全连接层。 因此，为了实现我们的模型， 我们只需在Sequential中添加一个带有10个输出的全连接层。 同样，在这里Sequential并不是必要的， 但它是实现深度模型的基础。 我们仍然以均值0和标准差0.01随机初始化权重。


```python
# PyTorch不会隐式地调整输入的形状。因此，
# 因此，我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```


    Sequential(
      (0): Flatten(start_dim=1, end_dim=-1)
      (1): Linear(in_features=784, out_features=10, bias=True)
    )

## 交叉熵损失函数


```python
loss = nn.CrossEntropyLoss()
```

## 优化算法
在这里，我们使用学习率为0.1的小批量随机梯度下降作为优化算法。


```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

## 训练


```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```


![svg](https://cdn.jsdelivr.net/gh/tangger2000/PicHost/img/20220105234604.svg)
    

