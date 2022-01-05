## 导入包和定义数据流


```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 定义权重和偏置矩阵

原始数据集中的每个样本都是 28×28 的图像。 但是对于softmax无法处理空间信息，这将在后面的卷积神经网络部分学习，于是，我们将展平每个图像，把它们看作长度为784的向量。   
在softmax回归中，我们的输出与类别一样多。 因为我们的数据集有10个类别，所以网络输出维度为10。 因此，权重将构成一个 784×10 的矩阵， 偏置将构成一个 1×10 的行向量。 与线性回归一样，我们将使用正态分布初始化我们的权重W，偏置初始化为0。


```python
num_inputs = 784
num_outputs = 10

# 高斯随机
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
W.shape, b.shape
```


    (torch.Size([784, 10]), torch.Size([10]))



## 实现softmax
实现softmax的三个步骤为：
1. 对每个项求幂（使用exp）；
2. 对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
3. 将每一行除以其规范化常数，确保结果的和为1。

回顾softmax的公式如下：
$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$


```python
def softmax(X):
    X_exp = torch.exp(X)
    # 矩阵中每一行为一个样本，所以对特征按行求和
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
```

用下列例子，验证上述代码是否正确


```python
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)
```


    (tensor([[0.2057, 0.3452, 0.0756, 0.1039, 0.2696],
             [0.0183, 0.1502, 0.3175, 0.0487, 0.4654]]),
     tensor([1.0000, 1.0000]))



## 实现softmax回归模型
softmax回归模型是：O=XW+b, y=softmax(O)


```python
# X.shape[batch_size,28,28] -(reshape)-> [batch_size, 784]
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```



## 实现交叉熵损失函数

交叉熵采用真实标签的预测概率的负对数似然。我们通过一个运算符选择所有元素。 例如，下面创建一个数据样本y_hat，其中包含2个样本在3个类别的预测概率， 以及它们对应的标签y。 有了y，我们知道在第一个样本中，第一类是正确的预测； 而在第二个样本中，第三类是正确的预测。 然后使用y作为y_hat中概率的索引， 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。


```python
y = torch.tensor([0, 2])
print(y)
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat)
print(y_hat[[0, 1], y])

# 如果看不懂上一句的代码意思，看下面的分解：
print('如果看不懂上一句的代码意思，看下面的分解：')
print('index',y_hat[[0, 1], [0, 2]])
print('[0,0]',y_hat[0, 0])
print('[1,2]',y_hat[1,2])
```

    tensor([0, 2])
    tensor([[0.1000, 0.3000, 0.6000],
            [0.3000, 0.2000, 0.5000]])
    tensor([0.1000, 0.5000])
    如果看不懂上一句的代码意思，看下面的分解：
    index tensor([0.1000, 0.5000])
    [0,0] tensor(0.1000)
    [1,2] tensor(0.5000)


现在我们只需一行代码就可以实现交叉熵损失函数。


```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y), y_hat, y
```


    (tensor([2.3026, 0.6931]),
     tensor([[0.1000, 0.3000, 0.6000],
             [0.3000, 0.2000, 0.5000]]),
     tensor([0, 2]))



## 分类精度
给定预测概率分布y_hat，当我们必须输出硬预测（hard prediction）时， 我们通常选择预测概率最高的类。当预测与标签分类y一致时，即是正确的。**分类精度即正确预测数量与总预测数量之比。** 虽然直接优化精度可能很困难（因为精度的计算不可导）， 但精度通常是我们最关心的性能衡量标准，我们在训练分类器时几乎总会关注它。  
为了计算精度，我们执行以下操作。 首先，如果y_hat是矩阵，那么假定第二个维度存储每个类的预测分数。 我们使用argmax获得每行中最大元素的索引来获得预测类别。 然后我们将预测类别与真实y元素进行比较。 由于等式运算符“==”对数据类型很敏感， 因此我们将y_hat的数据类型转换为与y的数据类型一致。 结果是一个包含0（错）和1（对）的张量。 最后，我们求和会得到正确预测的数量。  


```python
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        print(y_hat)
    cmp = y_hat.type(y.dtype) == y
    print(cmp)
    return float(cmp.type(y.dtype).sum())
```


```python
accuracy(y_hat, y) / len(y)
```

    tensor([2, 2])
    tensor([False,  True])

    0.5



## 评估模型的精度
对于任意数据迭代器data_iter可访问的数据集， 我们可以评估在任意模型net的精度。  
这里定义一个实用程序类Accumulator，用于对多个变量进行累加。 在上面的evaluate_accuracy函数中， 我们在Accumulator实例中创建了2个变量， 分别用于存储正确预测的数量和预测的总数量。 当我们遍历数据集时，两者都将随着时间的推移而累加。  


```python
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

由于我们使用随机权重初始化net模型， 因此该模型的精度应接近于随机猜测。 例如在有10个类别情况下的精度为0.1。


```python
evaluate_accuracy(net, test_iter)
```


    0.0862



## 训练
首先，我们定义一个函数来训练一个迭代周期。 请注意，updater是更新模型参数的常用函数，它接受批量大小作为参数。 它可以是d2l.sgd函数，也可以是框架的内置优化函数。


```python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.sum().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

《动手学深度学习》一书在此定义了一个在动画中绘制数据的实用程序类Animator。


```python
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

接下来我们实现一个训练函数， 它会在train_iter访问到的训练数据集上训练一个模型net。 该训练函数将会运行多个迭代周期（由num_epochs指定）。 在每个迭代周期结束时，利用test_iter访问到的测试数据集对模型进行评估。 我们将利用Animator类来可视化训练进度。  


```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

## 随机梯度下降优化函数
小批量随机梯度下降来优化模型的损失函数，设置学习率为0.1。


```python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

## 开始训练
现在，我们训练模型10个迭代周期。 请注意，迭代周期（num_epochs）和学习率（lr）都是可调节的超参数。 通过更改它们的值，我们可以提高模型的分类精度。


```python
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```


![svg](https://cdn.jsdelivr.net/gh/tangger2000/PicHost/img/20220105234300.svg)
    


## 预测
给定一系列图像，我们将比较它们的实际标签（文本输出的第一行）和模型预测（文本输出的第二行）。


```python
def predict_ch3(net, test_iter, n=20):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```


​    
![svg](https://cdn.jsdelivr.net/gh/tangger2000/PicHost/img/20220105234310.svg)
​    

