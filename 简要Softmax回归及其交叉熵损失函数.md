## 回归和分类的区别
- 回归估计一个连续值
- 分类预测一个离散类别

### 回归
- 单连续数值的输出
- 自然区间R
- 跟真实值的区间做损失

![回归](http://zh-v2.d2l.ai/_images/singleneuron.svg)

### 分类
- 通常多个输出
- 输出i是预测第i类的置信度
- 输出的个数等于类别的个数

![分类](http://zh-v2.d2l.ai/_images/softmaxreg.svg)

## 独热编码
既然是分类问题，很自然的我们就会想到，在之前的线性回归中，标签值y（真实y）是一个数。那么在分类问题中，我们用什么信息来表示一个类呢？同时还要要求不同的类编码不能混淆。深度学习分类分体采用了统计学家的一种分类数据的简单方法：**独热编码（one-hot encoding）**。  
独热编码的特性是：每个编码长度和所有的分类数量长度一致，类别对应的分量位置设置为1，其他所有分量设置为0。  
例如，有三个分类{狗、猫、鸡}，狗的标签是(1,0,0)、猫的标签是(0，1，0)、鸡的标签是(0,0,1)。  

## softmax运算本身

### 为什么需要softmax函数？
事实上，当我们拿到一个分类问题时，我们自然的会去将数据集中的不同分类打上one-hot标签。至此，数据流水线的问题已经得到了解决。现在我们来考虑一下计算损失前的事情。**很显然，分类问题的损失是真实标签与预测标签之间的一种度量。**  
考虑到，真实标签中，为1的类别直观上的含义是，这类事物是某事物的概率是100%，是其他事物的概率是0%，那么为了计算损失，很直观的想到一点，我们的预测标签也应当是一个概率。**那么，线性层的输出（神经网络的层多是线性层，最后一层多是全连接层）能否视为概率？**  
答案显然是否定的。为什么？
- 一是，线性层的输出之和并非为1。（真实标签的概率和是1，预测标签的概率是不是应该也为1？）
- 二是，线性层的输出可以是负数。（概率能为负数吗？）

### softmax函数干了什么事儿？
因此，我们必须保证任何数据的模型输出都是非负且总和为1。考虑一下，我们如何达到这一目的？  （**注意：oi不是yi，只是线性层的输出**）
- 总和为1：线性层的输出是有多少个分类就有多少个输出。例如，有3个分类，就是三个输出o1，o2，o3(假设全为正数). 于是（o1+o2+o3）/（o1+o2+o3）一定是1.而对于o1来说，他的线性输出就变成了概率（o1）/（o1+o2+o3）。
- 非负：我们上面是假设了线性层的呼出非负。但实际情况却是，线性层的输出是可能为负的，把负数变成负数很自然的能想到两种函数：绝对值和指数函数。在softmax函数中，softmax选择了后者，这样，对于上述的三分类问题，o1的概率就变成了exp(o1)/(exp(o1)+exp(o2)+exp(o3))。
- 我们可以将三分类推广至n分类，然后就得出了softmax函数如下：

对于一次输出中的某个分类的概率为： $\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$  
对于一次输出中的所有分类的概率向量$\hat{\mathbf{y}}$为：$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad$  
这样的话，我们我们可以看到，对于所有的j总有$0 \leq \hat{y}_j \leq 1$。因此，**softmax函数实际上输出的是关于所有分类的概率分布**。然而，softmax运算不会改变未规范化的预测o之间的顺序，只会确定分配给每个类别的概率。 因此，在预测过程中，我们仍然可以用下式来选择最有可能的类别。  
$\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.$

### 小批量样本的softmax
上面我们已经得到了一个样本输入的softmax公式。当模型的输入是一个样本的时候，softmax的输出是一个所有分类的概率分布，通常用向量表示。  
现在，我们将一个样本输入的softmax推广到小批量softmax。假设我们读取了一个批量的样本 $\mathbf{X}$ ， 其中特征维度（输入数量）为 $d$ ，批量大小为 $n$ 。 此外，假设我们在输出中有 $q$ 个类别。 那么小批量特征为 $\mathbf{X} \in \mathbb{R}^{n \times d}$， 权重为$\mathbf{W} \in \mathbb{R}^{d \times q}$， 偏置为$\mathbf{b} \in \mathbb{R}^{1\times q}$。 softmax回归的矢量计算表达式为：  
$\mathbf{O} = \mathbf{X} \mathbf{W} + \mathbf{b}$  
$\hat{\mathbf{Y}} = \mathrm{softmax}(\mathbf{O}).$

## 交叉熵损失函数
当我们通过上面的过程得到了预测概率后，我们就可以开始思考分类问题（softmax回归就是一个分类问题）的损失函数了。我们知道损失函数的本质是可以说是预测效果的度量，也可以是衡量真实标签与预测标签间的区别。**那么，对于两个概率分布，我们如何衡量概率分布间的区别？**  

### 单样本损失函数
我们先做如下符号定义：
- $\boldsymbol{y}^{(i)}$：样本i的真实标签向量
- $y^{(i)}$：样本i的真实标签向量$\boldsymbol{y}^{(i)}$中元素值取得1的位置，向量中**第**$y^{(i)}$号元素为1，其他为0.
- $y_j^{(i)}$：样本i的真实标签向量$\boldsymbol{y}^{(i)}$中的j分类位置指代的标量（概率），非1即0.
- $\boldsymbol{\hat y}^{(i)}$：样本i的预测标签向量（经过softmax后的预测概率分布）.
- $\hat y_j^{(i)}$：样本i的预测标签向量$\boldsymbol{\hat y}^{(i)}$中的j分类位置指代的标量（概率）.

我们可以像线性回归那样使用平方损失函数 $\|\boldsymbol{\hat y}^{(i)}-\boldsymbol{y}^{(i)}\|^2/2$ 。然而，想要预测分类结果正确，我们其实并不需要预测概率完全等于标签概率,我们只需要找到一个可以衡量两个概率密度分布差异的测量函数，这就是交叉熵函数（cross entropy）。交叉熵（cross entropy）是一个常用的衡量方法：  
$H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)},$   
于是，对于任意的真实标签$\boldsymbol{y}^{(i)}$和预测标签$\boldsymbol{\hat y}^{(i)}$的损失函数为：  
$l(\boldsymbol{y}^{(i)}, \boldsymbol{\hat y}^{(i)}) = H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)},$

因为，在向量$\boldsymbol{y}^{(i)}$中只有第$y^{(i)}$号元素为1，其余元素全为0.所以上式的求和自然就变成了：  
$l(\boldsymbol{y}^{(i)}, \boldsymbol{\hat y}^{(i)}) = H(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}) = -\log \hat y_{y^{(i)}}^{(i)}$  
也就是说，交叉熵只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。当然，遇到一个样本有多个标签时，例如图像里含有不止一个物体时，我们并不能做这一步简化。同时我们还应注意到，由于所有 $\hat{y}_j$ 都是预测的概率，所以它们的对数永远不会大于0。  

### 小批量损失函数
很显然，假设训练样本数为n，交叉熵损失函数定义为：  
$\ell(\boldsymbol{\Theta}) = \frac{1}{n} \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),$ 其中$\boldsymbol{\Theta}$代表模型参数。  
同样地，如果每个样本只有一个标签，那么交叉熵损失可以简化成:  
$\ell(\boldsymbol{\Theta}) = -(1/n) \sum_{i=1}^n \log \hat y_{y^{(i)}}^{(i)}$.

## 交叉熵损失函数的导数
**为简化下述对导数的讨论，我们将上述对于样本i的限制条件拿掉。** 即上述符号定义变更为：
- $\boldsymbol{y}$：某样本的真实标签向量
- $y_j$：某样本的真实标签向量$\boldsymbol{y}$中的j分类位置指代的标量（概率），非1即0.
- $\boldsymbol{\hat y}$：某样本的预测标签向量（经过softmax后的预测概率分布）.
- $\hat y_j$：某样本的预测标签向量$\boldsymbol{\hat y}$中的j分类位置指代的标量（概率）.

回顾一下，softmax函数和交叉熵损失函数的定义：  
$\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)}\\$  
$l(\boldsymbol{y}, \boldsymbol{\hat y}) = -\sum_{j=1}^q y_j \log \hat y_j \\$  
上式代入下式可推导：  
$l(\boldsymbol{y}, \boldsymbol{\hat y}) = - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
= \sum_{j=1}^q y_j (\log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j) \\
= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.$  
上述公式的推导，先利用了对数除法的性质，然后根据真实标签向量$\boldsymbol{y}$只有一个1，其他全为0推导得到。  
考虑相对于任何未规范化的预测 $o_j$ 的导数，我们可以得到导数：  
$\partial_{o_j} l(\boldsymbol{y}, \boldsymbol{\hat y}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.$  
换句话说，导数是我们softmax模型分配的概率与实际发生的情况（由独热标签向量表示）之间的差异。 从这个意义上讲，这与我们在回归中看到的非常相似， 其中梯度是观测值 $y$ 和估计值 $\hat y$ 之间的差异。


## 重新审视Softmax的实现
softmax函数 $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$ ， 其中 $\hat y_j$ 是预测的概率分布。  $o_j$ 是未规范化的预测o的第j个元素。 如果 $o_k$ 中的一些数值非常大， 那么 $exp(o_k)$ 可能大于数据类型容许的最大数字，即上溢（overflow）。 这将使分母或分子变为inf（无穷大）， 最后得到的是0、inf或nan（不是数字）的 $\hat y_j$ 。 在这些情况下，我们无法得到一个明确定义的交叉熵值。  
解决这个问题的一个技巧是： 在继续softmax计算之前，先从所有 ok 中减去 max(ok) 。 你可以看到每个 ok 按常数进行的移动不会改变softmax的返回值：  
$\begin{split}\begin{aligned}
\hat y_j =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\ = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}\end{split}$  
在减法和规范化步骤之后，可能有些 $o_j - \max(o_k)$ 具有较大的负值。 由于精度受限， $\exp(o_j - \max(o_k))$ 将有接近零的值，即下溢（underflow）。 这些值可能会四舍五入为零，使 $\hat y_j$ 为零， 并且使得 $\log(\hat y_j)$ 的值为-inf。 反向传播几步后，我们可能会发现自己面对一屏幕可怕的nan结果。  
尽管我们要计算指数函数，但我们最终在计算交叉熵损失时会取它们的对数。 通过将softmax和交叉熵结合在一起，可以避免反向传播过程中可能会困扰我们的数值稳定性问题。 如下面的等式所示，我们避免计算 $\exp(o_j - \max(o_k))$ ， 而可以直接使用 $o_j−max(o_k)$ ，因为 $\log(\exp(\cdot))$ 被抵消了。  
$\begin{split}\begin{aligned}
\log{(\hat y_j)} = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\ = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\ = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}\end{split}$  
我们也希望保留传统的softmax函数，以备我们需要评估通过模型输出的概率。 但是，我们没有将softmax概率传递到损失函数中， 而是在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数。

### 总结
1. **softmax定义函数在计算梯度时存在两个问题：**
    - **问题1：** 指数计算会使得softmax中的exp计算某些线性输出o后超出数据类型允许的值，导致上溢，使得分子或者分母变成inf，则最后softmax计算得到的概率是0、inf、nan，而0求交叉熵(取对数)是-inf，inf和nan无法求导。
    - **解决方法1：** 在计算softmax前，对线性层的所有输出ok减去max(ok)
    - **问题2：** 解决方法1虽然解决了问题1，但是会导致在某些线性输出中oj-max(ok)会具有较大的负值。而exp计算会使得负数接近0，受数据类型的精度限制，这个不为0的数会被四舍五入成为0，导致下溢。同样的，0求交叉熵（取对数）是-inf。反向传播几步后，会出现nan结果。
    - **解决办法2：** 对数运算和幂运算是可以互相抵消的运算。**softmax计算含有幂运算，计算交叉熵损失含有对数运算。**于是很自然的，通过将softmax和交叉熵结合在一起，可以避免反向传播过程中可能会困扰我们的数值稳定性问题（lim(x->0)logx=-inf）。 

2. **上述解决方式的优点：**
    - 解决了，softmax计算概率时由于精度的问题使得在计算交叉熵损失时出现的不可导的情况。
    - 同时还不影响softmax本身计算概率，因为，**我们通过解决办法2的方式计算损失并没有把softmax的计算结果作为输入，而是直接将线性层的结果作为输入，并不需要传统softmax函数来计算概率然后带入交叉熵函数求梯度。**

## 总结
- SoftMax回归是一个多类的回归模型
- 使用Softmax操作子得到每个类的预测置信度（概率）
- 使用交叉熵来衡量预测标签和真实标签的区别
