---
title: learning pytorch
date: 2017-05-12 22:14:12
tags:
- pytorch
- DL
---
PyTorch is a python package that provides two high-level features:
- Tensor computation (like numpy) with strong GPU acceleration
- Deep Neural Networks built on a tape-based autograd system

<!-- more -->

## 1. Install
添加环境变量
```
vim ~/.zshrc     
```
![6C12798DDE5163A414B2D388600BFFF2](/images/page4_images/6C12798DDE5163A414B2D388600BFFF2.jpg)
```
source ~/.zshrc      #重新加载环境变量
```
查看一下是否有conda
```
~ conda -V
conda 4.3.14
```
下载编译好的pytorch
```
conda install pytorch torchvision -c soumith
```
## 2. Deep Learning with PyTorch: A 60 Minute Blitz
### ① What is PyTorch?
It’s a Python based scientific computing package targeted at two sets of audiences:
- A replacement for numpy to use the power of GPUs
- a deep learning research platform that provides maximum flexibility and speed


#### Tensor
> Tensors are similar to numpy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

Tentor类似于numpy的ndarrays,另外还可以在GPU上使用Tensors来加速计算。下面进行实例演示：

创建一个未初始化的5x3的矩阵以及创建一个随机矩阵
![屏幕快照 2017-05-16 下午1.58.34](/images/page4_images/屏幕快照 2017-05-16 下午1.58.34_phcs2ol7r.png>)


我们可以得到X矩阵的尺寸, **注意** 这里的`torch.Size`实际上就是一个元组(tuple),所以它支持与元组相同的计算。

```python
>>> print(x.size())
torch.Size([5, 3])
```
关于加法的几种语法：
![加法语法](/images/page4_images/加法语法.png)

给出一个Tensor形式的输出：
![屏幕快照 2017-05-16 下午2.21.06](/images/page4_images/屏幕快照 2017-05-16 下午2.21.06.png>)

> 注：对一个Tensor变量使用`_`后缀会改变这个Tensor。例如`x.copy_(y),x.t_()`，这都会改变`x`。

#### Numpy Bridge
torch Tensor和numpy array可以互相转换，并且它们共享内存位置，一个改变另一个会随之改变，以下给出例子：
Converting torch Tensor to numpy Array
![屏幕快照 2017-05-16 下午5.22.23](/images/page4_images/屏幕快照 2017-05-16 下午5.22.23.png>)
onverting numpy Array to torch Tensor
```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
>>> print(a)
[ 2.  2.  2.  2.  2.]
>>> print(b)

 2
 2
 2
 2
 2
[torch.DoubleTensor of size 5]
```
> All the Tensors on the CPU except a CharTensor support converting to NumPy and back.


### ② Autograd: automatic differentiation
在PyTorc中，所有神经网络的中心都在`autograd`包里，`autograd`对Tensors中所有的操作都提供了自动微分。

#### Variable
`autograd.Variable `是包的中心类，它是一个Tensors包，当你完成计算的时候你可以调用`.backward()`就可以自动的计算所有的梯度。
当Variable用`.grad`计算得梯度时，你可以通过`.data`属性获得一个未处理的tensor。

自动梯度的执行还有一个非常重要的类是`Function`。`Variable`和`Function`是相互关联的共同构建了一个非循环图，它们可以编码一个完整的计算历史记录。每一个`variable`都有一个`.creator`属性，这个属性依赖于`Function`，`Function`创建了`Variable`(除去Varaiable通过用户创建，此时它们的`creator is None`)。

如果你想计算导数，你可以调用`Variable`中的`.backward()`。如果`Variable`是纯量(例如它只有一个元素)，你不需要为`.backward()`指定任何参数，如果有很多元素，你需要指定一个`grad_output`参数。

```python
import torch
from torch.autograd import Variable
# Create a variable:
x = Variable(torch.ones(2, 2), requires_grad=True)
>>>print(x)
Variable containing:
 1  1
 1  1
[torch.FloatTensor of size 2x2]
>>> torch.ones(2,2)

 1  1
 1  1
[torch.FloatTensor of size 2x2]
```
对variable做一些操作，y是执行的结果所以它有一个creator。
![屏幕快照 2017-05-17 下午1.32.18](/images/page4_images/屏幕快照 2017-05-17 下午1.32.18.png>)


#### Gradients
现在来做backprop,`out.backward`等同于`out.backward(torch.Tensor([1.0]))`。
```python
>>>out.backward()
#print gradients d(out)/dx
>>>print(x.grad)
Variable containing:
 4.5000  4.5000
 4.5000  4.5000
[torch.FloatTensor of size 2x2]
```
最后还有一个例子：
![屏幕快照 2017-05-17 下午2.17.24](/images/page4_images/屏幕快照 2017-05-17 下午2.17.24.png>)


### ③ Neural Networks
可以用`torch.nn`包来构建神经网络。`nn`依赖于`autograd`来定义模型并且对它们进行微分操作。`nn.Module`包含网络层的设置，`forward(input)`返回的是`output`。

神经网络典型的训练步骤如下：

- 定义神经网络的学习参数或者权重参数
- 输入数据集的迭代次数
- 通过网络处理输入
- 计算损失
- 计算网络参数的反向传播梯度
- 更新网络的权重`weight = weight + learning_rate * gradient`

#### Define the network
```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

OUT:
Net (
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear (400 -> 120)
  (fc2): Linear (120 -> 84)
  (fc3): Linear (84 -> 10)
)
```
你只需要定义`forward`函数，`backward`函数使用`autograd`自动定义，你可以在`forward`函数中使用任何的Tensor操作。

模型的学习参数通过`net.parameters()`返回
```python
>>> params = list(net.parameters())
>>> print(len(params))
10
>>> print(params[0].size())    #cov1's .weight
torch.Size([6, 1, 5, 5])
```
forward的输入是一个`autograd.Variable`，所以是可输出的
```python
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)
>>>Variable containing:
 0.0155  0.1107  0.0759 -0.0685  0.0056 -0.1105  0.0546  0.0828 -0.0309 -0.0636
[torch.FloatTensor of size 1x10]
```
将所有参数的梯度缓存置0，然后计算反向传播的随机梯度。
```python
>>> net.zero_grad()
>>> out.backward(torch.randn(1,10))
>>> print(input.grad)
None
```
> **注：** `torch.nn` only supports mini-batches.The entire `torch.nn` package only supports inputs that are a mini-batch of samples, and not a single sample.
For example, `nn.Conv2d` will take in a 4D Tensor of `nSamples x nChannels x Height x Width`.If you have a single sample, just use `input.unsqueeze(0)` to add a fake batch dimension.如果只有一个样本，你要使用这个函数来添加一些冒充的样本。


#### Loss Function
损失函数计算输出和目标值得差距。nn包里有几个[损失函数](http://pytorch.org/docs/nn.html#loss-functions)，A simple loss is: `nn.MSELoss` which computes the mean-squared error between the input and the target.
```python
output = net(input)
target = Variable(torch.range(1, 10))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
>>>print(loss)
Variable containing:
 38.3968
[torch.FloatTensor of size 1]
```

因此，当我们调用`loss.backward()`,关于loss的所有graph被微分,graph中所有的Variable通过梯度计算都将有他们的`.grad`Variable。
```python
print(loss.creator)  # MSELoss
print(loss.creator.previous_functions[0][0])  # Linear
print(loss.creator.previous_functions[0][0].previous_functions[0][0])  # ReLU
```
OUT :
```python
<torch.nn._functions.thnn.auto.MSELoss object at 0x7fec94050588>
<torch.nn._functions.linear.Linear object at 0x7fec940504a8>
<torch.nn._functions.thnn.auto.Threshold object at 0x7fec94050048>
```

#### Backprop

To backpropogate the error我们需要做的是     `loss.backward()`。你需要清空已经存在的梯度值, 否则现有的梯度值就会被计算进来。Now we shall call `loss.backward()`, and have a look at conv1’s bias gradients before and after the backward.
```python
>>> net.zero_grad()
>>> print('conv1.bias.grad before backward')
conv1.bias.grad before backward
>>> print(net.conv1.bias.grad)
Variable containing:
 0
 0
 0
 0
 0
 0
[torch.FloatTensor of size 6]
>>> loss.backward()
>>> print(net.conv1.bias.grad)
Variable containing:
-0.0346
-0.1167
 0.0691
 0.0270
 0.0682
 0.0790
[torch.FloatTensor of size 6]
```

#### Update the weights
The simplest update rule used in practice is the Stochastic Gradient Descent (SGD随机梯度下降):
```python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

当你使用神经网络的时候，你想使用不同的更新规则例如SGD,Nesterov-SGD,Adam,RMSProp等。 To enable this, 我们构造了一个包`torch.optim` 来执行所有的这些方法。 Using it is very simple:
```python
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
>>> net.conv1.bias.data

1.00000e-02 *
 -8.1654
 -7.9709
 -8.6368
  1.4395
 -6.3513
  9.4039
[torch.FloatTensor of size 6]
```

### ④ Training a classifier

### What about data?

实际上当你处理照片、文本、声音或者影像数据时，你可以使用标准的python安装包，它会将数据下载成numpy array。然后你可以将array转换成`torch.*Tensor`。
- For images, packages such as Pillow, OpenCV are useful.
- For audio, packages such as scipy and librosa
- For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful.


Specifically for `vision`, we have created a package called `torchvision`, that has data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz., `torchvision.datasets` and t`orch.utils.data.DataLoader`.

在这个教程中，我们使用了CIFAR10数据集. 它包含: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

Now we should...
**Training an image classifier**

1. Load and normalizing the CIFAR10 training and test datasets using torchvision

```python
import torch
import torchvision
import torchvision.transforms as transforms
#The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

2. Define a Convolution Neural Network

将上一节的神经网络cp过来，修改为3通道输入。
```python
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```
3. Define a Loss function and optimizer

```python
import torch.optim as optim
#交叉熵损失函数和SGD梯度下降方法
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```
4. Train the network on the training data

```python
for epoch in range(2):  # loop(遍历) over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```
5. Test the network on the test data

我们进行了两次迭代，下面我们看一下这个网络是否学到了一些东西。
```python
# 1 first step. Let us display an image from the test set to get familiar.
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 2 second step.what the neural network thinks these examples above are
outputs = net(Variable(images))
# 3 The outputs are energies for the 10 classes. Higher the energy for a class, the more the network thinks that the image is of the particular class. So, let’s get the index of the highest energy:

_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j][0]]
                              for j in range(4)))

# 4 then let us look at how the network performs on the whole dataset.
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# 5 what are the classes that performed well, and the classes that did not perform well
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

以上就是基础内容介绍，下面我们来看一些例子：

## 3. pytorch-tutorial
[00 - PyTorch Basics](/Users/kaikaidexiaomaike/project/myblog/source/pytorchtst/00 PyTorch Basics.ipynb)
[01 - Linear Regression](/Users/kaikaidexiaomaike/project/myblog/source/pytorchtst/01 Linear Regression.ipynb)
[02 - Logistic Regression](/Users/kaikaidexiaomaike/project/myblog/source/pytorchtst/02 Logistic Regression.ipynb)
[03 - Feedforward Neural Network](/Users/kaikaidexiaomaike/project/myblog/source/pytorchtst/03 Feedforward Neural Networ.ipynb)
[04 - Convolutional Neural Network](/Users/kaikaidexiaomaike/project/myblog/source/pytorchtst/04 Convolutional Neural Network.ipynb)
[05 - Deep Residual Network](/Users/kaikaidexiaomaike/project/myblog/source/pytorchtst/05 Deep Residual Network.ipynb)
[06 - Recurrent Neural Network](/Users/kaikaidexiaomaike/project/myblog/source/pytorchtst/06 Recurrent Neural Network.ipynb)
07 - Bidirectional Recurrent Neural Network
08 - Language Model
09 - Image Captioning
10 - Generative Adversarial Network
11 - Deep Convolutional Generative Adversarial Network
12 - Deep Q Network


## Reference
[1] [pytorch官方网站](http://pytorch.org/)
[2] [PyTorch documentation](http://pytorch.org/docs/index.html)
[2] [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)




Thanks ：）
