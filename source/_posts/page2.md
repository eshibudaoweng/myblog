---
title: What makes ImageNet good for transfer learning
date: 2017-03-27 09:53:55
tags:
- papers
- fine-grained
- transfer learning
---
**Task**:谈谈[2016_arxiv_What makes ImageNet good for transfer learning](/papers/2016_arxiv_What makes ImageNet good for transfer learning.pdf)这一篇文章对做fine-grained的启发。
<!-- more -->
首先提出了一个问题，学习一个好的、通用的特征对一个ImageNet数据集来说至关重要的是哪些内在属性？本论文对以下几个方面做了测试：
- more pre-training
- more object classes
- split the data into classes
- coarse classes or fine-grained classes
- more classes or more examples per class

## 1 摘要
解决各种计算及视觉问题的实际操作目前可以分为两步

1. 首先训练一个CNN对ImageNet做图像分类（pre-training）
2. 应用这些特征到一个新的目标任务（fine-tuning）

目前，使用这些ImageNet预训练CNN特征在以下几项工作中取得良好的结果：

- 图像分类(image classification)
- 目标检测(object detection)
- 行为识别(action recognition)
- 人体姿态估计(human pose estimation)
- 图像分割(image segmentation)
- 光流(optical flow)

论文总结了以下几个工作：
1. 迁移学习需要多少预训练ImageNet样本才足够？预训练只用ImageNet数据的一半数量（每个类别500个图像，而不是1000个图像），结果发现迁移学习效果只有小幅度的下降，该下降远小于分类任务本身的下降。--> Section 4 and Figure 1
2. 迁移学习需要多少预训练ImageNet类才足够？预训练所用类的减少仅仅对迁移学习的效果造成了小幅度的下降。因吹斯挺，对于一些迁移任务，预训练更少的类可以带来更好的效果。-->Section 5.1 and Figure 2
3. 迁移学习中，学习到好的特征对细粒度识别有多重要？-->Section 5.2 and Figure 2
4. 给定相同的预训练图像的预算，我们需要需要更多的类还是每个类更多的图片？训练更少的类每类更多的图片的效果要比训练少的类每类更多的图片好。-->Section 5.5 and Table 2
5. 数据集数量越多就越好吗？并不是。-->Section 6, and Table 9

## 2 相关工作
1. 理解CNN的内部表征
2. 影响微调的因素
3. 其它的预训练方法

## 3 实验设置
>pre-training:使用监督学习初始化CNN参数的用于ImageNet分类任务过程称为预训练。
>finetuning：通过对目标数据集的持续训练来调整预训练CNN的过程称为finetuning

**实验框架**：AlexNet
**实验设置和代码**：Faster- RCNN
**实验微调**：首先替换AlexNet的FC-8层为一个随机初始化，全连接层有378个输出单元。使用随机梯度下降( stochastic gradient descen)进行50K次迭代，初始学习率为0.001，每20K次迭代减少10倍（1/10）。
**实验进程**：每次微调进行三次试验，每次实验只跑一次，用 mean ± standard deviation来总结效果。

## 4 预训练样本的数量是如何影响迁移性能的
ImageNet分类任务（预训练任务）的性能随着训练数据量而不断增加，而在迁移任务上，增加预训练样本数量，性能提升明显更慢。

## 5 预训练分类任务如何影响迁移性能

### 5.1预训练类的数量对迁移性能的影响
有大量的类别用来训练是有用的。但753和918比1000类的性能好一些，侧面说明有时候太多的会对获取好的全局特征比例，是不是标签集的原因呢？

### 5.2 细粒度分类对学前迁移特征重要吗
marginally helpful

### 5.3 用粗类训练产生的特征与细粒度识别的相关性
虽然用粗类来训练，但CNN隐含地去发现能够区分更精细的类的特征。

### 5.4 用细类训练产生的特征与粗类识别的相关性
使类标签更多的考虑视觉上共同之处而不是简单地WordNet语义。

### 5.5更多的类好还是每一类更多的样本好
后者性能稍微好一点

### 5.6 预训练的类也用于目标任务的重要性
如果预训练和迁移任务的类是通用的，性能会有所提升。

## 6 非目标类数据量的增加会提高性能吗
not good.

## Conclusion
总的来说这篇论文做个几个实验总结了影响迁移学习效果的主要因素有什么，可以依据此来提高迁移学习任务的效果。
~~其实觉得并没有什么用~~
Thanks:)
