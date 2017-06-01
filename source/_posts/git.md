---
title: How to use git
date: 2017-03-19 13:57:12
tags:
- git
---
从git小白做起，学一点记一点。
毕竟不知道GitHub的coder只是一个code搬运工。。。
<!-- more -->

## Q1:Add existing project to github
如何将一个本地维护的项目，转换为一个git项目，并托管到GitHub上。
1.进入项目所在的本地目录，将目录初始化为一个Git项目
```
$ git init
```
2.GitHub官方声明的第二步让添加一个README.md，但是后续也可以添加，这里就随便啦。
```
$ git add README.md
```
3.本地提交
```
$ git add .     将所有文件放在新的本地git仓库
$ git commit -m "first commit"    
```
> 注：如果本地已经有`.gitignore`文件，会按照已有规则过滤掉不需要的文件。如果不想添加所有的文件，把`.`替换成具体的文件名。`first commit`是对应的代码提交信息。

4.进入自己的GitHub网站，创建一个新的仓库

> 为了避免冲突，先不要勾选`README`和`LICENSE`选项。

5.在生成的项目主页上有仓库的地址，复制
![图1](/images/仓库地址.png)

6.回到terminal,添加远程关联,将本地仓库关联到远程仓库
```
$ git remote add origin https://github.com/eshibudaoweng/myblog.git  
$ git remove -v    查看状态
```
7.提交代码到GitHub仓库
```
$ git push origin master
```
**参考** GitHub官方教程
![图2](/images/GitHub截图1.png)



## Q2:
> git status    查看状态
> 恢复修改之前的状态 git checkout -- file.md

> 创建一个分支 git checkout -b madifix


> 在分支里任意操作，例如创建一个文件 git add hello.txt

> 对你这次的提交添加一个注释 git commit -m "注释"

> 切换到另一个分支 git checkout master

> 打印提交记录 git log

> 查看源码的远程仓库 git remote -v

> 将你的github网站添加进来 git remote add origin https://github.com/eshibudaoweng/part_model.git

> 提交你的代码 git push -u origin master

> 将已经存在的远程仓库移除 git remote remove origin

## Reference:
[GitHub官方帮助](https://help.github.com)
[一张描述GitHub操作的简图](/papers/git_cheat_sheet.pdf)

待更新...
