---
title: Start
date: 2017-03-15 16:47:02
tags:
- Hexo
- mac
---

### 安装Hexo
安装git

```
$ brew install git
```
安装Node.js
```
$ wget -qO- https://raw.github.com/creationix/nvm/master/install.sh | sh
```
<!-- more -->
重启终端并执行下列命令
```
$ nvm install stable
```
所有必备的应用程序安装完成后，即可使用 npm 安装 Hexo
```
$ npm install -g hexo-cli
```

### 建站
安装 Hexo 完成后，执行下列命令，Hexo 将会在指定文件夹中新建所需要的文件。
```
$ hexo init <folder>
$ cd <folder>
$ npm install
```
### 配置

#### **hexo+git** 在服务器上配置
创建非root用户账号
```
$ addusr git
```
为git用户添加sudo 权限
```
$ gpasswd -a git sudo
```
安装git
```bash
$ sudo apt-get install git-core
```
安装apache2
这里已经安装，不在描述

配置git
```
$ mkdir hexo.git
$ cd hexo.git
$ git init --bare
```
创建网站目录
```
$ cd /var/www
$ mkdir hexo
```

变更拥有者
```
$ chown git:git -R /var/www/hexo
```
配置 Git Hooks
```
cd ~/hexo.git/hooks
```
编辑post-receive文件：
```bash
#!/bin/bash
GIT_REPO=/home/git/hexo.git #git 仓库
TMP_GIT_CLONE=/tmp/hexo
PUBLIC_WWW=/var/www/hexo #网站目录 apache2中已经配置了
rm -rf ${TMP_GIT_CLONE}
git clone $GIT_REPO $TMP_GIT_CLONE
rm -rf ${PUBLIC_WWW}/*
cp -rf ${TMP_GIT_CLONE}/* ${PUBLIC_WWW}
```
#### **hexo+git** 在本机上的配置
创建公钥和母钥
```
$ cd ~/.ssh
$ ssh-keygen -t rsa
```
生成
```
id_rsa      id_rsa.pub
```
安装 ssh-copy-id
```
$ brew install ssh-copy-id
```
ssh-copy-id命令可以把本地的ssh公钥文件安装到远程主机对应的账户下
执行如下命令
```
$ ssh-copy-id user@host
```
其中将user替换为自己服务器用户名，host替换为对应的ip地址。通过此命令可以将本地的ssh公钥发送到目标主机上，然后登陆主机账户即可免密码登陆。我的执行如下
```
$ ssh-copy-id git@www.oopy.org
```
#### 修改 deploy 参数
![add1](/images/add1.png)

### 写作
1.创建一篇新的文章
```
$ hexo new myblog
```
2.生成静态文件
```
$ hexo generate
```
3.生成静态文件后部署网站
```
$ hexo deploy
```

More into : [Hexo](https://hexo.io/)

## reference:
- https://blog.yizhilee.com/post/deploy-hexo-to-vps/
- http://www.dullong.com/deploy-hexo-blog-to-my-own-server.html
