---
title: pagetest
date: 2017-03-20 22:13:00
tags:
- web
- design
---

## 博客主题配置修改记录

自行fork修改,[Yilia源码目录结构及构建须知](https://github.com/litten/hexo-theme-yilia/wiki/Yilia%E6%BA%90%E7%A0%81%E7%9B%AE%E5%BD%95%E7%BB%93%E6%9E%84%E5%8F%8A%E6%9E%84%E5%BB%BA%E9%A1%BB%E7%9F%A5)

<!-- more -->

### 添加Gitment评论系统
> 感谢[imsun同学](https://imsun.net/)的[gitment插件](https://imsun.net/posts/gitment-introduction/)。
基本上按照作者给的文档就可以搭建成功，这里简单地给出配置过程和几个细节性的问题。




1. 注册 [OAuth Application](https://github.com/settings/applications/new)
callback URL中填写评论页面对应的域名，如我的是www.dearkai.com，得到一个 client ID 和一个 client secret，这个将被用于之后的用户登录。
2. 在yilia主题中引入Gitment
- 在`<mysite>/themes/yilia/_config.yml`中添加gitment主题,具体内容在我的GitHub中可以查看到。
- 在`<mysite>/themes/yilia/layout/_partial/article.ejs`中t添加页面修改。
- 在`<mysite>/themes/yilia/layout/_partial/post/gitment.ejs`添加如下代码
```
<div id="container"></div>
<link rel="stylesheet" href="https://imsun.github.io/gitment/style/default.css">
<script src="https://imsun.github.io/gitment/dist/gitment.browser.js"></script>
<script>
var gitment = new Gitment({
  id: '页面 ID', // 可选。默认为 location.href
  owner: '你的 GitHub ID',
  repo: '存储评论的 repo',
  oauth: {
    client_id: '你的 client ID',
    client_secret: '你的 client secret',
  },
})
gitment.render('container')
</script>
```
3. 初始化评论
添加之后，我们在Hexo中生成并上传静态页面，可以看到在博客的下方已经有了gitment插件。我们登陆自己的github账号，可以看到一个蓝色的按钮，点击它就可以初始化该页面的评论框。如下图
![屏幕快照 2017-05-31 下午5.08.10](/images/屏幕快照 2017-05-31 下午5.08.10.png>)
4. 结束
可以测试一下，emmmmmmmm……没毛病
![屏幕快照 2017-05-31 下午5.06.53](/images/屏幕快照 2017-05-31 下午5.06.53.png>)





### 打赏的背景图颜色
在/themes/yilia/source-src/css/tooltip.scss这一路径里修改。在tooltip-inner封装中。

### 代码块
行内`代码`块测试,改成pink哒。在路径/themes/yilia/source-src/css/article.scss

### 添加RSS
```
npm install hexo-generator-feed --save
```
<!-- more -->

### 添加sitemap
```
npm install hexo-generator-sitemap --save
```




待更新...
