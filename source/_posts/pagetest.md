---
title: pagetest
date: 2017-03-20 22:13:00
tags:
- web
- design
---

## 博客主题配置修改记录

自行fork修改,[Yilia源码目录结构及构建须知](https://github.com/litten/hexo-theme-yilia/wiki/Yilia%E6%BA%90%E7%A0%81%E7%9B%AE%E5%BD%95%E7%BB%93%E6%9E%84%E5%8F%8A%E6%9E%84%E5%BB%BA%E9%A1%BB%E7%9F%A5)

- 打赏的背景图颜色
在/themes/yilia/source-src/css/tooltip.scss这一路径里修改。在tooltip-inner封装中。

- 行内`代码`块测试,改成pink哒。在路径/themes/yilia/source-src/css/article.scss

- 添加RSS
```
npm install hexo-generator-feed --save
```
<!-- more -->

- 添加sitemap
```
npm install hexo-generator-sitemap --save
```

待更新...
