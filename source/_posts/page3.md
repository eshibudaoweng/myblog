---
title: Neural Activation Constellations-Unsupervised Part Model Discovery with Convolutional Networks
date: 2017-04-06 10:59:05
tags:
- fine-grained
- papers
---
[2015_iccv_Neural Activation Constellations:Unsupervised Part Model Discovery with Convolutional Networks](/papers/2015_iccv_Neural Activation Constellations- Unsupervised Part Model Discovery with Convolutional Networks.pdf)提出了一种基于部件模型的利用CNN来无监督提取差异斑块的神经激活簇。自己的[翻译](/papers/page3.pdf)。

<!-- more -->
下面结合程序简要说一下这边论文的Algorithm：
1.Extract parts
> 如果部件在图像中是可见的，那么由部件提案直接决定部件的定位；否则，应用贝叶斯准则预测部件的位置分布，计算部件的平均特征值。
```java
if exist(opts.part_loc_file)
    fprintf('Loading part locs from %s\n',opts.part_loc_file);
    load(opts.part_loc_file,'part_locs');
else
    fprintf('Calculating part locations for all channels and all images...\n');
    fprintf('IMPORTANT: This will take quite some time, especially with large nets like VGG19\n' );
    part_locs = parts_locs_from_grads(opts);
end
```
2.Learn part model
> 使用训练模型来学习部件模型，一般来说，对于每一类分别学习到一个模型，部件模型迭代5次，学习到的部件模型使用了5个视角，每个视角选择使用频率最高的10个部件。
```java
if exist(opts.caffe_part_model)
    fprintf('Loading part model from cache...\n');
    load(opts.caffe_part_model,'channel_ids','part_visibility');
else
    fprintf('Calculating part model...\n');
    [ channel_ids, part_visibility] = evaluate_part_locs_anchor_multiview(part_locs, load(opts.tr_ID_file), ...
            load(opts.labels_file), opts.no_selected_parts, opts.no_visible_parts, opts.view_count, opts.iterations);
    save(opts.caffe_part_model,'channel_ids','part_visibility');
end
```
3.Classification framework
> 给定被选择部件的预测定位之后，以每个部件为中心剪裁正方形边框计算它的特征。如果parts不可见，使用图像的平均特征作为代替。这样做的原因是为了后边的微调，因为特定领域的数据集通常很小，因此CNN容易发生过拟合，如果训练图像被剪裁成感兴趣的斑块，可以有效防止过拟合。
```java
if (~opts.estimate_bbox || exist(opts.est_bbox_file)) && exist(opts.caffe_window_file_train) && ...
        exist(opts.caffe_window_file_val)
    fprintf('Loading estimated bounding boxes and region proposals from disk...\n');
elseif ~opts.use_bounding_box && ~opts.finetuning
    fprintf('Region proposals for fine-tuning or estimated bboxes are not needed, skipping...\n');
else
    fprintf('Generating region proposals and estimated bounding boxes for CNN finetuning...\n');
    selsearch_object_detector( channel_ids(1:opts.no_selected_parts), part_locs, part_visibility, opts );
end
```
4.Fine-tuning
5.Classification
```java
fprintf('Starting classification...\n');
part_box_classification_multiscale( channel_ids(1:opts.no_selected_parts), part_locs, opts );
```
Thanks:)
