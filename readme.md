# 期中大作业设想

图像风格分类/迁移

创新部分：多图风格特征提取/语义分割风格迁移
多图像生成模板

## 参考

[原论文](https://arxiv.org/pdf/1508.06576)
[知乎讲解](https://zhuanlan.zhihu.com/p/366949989)
[csdn卷积神经网络示例](https://blog.csdn.net/qq_42589613/article/details/142553129)
[csdn原理展示](https://blog.csdn.net/weixin_66526635/article/details/135549097)
[快速迁移讲解](https://www.cnblogs.com/Anita9002/p/9115757.html)
[csdn发展历史沿袭简介](https://blog.csdn.net/weixin_43783814/article/details/132418279)

步骤：

- 数据预处理：修改分辨率统一图片大小，增加batch(处理单张图片时也加一个维度符合tensorflow模型的适配)
- 训练输入输出模型
