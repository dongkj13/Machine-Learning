## SVC, NuSVC, LinearSVC（支持向量分类）
- SVC和 NuSVC差不多，区别仅仅在于对损失的度量方式不同，而LinearSVC从名字就可以看出，他是线性分类，也就是不支持各种低维到高维的核函数，仅仅支持线性核函数，对线性不可分的数据不能使用。
- 如果有经验知道数据是线性可以拟合的，那么使用LinearSVC去分类 或者LinearSVR去回归，它们不需要我们去慢慢的调参去选择各种核函数以及对应参数， 速度也快。如果我们对数据分布没有什么经验，一般使用SVC去分类或者SVR去回归，这就需要我们选择核函数以及对核函数调参了。
- 如果我们对训练集训练的错误率或者说支持向量的百分比有要求的时候，可以选择NuSVC分类。它们有一个参数来控制这个百分比。

## SVR, NuSVR, LinearSVR（支持向量回归）
[支持向量机原理(五)线性支持回归](http://www.cnblogs.com/pinard/p/6113120.html)

## 参考资料

[scikit-learn 支持向量机算法库使用小结](https://www.cnblogs.com/pinard/p/6117515.html)

[支持向量机高斯核调参小结](http://www.cnblogs.com/pinard/p/6126077.html)
