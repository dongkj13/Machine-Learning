## 最小二乘法模型
```python
regr = linear_model.LinearRegression()
```
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_{\theta}(x_i) - y_i)^2
```

## Ridge回归
```python
regr = linear_model.Ridge(alpha = 0.1)
```
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_{\theta}(x_i) - y_i)^2 + \alpha \sum_{i=1}^n \theta_i^2
```
## Lasso回归
```python
regr = linear_model.Lasso(alpha = 0.1)
```
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_{\theta}(x_i) - y_i)^2 + \alpha \sum_{i=1}^n ||\theta_i||
```
## ElasticNet回归
```python
regr = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
```
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_{\theta}(x_i) - y_i)^2 + \alpha_1 \sum_{i=1}^n \theta_i^2 + \alpha_2 \sum_{i=1}^n ||\theta_i||
```
## 交叉验证
```python
regr = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
regr = linear_model.LassoCV(alphas=[0.1, 1.0, 10.0])
regr = linear_model.ElasticNetCV(alphas=[0.1, 1.0, 10.0])
```
通过设置cv参数实现不同的验证方式（留一验证，K折验证）

## 参考资料

[线性模型（一）--广义线性模型（GLM）简介](http://blog.csdn.net/Fleurdalis/article/details/54864405)

[线性模型（二）-- 线性回归公式推导](http://blog.csdn.net/Fleurdalis/article/details/54931721)

[线性模型（二）－－对线性回归的几点思考](http://blog.csdn.net/Fleurdalis/article/details/54953573)

[线性模型（三）－－ridge、lasso、ElasticNet回归](http://blog.csdn.net/fleurdalis/article/details/55059516)

[机器学习-第3周，岭回归和lasso](http://f.dataguru.cn/thread-598486-1-1.html)