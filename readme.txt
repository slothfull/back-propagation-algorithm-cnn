# [项目目的] 
  通过使用matlab语言实现神经网络正向传播、反向传播算法的实现，加深对bp优化算法在神经网络参数优化中应用的理解。


# [关于反向传播算法的理论推导] 
  关于代码中涉及的神经网络反向传播算法的推导 可以参考github中维护的博客中内容进行理解：
  https://twistfatezz.github.io/documentation/2020/03/16/post-backpropagation/
  补充说明：由于github代理原因 部分多行形式的Latex代码可能不能正常渲染出来 可以尝试将Latex公式粘贴到相应解释器进行查看


# [文件说明] 总共含有6个 .m 文件
  CNN_upweight.m:   -> 神经网络backprogation误差反向传播推演函数 用于推导每次训练时 后一层的误差反向传播到前一层后的大小
  LeNet_5.m         ->  + LeNet_5的正向inference网络结构的定义
                        + MINST数据集的预加载 
                        + 反向传播算法的调用(神经网络的训练)
                        + 神经网络在测试数据集上的测试(仅涉及inference)
  convolution.m     ->  2d卷积操作的inference实现 
  convolution_f1.m  ->  LeNet_5中全连接层(fully-connected lay)的inference实现
  init_kernel.m     ->  5x5 以及 12x12 尺寸的神经网络卷积算子初始化函数 (注意卷积算子在初始化时需要进行归一化)
  mnist.mat         ->  MNIST公知手写数字数据集本地文件
  pooling.m         ->  LeNet网络结构中池化层的实现 (抑制过拟合，神经网络参数稀疏化)


# [数据配置] 
  采用公知MNIST数据库中的数据进行训练和测试


# [运行方式] 
  1 将所有 .m 文件放在同一目录下 使用matlab(>2016) 打开 LeNet_5.m 源程序文件
  2 修改路径参数：将load()函数中路径部分替换成 神经网络代码训练&测试用到的数据集 文件夹中的.mat数据文件存放的路径即可:
    > total_data=load(‘xxxx/mnist.mat');


# [运行结果] 
  由于Lenet5模型的表达能力有限，经过200epoch训练之后，在测试集合上可以达到80%的准确度。
  本项目重点不在于确定网络复杂度，修改网络结构，以追求高精度的测试效果。


# [运行时间time profile] 
  训练时间大约需要 8.544975e+02 seconds 测试时间大约需要 8.956718e+00 seconds


# [关于github仓库提交出现问题 -> remote: Support for password authentication was removed on August 13, 2021.]
  cd github settings & cd developer settings & generate new tmp token -> <your token>
  git remote set-url origin  https://<your_token>@github.com/<USERNAME>/<REPO>.git
  git push origin main
