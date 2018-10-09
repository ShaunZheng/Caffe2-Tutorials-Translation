# Intro Tutorial


## Caffe2 Concepts

您可以在下面了解更多关于Caffe2的主要概念，这些概念对理解和开发Caffe2模型至关重要。

### Blobs and Workspace, Tensors#

Caffe2 中的数据被组织为 blob。 blob 只是内存中一个被命名的数据块。大多数 blob 包含一个张量（想想多维数组），在 Python 中它们被转换为 numpy 数组（numpy 是一个流行的Python数值库，已经作为 Caffe2 的先决条件安装）。

一个工作区存储所有 blob。以下示例显示如何将 Blob 提供到（feed）工作空间并再次获取（fetch）它们。在您开始使用它们的那一刻，工作区就已初始化。

```
from caffe2.python import workspace, model_helper
import numpy as np
# Create random tensor of three dimensions
x = np.random.rand(4, 3, 2)
print(x)
print(x.shape)

workspace.FeedBlob("my_x", x)

x2 = workspace.FetchBlob("my_x")
print(x2)
```

### Nets and Operators

Caffe2 中的基本的模型抽象是 net（ network 的简称）。一个 net 是一个计算图，每个运算符采用一组输入 blob 并生成一个或多个输出 blob。

在下面的代码块中，我们将创建一个超级简单的模型。它将包含以下组件：

- 一个全连接层（FC）
- 一个带有 Softmax 的 Sigmoid 激活函数
- 一个交叉熵 loss

直接编写网络非常繁琐，因此最好使用有助于创建网络的Python类模型助手。即使我们调用它并传入单个名称“我的第一个网络”， ModelHelper 也会创建两个相互关联的网络：

1. 一个是用来初始化参数（参考init_net）
2. 一个是用来运行实际的训练（参考exec_net）

```
# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)
```

我们创建了一些随机数据和随机标签，然后将它们作为blob提供给工作区。

```
# Create model using a model helper
m = model_helper.ModelHelper(name="my first net")
```

您现在已经使用 model_helper 创建了我们之前提到的两个网络（init_net 和 exec_net）。下一步，我们计划在此模型中使用 FC 运算符添加全连接层，但首先我们需要通过创建为权重和偏差随机填充的 blob 来进行准备工作，以获得期望的 FC op。（翻译可能有瑕疵）当我们添加 FC op 时，我们将通过引用权重和偏置的 blob 的名称，并将该名称作为输入。

```
weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])
```

在 Caffe2 中，FC op 接收输入 blob（我们的数据），权重和偏差。使用 XavierFill 或 ConstantFill 的权重和偏差都将采用空数组，名称和形状（如 shape = [output，input]）。

```
fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
pred = m.net.Sigmoid(fc_1, "pred")
softmax, loss = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

```


查看上面的代码块：

首先，我们在内存中创建了输入数据和标签 blob（实际上，您将从输入数据源（如数据库）加载数据）。请注意，数据和标签 blob 的第一个维度为 “16”;这是因为模型的输入是一次只有 16 个样本的小批量。许多 Caffe2 运算符可以通过 ModelHelper 直接访问，并且可以一次处理一小批输入。


其次，我们通过定义一组运算符来创建模型：FC，Sigmoid 和 SoftmaxWithLoss。注意：此时，运算符未执行，您只是创建模型的定义。

Model helper 将创建两个 net：m.param_init_net，这是一个只运行一次的 net。它将初始化所有参数blob，例如FC层的权重。实际训练是通过执行 m.net 完成的。这对您来说是透明的，并且会自动发生。

网络定义存储在protobuf结构中（有关详细信息，请参阅 [Google’s Protocol Buffer documentation](https://developers.google.com/protocol-buffers/)）。您可以通过调用net.Proto（）轻松检查它：

```
print(m.net.Proto())
```

输出应如下所示：

```
name: "my first net"
op {
  input: "data"
  input: "fc_w"
  input: "fc_b"
  output: "fc1"
  name: ""
  type: "FC"
}
op {
  input: "fc1"
  output: "pred"
  name: ""
  type: "Sigmoid"
}
op {
  input: "pred"
  input: "label"
  output: "softmax"
  output: "loss"
  name: ""
  type: "SoftmaxWithLoss"
}
external_input: "data"
external_input: "fc_w"
external_input: "fc_b"
external_input: "label"
```

你还应该看一下 param 的初始化 net：

```
print(m.param_init_net.Proto())
```

您可以看到这里的两个运算符如何为 FC 运算符的权重和偏差 blob 创建随机填充。

这是 Caffe2 API 的主要思想：使用 Python 方便地组合网络来训练你的模型，将这些网络传递给 `C++` 代码作为序列化的 protobuffers，然后使用 C++ 代码以最高的性能运行网络。


### Executing


现在，当我们定义好模型训练运算符，我们可以开始运行它来训练我们的模型。

首先，我们只运行一次 param 初始化：

```
workspace.RunNetOnce(m.param_init_net)
```

请注意，像往常一样，这实际上会将 param_init_net 的 protobuffer 传递给 C++ 运行时以便执行。


然后我们创建实际的训练 net：

```
workspace.CreateNet(m.net)
```


我们创建它一次，然后我们可以多次有效地运行它：

```
# Run 100 x 10 iterations
for _ in range(100):
    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)

    workspace.FeedBlob("data", data)
    workspace.FeedBlob("label", label)

    workspace.RunNet(m.name, 10)   # run for 10 times
```

注意我们如何将 m.name 而不是 net 定义本身传递给 RunNet（）。由于网络是在工作空间内创建的，因此我们不需要再次传递定义。


执行后，您可以检查存储在输出blob（包含张量，即numpy数组）中的结果：

```
print(workspace.FetchBlob("softmax"))
print(workspace.FetchBlob("loss"))
```


### Backward pass

这个网络只包含正向传播，因此它不会学习任何东西。通过在正向传递中为每个运算符添加梯度运算符来创建向后传递（反向传播）。

如果您想尝试此操作，请添加以下步骤并检查结果！

在调用 RunNetOnce() 之前插入：

```
m.AddGradientOperators([loss])
```

检查protobuf输出：

```
print(m.net.Proto())
```

这里是总结概述，你可以在[教程](https://caffe2.ai/docs/tutorials.html)中学到很多内容。




