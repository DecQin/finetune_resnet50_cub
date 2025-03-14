import logging
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

class ModelEmaV2(nn.Module):
    """
    Model Exponential Moving Average V2

    这个类用于维护模型状态字典（包括参数和缓冲区）中所有内容的指数移动平均值。
    V2版本相对更简单，它不基于名称来匹配参数/缓冲区，而是简单地按顺序迭代。它可以与torchscript（整个模型的JIT编译）一起工作。

    该功能旨在允许实现类似以下TensorFlow功能的功能：
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    在一些训练方案中，为了使训练效果良好，需要使用权重的平滑版本。例如，谷歌在训练MNASNet、MobileNet-V3、EfficientNet等网络时使用的超参数，采用了RMSprop优化器，其衰减周期较短（2.4 - 3个周期）且学习率衰减率较慢（0.96 - 0.99），这就需要对权重进行EMA平滑处理才能匹配相应的结果。使用时要注意相对于每个周期的更新次数所使用的衰减常数。

    为了避免EMA占用GPU资源，可以设置device='cpu'。这样做会节省一些内存，但会禁用对EMA权重的验证。验证工作需要在单独的进程中手动进行，或者在训练停止收敛之后进行。

    此类在模型初始化、GPU分配以及分布式训练包装器的顺序中，其初始化位置比较敏感。
    """
    def __init__(self, model, decay=0.9999, device=None):
        """
        类的初始化函数

        参数:
        - model: 要进行EMA操作的原始模型实例
        - decay: 指数移动平均的衰减率，控制着旧的EMA值和新的模型参数值之间的权重，默认值为0.9999
        - device: 可选参数，指定在哪个设备（如'cpu'或'cuda:0'等）上执行EMA操作，如果为None，则与原始模型在相同设备上执行，默认值为None
        """
        super(ModelEmaV2, self).__init__()
        # 对传入的模型进行深拷贝，用于累积权重的移动平均值
        # 这样做是为了创建一个独立的副本，后续在这个副本上更新EMA权重，而不影响原始模型
        self.module = deepcopy(model)
        # 将这个用于EMA的模型副本设置为评估模式，因为在计算EMA时不需要进行梯度计算和训练相关操作
        self.module.eval()
        self.decay = decay
        self.device = device  # 若设置了该设备，则在与原始模型不同的设备上执行EMA操作
        if self.device is not None:
            # 如果指定了设备，将EMA模型副本移动到指定设备上
            self.module.to(device=device)

    def _update(self, model, update_fn):
        """
        内部的更新函数，用于实际更新EMA模型的参数和缓冲区

        参数:
        - model: 原始的模型实例，提供最新的参数值用于更新EMA
        - update_fn: 一个函数，用于定义如何根据旧的EMA值（e）和新的模型参数值（m）来计算更新后的EMA值，通常是一个按照EMA计算规则定义的lambda函数

        此函数在不计算梯度的情况下（因为是更新EMA权重，不需要反向传播），遍历EMA模型副本和原始模型的状态字典中的每一个对应的值（参数和缓冲区），按照给定的更新函数进行更新。
        如果指定了不同的设备，还会将原始模型的参数值移动到对应的设备上，确保在同一设备上进行计算更新。
        """
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        """
        更新EMA模型的参数

        参数:
        - model: 原始的模型实例，提供最新的参数值用于更新EMA

        首先判断传入的模型是否是经过DataParallel包装后的模型，如果是，则获取其内部的实际模型模块（因为DataParallel会对模型进行包装，实际参数在其内部的.module属性中）。
        然后调用内部的_update函数，使用定义好的指数移动平均计算规则（lambda函数）来更新EMA模型的参数，即按照 self.decay * e + (1. - self.decay) * m 的方式更新，其中e是旧的EMA值，m是新的模型参数值。
        """
        if isinstance(model, nn.DataParallel):
            model = model.module
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        """
        直接将EMA模型的参数设置为与提供的模型参数相同

        参数:
        - model: 原始的模型实例，其参数将被用来直接设置EMA模型的参数

        调用内部的_update函数，通过传入一个特定的lambda函数（lambda e, m: m），将EMA模型的参数直接替换为传入模型的参数，实现将EMA参数设置为与提供的模型参数相同的功能。
        """
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, x):
        """
        前向传播函数

        参数:
        - x: 输入的数据张量

        当使用这个EMA模型实例进行前向传播时（例如在验证阶段使用EMA模型来获取预测结果），调用内部保存的EMA模型副本的前向传播函数，返回相应的输出结果，就如同使用普通模型进行前向传播一样。
        """
        return self.module.forward(x)