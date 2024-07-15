import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class VDMConfig:
    """VDM configurations. 对模型的具体配置 """
    # 噪声调度的配置
    gamma_type: str
    gamma_min: float
    gamma_max: float


#########严格单调递增的全连接层用于产生噪声 #########
class DenseMonotone(nn.Dense):
    """严格单调递增的全连接层."""

    # 这个就像是噪声网络中的用来产生噪声的了
    @nn.compact
    def __call__(self, inputs):
        #  将输入转为数组 并且将数据类型完成转换
        inputs = np.asarray(inputs, dtype=self.dtype)
        #  初始化权重矩阵
        kernel = self.param('kernel',
                            self.kernel_init,
                            (inputs.shape[-1], self.features))
        #  保证所有到的权重非负 从而实现 噪声的单调
        kernel = abs(np.asarray(kernel, self.dtype))
        # 这里暂时存在问题 如果y是二维数组y = np.dot(inputs, kernel)
        # 如果不是要根据 具体的input以及kernel的维度来进行修改
        # 计算 inputs 的最后一个维度的索引
        # last_dim_index = len(inputs.shape) - 1
        #
        # # 使用 np.tensordot 沿着 inputs 的最后一个维度和 kernel 的第一个维度进行矩阵乘法
        # y = np.tensordot(inputs, kernel, axes=([last_dim_index], [0]))
        y = np.tensordot(inputs, kernel, axes=((-1,), (0,)), dtype=np.float16)
        # , dtype = self.precision 这个自动精度控制无法实现 因为不知道要的精度是多少
        # 暂时设为float 32
        # y = jax.lax.dot_general(inputs, kernel,
        #                         (((inputs.ndim - 1,), (0,)), ((), ())),
        #                         precision=self.precision)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,))
            bias = torch.as_tensor(bias, ).to(self.dtype)
            y = y + bias
        return y


######### 三种 噪声计划 用来控制扩散过程中的噪声水平/信噪比 #########

'''
NoiseSchedule_FixedLinear 和 NoiseSchedule_Scalar 都提供了线性的噪声增长模型，
但 NoiseSchedule_Scalar 通过引入可学习的参数，为模型提供了额外的优化能力和适应性。
这种适应性使得 NoiseSchedule_Scalar 在训练过程中可以根据数据的特征进行更精细的调整
'''


class NoiseSchedule_NNet(nn.Module):
    # 一个可学习的网络 就是文中提到的 三层权重限制为正的单调网络
    config: VDMConfig
    n_features: int = 1024  # 第二层上有1024个输出
    nonlinear: bool = True  # 决定网络是否包含非线性变换

    def setup(self):
        # 类的初始化
        config = self.config

        n_out = 1
        kernel_init = nn.initializers.normal()

        '''
        这是在DIFUSCO中线性扩散的噪声比例的起始以及结束
            b0 = 1e-4 (0.0001)
            bT = 2e-2 (0.02)
            这里只是一个测试值 后续可以调整
        '''
        init_bias = 1e-4 # 噪声的初始水平
        init_scale = 0.999 - init_bias  # 噪声的范围

        # DenseMonotone代表全连接层
        self.l1 = DenseMonotone(n_out,
                                kernel_init=constant_init(init_scale),
                                bias_init=constant_init(init_bias))  # 第一层 输出为1维 用于生成初始噪声水平
        if self.nonlinear:
            self.l2 = DenseMonotone(self.n_features, kernel_init=kernel_init)  # 第二层 引入非线性 将数据映射到1024维
            self.l3 = DenseMonotone(n_out, kernel_init=kernel_init, use_bias=False)  # 第三层 引入非线性 重新映射回1 维

    @nn.compact
    def __call__(self, t, det_min_max=False):
        '''
        如果输入 t 是一个标量或零维数组，最终输出的 h 将是一个形状为 (1,) 的一维数组。
        如果输入 t 是一个一维数组，表示批量中的时间步，最终输出的 h 将是一个形状为 (batch_size,) 的一维数组，其中 batch_size 是输入 t 数组的长度。
        包含了对应每个时间步的噪声水平估计值

        如果输入 t 是一个标量（即一个单一的数值，没有数组形状），则输出 h 的形状将是 (1,)。
        这表示输出是一个长度为 1 的一维数组，其中包含了根据输入标量 t 计算得出的单一噪声水平估计值。
        如果输入 t 是一个零维数组（在 NumPy 中，零维数组可以被视为一个单独的点或元素，它没有形状），则输出 h 的形状同样将是 (1,)。
        如果输入 t 是一个一维数组（例如，包含了一批时间步的信息），则输出 h 的形状将是 (batch_size,)，
        其中 batch_size 是输入数组 t 的长度。这意味着输出是一个一维数组，其长度与输入数组 t 中的元素数量相同，每个元素对应一个基于时间步 t 的噪声水平估计。

        Args:
            t: 时间步
            det_min_max:

        Returns:噪声数组h

        '''
        # 声明 输入t的类型支持标量、零维和一维输入
        # 这里待修改
        assert (np.isscalar(t) or np.ndim(t) == 0 or np.ndim(t) == 1), "Input t must be a scalar, 0-dimensional, or 1-dimensional."

        # 将标量或零维数组的t 分别通过乘以（1，1）数组和reshape变为一个二维数组
        '''
        将输入数据扩充成二维数组（通常是矩阵形式）是为了满足全连接层（Dense layers）的输入要求。
        全连接层期望输入具有固定的维度，这通常包括两个维度：一个批量大小维度（batch size）和一个特征维度。
        即使批量大小为1，这种二维表示也是必需的，因为它允许使用矩阵乘法来高效地实现层的计算。
        '''
        if np.isscalar(t) or len(t.shape) == 0:
            t = t * np.ones((1, 1))
        else:
            t = np.reshape(t, (-1, 1))

        h = self.l1(t)  # 通过第一层 h.shape ==(-1,1)
        if self.nonlinear:  # 如果启用非线性变换
            _h = 2. * (t - .5)  # 将输入缩放至 [-1, +1]
            _h = self.l2(_h)
            _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
            _h = self.l3(_h) / self.n_features
            h += _h  # 非线性层的输出 _h 加到初步噪声水平估计 h 上，合并为最终的噪声计划输出
        #   这就实现了一个学习时间步 t 与噪声水平之间复杂关系的模型

        beta = np.clip(h, 0, 1)
        beta = np.squeeze(beta, axis=-1)
        # 这里对最后的输出进行了修改 适配DIFUSCO
        return beta


def constant_init(value, dtype='float32'):
    '''创建具有恒定值的初始化操作 用于NN中的W和b的初始化

    Args:
      value: 用于初始化的值
      dtype: 输出数组的类型

    Returns:创建了一个所有元素都是 value 的数组。

    '''

    def _init(key, shape, dtype=dtype):
        return value * np.ones(shape, dtype)

    return _init
